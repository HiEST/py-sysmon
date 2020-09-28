import os
import time

import subprocess
import argparse
import glob
import psutil
import shlex

from pathlib import Path

import numpy as np
import pandas as pd
import json
from tqdm import tqdm

from utils.monitors import ResourceMonitor, RAPLMonitor, PCMMonitor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--bin", 
        help="Path to the benchmark app binary",
        type=str,
        default='./bin/benchmark_app'
    )
    parser.add_argument(
        "-d", "--device", 
        help="Device to run the benchmark.",
        type=str,
        default='CPU'
    )
    parser.add_argument(
        "-c", "--config", 
        help="Config file from which configurations of {cores, requests, streams, and batch size} are read.",
        type=str,
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to output directory file where results are stored.",
        default="./results/",
        type=str
    )
    parser.add_argument(
        "--prefix",
        help="Prefix to name the different output files generated.",
        type=str,
        default=None
    )
    parser.add_argument(
        "-t", "--time",
        help="Duration of each experiment.",
        type=int,
        default=None
    )
    parser.add_argument(
        "-i", "--input", 
        help="Path to the input video or directory containing the videos.",
        type=str,
    )
    parser.add_argument(
        "--plugin", 
        help="GST_PLUGIN_PATH (directory where latency pluting is).",
        type=str,
    )
    parser.add_argument(
        "--sync", 
        help="Synchronous decoding (decoding rate locked at input's framerate).",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "--no-latency", 
        help="Skip the second run to get latency.",
        default=False,
        action='store_true'
    )
    
    args = parser.parse_args()

    if '.mp4' in args.input:
        if not os.path.isfile(args.input):
            raise ValueError('{} does not exist.'.format(args.input))
        inputs = [Path(args.input)]
    elif os.path.isdir(args.input):
        inputs = [video for video in Path(args.input).glob('*.mp4')]
    else:
        raise ValueError(
            '{} is not a valid directory nor video file.'.format(args.input)
        )

    cpu_count = psutil.cpu_count(logical=False)

    if args.config is not None:
        configs = pd.read_csv(args.config)
    else:
        default_config = [[cpu_count, 1]]
        configs = pd.DataFrame(default_config, columns=['cores', 'procs'])

    #System monitors
    cpu_monitor = ResourceMonitor()
    rapl_monitor = RAPLMonitor()
    pcm_monitor = PCMMonitor()

    proc = psutil.Process()
    benchmark_stats = []
    benchmark_metrics = ['Video', 'CPUs', 'Procs', 'Device', 'Sync', 'Codec', 'Resolution', 'Throughput']
    benchmark_metrics += ['Latency Avg', 'Latency Max', 'Latency Min', 'Latency 95%', 'Latency 99%', 'Latency Median']

    configs_per_video = len(configs)
    if not args.no_latency:
        configs_per_video = configs_per_video * 2
    total_runs = len(inputs) * configs_per_video

    gst_pipeline = 'gst-launch-1.0 {} filesrc location={} ! qtdemux ! {} {}parse ! {} {} ! fakesink sync={}'    
    ffprobe = 'ffprobe -loglevel quiet -print_format json -show_format -show_streams {}'

    with tqdm(total=total_runs, desc="Total runs") as pbar:
        for video in tqdm(inputs, desc='Videos to decode', total=len(inputs)):
            video_name = video.stem

            ffproc = subprocess.Popen(
                shlex.split(ffprobe.format(str(video))),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            out, err = ffproc.communicate()
            video_info = json.loads(out)

            codec = video_info['streams'][0]['codec_name']
            resolution = video_info['streams'][0]['height']
            fr, div = video_info['streams'][0]['avg_frame_rate'].split('/')
            frame_rate = float(fr) / float(div)
            bitrate = int(video_info['streams'][0]['bit_rate'])
            # number_frames = int(video_info['streams'][0]['nb_frames'])

            if codec == 'hevc':
                codec = 'h265'

            if args.device == 'cpu':
                decoder = f'avdec_{codec}'
            else:
                decoder = f'vaapi{codec}dec'

            gst_throughput = gst_pipeline.format('', str(video), '', codec, decoder, '', str(args.sync))
            gst_latency = gst_pipeline.format(f'--gst-plugin-path={args.plugin}', str(video), 'markin name=moo !', codec, decoder, '! markout', str(args.sync))

            for _, cores, procs in tqdm(configs.itertuples(), desc='Runs with {}'.format(video_name), total=configs_per_video, leave=False):
                if not isinstance(cores, int):
                    cores = cpu_count
                if not isinstance(procs, int):
                    procs = 1

                proc.cpu_affinity(list(np.arange(0, cores)))

                pcm_monitor.start(interval=1.0)
                cpu_monitor.start(interval=1.0)
                rapl_monitor.start(interval=1.0)

                # 1. First run to get throughput and telemetry
                t0 = time.time()
                subproc = subprocess.Popen(
                    shlex.split(gst_throughput),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

                # If synchronous decoding, gst has to finish or we won't be able to get performance metrics
                timeout = None
                if args.time is not None:
                    timeout = args.time

                out, err = subproc.communicate(timeout=timeout)
                t1 = time.time()

                cpu_monitor.stop(checkpoint=True)
                rapl_monitor.stop(checkpoint=True)
                pcm_monitor.stop(checkpoint=True)

                runtime = None
                for line in out.decode('utf-8').split('\n')[-10:]:
                    if 'Execution ended after' in line:
                        runtime = line.split(' ')[-1]
                        hours, minutes, seconds = runtime.split(':')
                        runtime = int(hours)*3600 + int(minutes)*60 + float(seconds)
                
                if runtime is None:
                    print('out: ')
                    print(out.decode('utf-8'))
                    print('error: ')
                    print(err.decode('utf-8'))
                    print(gst_throughput)
                    
                relative_speed = float(video_info['streams'][0]['duration']) / runtime
                decoding_fps = frame_rate * relative_speed

                # 2. Second run to only get latency, unless otherwise specified
                if not args.no_latency:
                    pbar.update(1)

                    os.environ['GST_DEBUG'] = 'markout:5'
                    subproc = subprocess.Popen(
                        shlex.split(gst_latency),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    out, err = subproc.communicate(timeout=timeout)
                    err = err.decode('utf-8')
                    frame_latencies = []
                    for line in err.split('\n'):
                        if 'Mark Duration' not in line:
                            continue
                        duration = float(line.split(':')[-1].strip().replace('ms',''))
                        frame_latencies.append(duration)

                    del os.environ['GST_DEBUG']
                
                    lat = np.array(frame_latencies)
                    latency_stats = [
                        lat.mean(),
                        lat.min(),
                        lat.max(),
                        np.percentile(lat, 95),
                        np.percentile(lat, 99),
                        np.median(lat)
                    ]
                else:
                    latency_stats = [0, 0, 0, 0, 0, 0]


                stats = [video_name, cores, procs, args.device, args.sync, codec, bitrate, resolution, decoding_fps ] + latency_stats
                benchmark_stats.append(stats)

                pbar.update(1)
                
    benchmark_metrics = ['Video', 'CPUs', 'Procs', 'Device', 'Sync', 'Codec', 'Bitrate', 'Resolution', 'Throughput']
    benchmark_metrics += ['Latency Avg', 'Latency Max', 'Latency Min', 'Latency 95%', 'Latency 99%', 'Latency Median']
    df = pd.DataFrame(benchmark_stats, columns=benchmark_metrics)

    if args.prefix:
        output_file = args.output + '/' + args.prefix + '-'
    else:
        output_file = args.output + '/'
    
    df.to_csv('{}summary.csv'.format(output_file), sep=',', index=False, float_format='%.3f')
    df = pd.concat([df, cpu_monitor.checkpoints, rapl_monitor.checkpoints, pcm_monitor.checkpoints], axis=1, sort=False)
    df.to_csv('{}detailed.csv'.format(output_file), sep=',', index=False, float_format='%.3f')
    
        
if __name__ == '__main__':
    main()

