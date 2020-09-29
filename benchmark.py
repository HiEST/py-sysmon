import os
import sys
import time

import subprocess
import argparse
import glob
import psutil
import shlex
import logging

from pathlib import Path

import numpy as np
import pandas as pd
import json
from tqdm import tqdm

from utils.monitors import ResourceMonitor, RAPLMonitor, PCMMonitor

MIN_RUNTIME = 5

class PipelineError(Exception):
    """Raised when the gst pipeline won't preroll"""
    pass

def run(cmdline, cores_per_proc=1, procs=1, monitors=None, timeout=None, environ=None, retries=5):
    
    if isinstance(cmdline, str):
        cmdline = shlex.split(cmdline)

    
    if cores_per_proc is None: # Don't set cpu affinity
        cpusets = None
    else:
        cpusets = []
        for i in range(procs):
            cpuid_down = cores_per_proc*i
            cpuid_up = cores_per_proc*(i+1)
            proc_cpuset = list(np.arange(cpuid_down, cpuid_up, 1))
            cpusets.append(proc_cpuset)
    
    while retries > 0:
        try:

            for monitor in monitors:
                monitor.start(interval=1.0)


            subprocs = []
            for i in range(procs):
                if environ is not None:
                    os.environ['GST_DEBUG'] = environ

                subproc = subprocess.Popen(
                    cmdline,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

                if environ is not None:
                    del os.environ['GST_DEBUG']
                    environ = None

                if cpusets is not None:
                    proc = psutil.Process(pid=subproc.pid)
                    proc.cpu_affinity(cpusets[i])
                subprocs.append(subproc)


            error_found = False
            for i in range(procs):
                out, err = subprocs[i].communicate(timeout=timeout)
                out = out.decode('utf-8')
                err = err.decode('utf-8')
                subprocs[i] = [out, err]

                if 'ERROR' in err:
                    error_found = True
                    print(err)
                
            if not error_found:
                for monitor in monitors:
                    monitor.stop(checkpoint=True)
                return subprocs

        except:
            raise
        
        for monitor in monitors:
            monitor.stop(checkpoint=False)
        retries = retries - 1

    # if we reach this point, retries = 0
    raise PipelineError("Command failed all retries: {}".format(' '.join(cmdline)))


def get_inputs(input_path, benchmark):
    if benchmark == 'decoding':
        extensions = ['.mp4', '.webm']
    elif benchmark == 'inference':
        extensions = ['.xml']

    inputs = []
    if any(ext in input_path for ext in extensions):
        if not os.path.isfile(args.input):
            raise ValueError('{} does not exist.'.format(args.input))
        inputs = [Path(input_path)]
    elif os.path.isdir(input_path):
        inputs = []
        for ext in extensions:
            inputs.extend([f for f in Path(input_path).glob('*{}'.format(ext))])
    else:
        raise ValueError(
            '{} is not a valid directory nor a valid input file.'.format(input_path)
        )

    return inputs


def get_video_info(video):
    ffprobe = 'ffprobe -loglevel quiet -print_format json -show_format -show_streams {}'
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
    duration = float(video_info['streams'][0]['duration'])

    if codec == 'hevc':
        codec = 'h265'

    return [codec, resolution, frame_rate, bitrate, duration]

def decoding(args):
    if args.prefix:
        output_file = args.output + '/' + args.prefix + '-'
    else:
        output_file = args.output + '/'

    inputs = get_inputs(args.input, benchmark=args.benchmark)

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
    monitors = [cpu_monitor, rapl_monitor, pcm_monitor]

    proc = psutil.Process()

    benchmark_stats = []
    benchmark_metrics = ['Video', 'CPUs', 'Procs', 'Device', 'Sync', 'Codec', 'Bitrate', 'Resolution', 'Throughput']
    benchmark_metrics += ['Latency Avg', 'Latency Max', 'Latency Min', 'Latency 95%', 'Latency 99%', 'Latency Median', 'Latency StdDev']

    configs_per_video = len(configs)
    if not args.no_latency:
        configs_per_video = configs_per_video * 2
    total_runs = len(inputs) * configs_per_video

    gst_pipeline = 'gst-launch-1.0 {} filesrc location={} ! qtdemux ! {} {}parse ! {} {} ! fakesink sync={}'    

    with tqdm(total=total_runs, desc="Total runs") as pbar:
        for video in tqdm(inputs, desc='Videos to decode', total=len(inputs)):
            video_name = video.stem

            codec, resolution, frame_rate, bitrate, video_duration = get_video_info(video)
            if args.device.lower() == 'cpu':
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

                if cores * procs > cpu_count:
                    cores = None # Don't set cpu affinity

                # 1. First run to get throughput and telemetry
                try:
                    outputs = run(
                        gst_throughput, 
                        cores_per_proc=cores,
                        procs=procs,
                        monitors=monitors, 
                        timeout=args.time
                    )

                except PipelineError as e:
                    logging.warning('Skipping experiment with video {} and {} procs with {} cpus'.format(video_name, procs, cores))
                    continue
                except:
                    # Save work
                    logging.error("Benchmark failed with pipeline: {}".format(gst_latency))
                    logging.error("Saving current work...")
                    df = pd.DataFrame(benchmark_stats, columns=benchmark_metrics)
                    df.to_csv('{}summary.csv.bak'.format(output_file), sep=',', index=False, float_format='%.3f')
                    df = pd.concat([df, cpu_monitor.checkpoints, rapl_monitor.checkpoints, pcm_monitor.checkpoints], axis=1, sort=False)
                    df.to_csv('{}detailed.csv.bak'.format(output_file), sep=',', index=False, float_format='%.3f')
                    raise
                    # raise SystemExit('Exiting benchmark')

                runtimes = []
                throughputs = []
                for output in outputs:
                    out = output[0]
                    err = output[1]

                    runtime = None
                    for line in out.split('\n')[-10:]:
                        if 'Execution ended after' in line:
                            runtime = line.split(' ')[-1]
                            hours, minutes, seconds = runtime.split(':')
                            runtime = int(hours)*3600 + int(minutes)*60 + float(seconds)
                        
                    relative_speed = video_duration / runtime
                    decoding_fps = frame_rate * relative_speed
                    if runtime < MIN_RUNTIME:
                        logging.warning("Runtime is too low for meaningful telemetry (runtime: {})".format(runtime))

                    runtimes.append(runtime)
                    throughputs.append(decoding_fps)

                # Average runtime and throughput
                avg_runtime = np.mean(np.array(runtimes))
                avg_fps = np.mean(np.array(throughputs))

                # 2. Second run to only get latency, unless otherwise specified
                if not args.no_latency:
                    pbar.update(1)

                    try:
                        outputs = run(
                            gst_latency, 
                            cores_per_proc=cores,
                            procs=procs,
                            monitors=monitors,
                            timeout=args.time,
                            environ='markout:5'
                        )
                    except:
                        # Save work
                        logging.error("Benchmark failed with pipeline: {}".format(gst_latency))
                        logging.error("Saving current work...")
                        df = pd.DataFrame(benchmark_stats, columns=benchmark_metrics)
                        df.to_csv('{}summary.csv.bak'.format(output_file), sep=',', index=False, float_format='%.3f')
                        df = pd.concat([df, cpu_monitor.checkpoints, rapl_monitor.checkpoints, pcm_monitor.checkpoints], axis=1, sort=False)
                        df.to_csv('{}detailed.csv.bak'.format(output_file), sep=',', index=False, float_format='%.3f')
                        raise
                        raise SystemExit('Exiting benchmark')

                    latency_stats = []
                    # Only interested in the output of the first instance
                    out = outputs[0][0]
                    err = outputs[0][1]
                
                    frame_latencies = []
                    for line in err.split('\n'):
                        if 'Mark Duration' not in line:
                            continue
                        duration = float(line.split(':')[-1].strip().replace('ms',''))
                        frame_latencies.append(duration)
                
                    lat = np.array(frame_latencies)
                    latency_stats = [
                        lat.mean(),
                        lat.min(),
                        lat.max(),
                        np.percentile(lat, 95),
                        np.percentile(lat, 99),
                        np.median(lat),
                        np.std(lat)
                    ]
                else:
                    latency_stats = [0, 0, 0, 0, 0, 0]


                if cores is None:
                    cores = cpu_count
                stats = [video_name, cores, procs, args.device, args.sync, codec, bitrate, resolution, avg_fps ] + latency_stats
                benchmark_stats.append(stats)

                pbar.update(1)
                
    df = pd.DataFrame(benchmark_stats, columns=benchmark_metrics)
    
    df.to_csv('{}summary.csv'.format(output_file), sep=',', index=False, float_format='%.3f')
    df = pd.concat([df, cpu_monitor.checkpoints, rapl_monitor.checkpoints, pcm_monitor.checkpoints], axis=1, sort=False)
    df.to_csv('{}detailed.csv'.format(output_file), sep=',', index=False, float_format='%.3f')
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--bin", 
        help="Path to the benchmark app binary",
        type=str,
        default='./bin/benchmark_app'
    )
    parser.add_argument(
        "--benchmark", 
        help="Benchmark: [decoding, inference]",
        type=str,
        default='decoding'
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
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    decoding(args)
    
        
if __name__ == '__main__':
    main()

