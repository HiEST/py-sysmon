import argparse
import glob
import json
import logging
import os
import shlex
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm

from utils.monitors import PCMMonitor, TurboStatMonitor, ResourceMonitor, NVMonitor

MIN_RUNTIME = 5


class PipelineError(Exception):
    """Raised when the gst pipeline won't preroll"""
    pass


def run(cmdline, monitors, timeout=None, retries=5):
    if isinstance(cmdline, str):
        cmdline = shlex.split(cmdline)

    while retries > 0:
        try:
            for monitor in monitors:
                monitor.start(interval=1.0)

            subproc = subprocess.Popen(
                cmdline,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            out, err = subproc.communicate(timeout=timeout)
            out = out.decode('utf-8')
            err = err.decode('utf-8')

            if 'ERROR' not in err:
                for monitor in monitors:
                    monitor.stop(checkpoint=True)
                return out, err

        except Exception:
            raise

        for monitor in monitors:
            monitor.stop(checkpoint=False)
        retries = retries - 1

    # if we reach this point, retries = 0
    raise PipelineError("Command failed all retries: {}"
                        .format(' '.join(cmdline)))


def get_video_info(video):
    ffprobe = ('ffprobe -loglevel quiet -print_format '
               'json -show_format -show_streams {}')
    ffproc = subprocess.Popen(
        shlex.split(ffprobe.format(str(video))),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out, err = ffproc.communicate()
    video_info = json.loads(out)

    codec = video_info['streams'][0]['codec_name']
    resolution = video_info['streams'][0]['height']
    duration = video_info['streams'][0]['duration']
    fr, div = video_info['streams'][0]['avg_frame_rate'].split('/')
    frame_rate = float(fr) / float(div)
    bitrate = int(video_info['streams'][0]['bit_rate'])
    return [codec, resolution, duration, frame_rate, bitrate]


def benchmark_ffmpeg(video, configs, device, sync, timeout, monitors):
    codec, resolution, duration, frame_rate, bitrate = get_video_info(video)

    if codec == 'h265':
        codec = 'hevc'

    if device == 'cpu':
        decoder = f'{codec}'
    elif device == 'nvidia':
        decoder = f'{codec}_cuvid'

    ffmpeg_cli = (f'ffmpeg -hide_banner -nostdin -flags unaligned '
                  f'-c:v {decoder} -i {str(video)} -f null -')

    video_name = video.stem
    configs_per_video = len(configs)

    proc = psutil.Process()
    for _, cores, procs in tqdm(configs.itertuples(),
                                desc='Runs with {}'.format(video_name),
                                total=configs_per_video, leave=False):
        if not isinstance(cores, int):
            cores = psutil.cpu_count(logical=False)
        if not isinstance(procs, int):
            procs = 1

        proc.cpu_affinity(list(np.arange(0, cores)))

        # 1. First run to get throughput and telemetry
        runtime = None
        try:
            t0 = time.time()
            out, err = run(ffmpeg_cli, monitors, timeout=timeout)
            t1 = time.time()
            runtime = t1-t0
        except PipelineError:
            logging.warning(('Skipping experiment with video {} '
                             'and {} procs with {} cpus').format(video_name,
                                                                 procs, cores))
            continue
        except Exception as e:
            raise type(e)(f'{str(e)}\nFailed with pipeline: {ffmpeg_cli}')

        for line in err.split('\n'):
            # if 'Execution ended after' in line:
            #     runtime = line.split(' ')[-1]
            #     hours, minutes, seconds = runtime.split(':')
            #     runtime = int(hours)*3600 + int(minutes)*60 + float(seconds)
            if 'fps=' in line:
                all_fps = []
                for subline in line.split('\r'):
                    metrics = subline.strip().split(' ')
                    for m in metrics:
                        if 'fps' in m:
                            fps = m.split('=')[1]
                            try:
                                fps = float(fps)
                            except Exception:
                                raise Exception("fps metric wrong "
                                                "format (not fp=x)")
                            all_fps.append(fps)

                fps = sum(all_fps)/len(all_fps)

        decoding_fps = fps
        # relative_speed = float(duration) / runtime
        # decoding_fps = frame_rate * relative_speed
        if runtime < MIN_RUNTIME:
            logging.warning(("Runtime is too low for meaningful "
                             "telemetry (runtime: {})").format(runtime))

        # FIXME: If possible, get decoding latency with ffmpeg
        # 2. No second run for latency with ffmpeg
        latency_stats = [0, 0, 0, 0, 0, 0, 0]

        stats = [video_name, cores, procs, device,
                 sync, codec, bitrate, resolution,
                 decoding_fps] + latency_stats
        return stats


def benchmark_gst(video, configs, device, sync,
                  timeout, plugin_path, no_latency, monitors):
    codec, resolution, duration, frame_rate, bitrate = get_video_info(video)
    gst_pipeline = ('gst-launch-1.0 {} filesrc location={} ! '
                    'qtdemux ! {} {}parse ! {} {} ! fakesink sync={}')

    if codec == 'hevc':
        codec = 'h265'

    if device == 'cpu':
        decoder = f'avdec_{codec}'
    else:
        decoder = f'vaapi{codec}dec'

    gst_throughput = gst_pipeline.format('', str(video), '',
                                         codec, decoder, '', str(sync))
    gst_latency = gst_pipeline.format(f'--gst-plugin-path={plugin_path}',
                                      str(video), 'markin name=moo !', codec,
                                      decoder, '! markout', str(sync))

    video_name = video.stem
    configs_per_video = len(configs)

    proc = psutil.Process()
    for _, cores, procs in tqdm(configs.itertuples(),
                                desc='Runs with {}'.format(video_name),
                                total=configs_per_video, leave=False):
        if not isinstance(cores, int):
            cores = psutil.cpu_count(logical=False)
        if not isinstance(procs, int):
            procs = 1

        proc.cpu_affinity(list(np.arange(0, cores)))

        # 1. First run to get throughput and telemetry
        try:
            out, err = run(gst_throughput, monitors, timeout=timeout)
        except PipelineError:
            logging.warning(('Skipping experiment with video {} '
                             'and {} procs with {} cpus').format(video_name,
                                                                 procs, cores))
            continue
        except Exception as e:
            raise type(e)(e.message +
                          f'. Failed with pipeline: {gst_throughput}')

        runtime = None
        for line in out.split('\n')[-10:]:
            if 'Execution ended after' in line:
                runtime = line.split(' ')[-1]
                hours, minutes, seconds = runtime.split(':')
                runtime = int(hours)*3600 + int(minutes)*60 + float(seconds)

        relative_speed = float(duration) / runtime
        decoding_fps = frame_rate * relative_speed
        if runtime < MIN_RUNTIME:
            logging.warning(("Runtime is too low for meaningful "
                             "telemetry (runtime: {})").format(runtime))

        # 2. Second run to only get latency, unless otherwise specified
        if not no_latency:
            os.environ['GST_DEBUG'] = 'markout:5'
            try:
                out, err = run(gst_latency, monitors, timeout=timeout)
            except Exception as e:
                raise type(e)(e.message +
                              f'. Failed with pipeline: {gst_latency}')

            del os.environ['GST_DEBUG']

            frame_latencies = []
            for line in err.split('\n'):
                if 'Mark Duration' not in line:
                    continue
                duration = float(line.split(':')[-1].strip().replace('ms', ''))
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
            latency_stats = [0, 0, 0, 0, 0, 0, 0]

        stats = [video_name, cores, procs, device,
                 sync, codec, bitrate, resolution,
                 decoding_fps] + latency_stats
        return stats


def parse_arguments(parser):
    parser.add_argument(
        "-b", "--benchmark",
        choices=['ffmpeg', 'gst'],
        help="Application to use as benchmark",
        type=str,
        default='ffmpeg'
    )
    parser.add_argument(
        "-d", "--device",
        help="Device to run the benchmark.",
        type=str,
        default='CPU'
    )
    parser.add_argument(
        "-c", "--config",
        help=("Config file from which configurations of "
              "{cores, requests, streams, and batch size} are read."),
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
        required=True,
        type=str,
    )
    parser.add_argument(
        "--plugin",
        help="GST_PLUGIN_PATH (directory where latency pluting is).",
        type=str,
    )
    parser.add_argument(
        "--sync",
        help=("Synchronous decoding "
              "(decoding rate locked at input's framerate)."),
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "--no-latency",
        help="Skip the second run to get latency.",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "--monitor-nvidia",
        help="Enable nvidia monitor.",
        default=False,
        action='store_true'
    )

    args = parser.parse_args()
    return args


def main():
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=logging.DEBUG)

    if args.prefix:
        output_file = args.output + '/' + args.prefix + '-'
    else:
        output_file = args.output + '/'

    if '.mp4' in args.input:
        if not os.path.isfile(args.input):
            raise ValueError('{} does not exist.'.format(args.input))
        inputs = [Path(args.input)]
    elif os.path.isdir(args.input):
        inputs = [video for video in Path(args.input).glob('*.mp4')]
        inputs.extend([video for video in Path(args.input).glob('*.webm')])
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

    # System monitors
    cpu_monitor = ResourceMonitor()
    turbo_monitor = TurboStatMonitor()
    pcm_monitor = PCMMonitor()
    monitors = [cpu_monitor, turbo_monitor, pcm_monitor]
    if args.monitor_nvidia:
        nv_monitor = NVMonitor()
        monitors.append(nv_monitor)

    benchmark_stats = []
    benchmark_metrics = ['Video', 'CPUs', 'Procs', 'Device', 'Sync',
                         'Codec', 'Bitrate', 'Resolution', 'Throughput']
    benchmark_metrics += ['Latency Avg', 'Latency Max', 'Latency Min',
                          'Latency 95%', 'Latency 99%',
                          'Latency Median', 'Latency StdDev']

    configs_per_video = len(configs)
    total_runs = len(inputs) * configs_per_video

    device = args.device.lower()
    with tqdm(total=total_runs, desc="Total runs") as pbar:
        for video in tqdm(inputs, desc='Videos to decode', total=len(inputs)):

            try:
                if args.benchmark == 'gst':
                    stats = benchmark_gst(video, configs, device,
                                          args.sync, args.time, args.plugin,
                                          args.no_latency, monitors)
                elif args.benchmark == 'ffmpeg':
                    stats = benchmark_ffmpeg(video, configs, device,
                                             args.sync, args.time, monitors)
            except Exception:
                # Save work
                logging.error("Saving current work...")
                df = pd.DataFrame(benchmark_stats,
                                  columns=benchmark_metrics)
                df.to_csv('{}summary.csv.bak'.format(output_file),
                          sep=',', index=False, float_format='%.3f')

                df = pd.concat([df,
                                cpu_monitor.checkpoints,
                                turbo_monitor.checkpoints,
                                pcm_monitor.checkpoints],
                               axis=1, sort=False)
                if args.monitor_nvidia:
                    df = pd.concat([df,
                                    nv_monitor.checkpoints])

                df.to_csv('{}detailed.csv.bak'.format(output_file),
                          sep=',', index=False, float_format='%.3f')
                raise

            benchmark_stats.append(stats)
            pbar.update(1)

    df = pd.DataFrame(benchmark_stats, columns=benchmark_metrics)

    df.to_csv('{}summary.csv'.format(output_file),
              sep=',', index=False, float_format='%.3f')
    df = pd.concat([df,
                    cpu_monitor.checkpoints,
                    turbo_monitor.checkpoints,
                    pcm_monitor.checkpoints],
                   axis=1, sort=False)
    if args.monitor_nvidia:
        df = pd.concat([df, nv_monitor.checkpoints])

    df.to_csv('{}detailed.csv'.format(output_file),
              sep=',', index=False, float_format='%.3f')


if __name__ == '__main__':
    main()
