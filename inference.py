import os
import time

import subprocess
import argparse
import glob
import psutil

from pathlib import Path

import numpy as np
import pandas as pd
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
        "-m", "--model",
        help="Path to the model or model's directory to use.",
        type=str,
        default='./models/'
    )
    parser.add_argument(
        "-c", "--config", 
        help="Config file from which configurations of {cores, requests, streams, and batch size} are read.",
        type=str,
        # required=True
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
        "-t",
        help="Duration of each experiment.",
        type=int,
        default=2
    )
    parser.add_argument(
        "-p", "--precision",
        help="Specifies precision of the model, if one model is set. If a directory, it will select only models of the selected precision.",
        type=str,
        default=None
    )
    parser.add_argument(
        "--source",
        help="File with environment variables to 'source'.",
        type=str,
        default="/opt/intel/openvino/bin/setupvars.sh"
    )
    
    args = parser.parse_args()

    if 'xml' in args.model:
        if not os.path.isfile(args.model):
            raise ValueError('{} does not exist.'.format(args.model))
        models = [Path(args.model)]
    elif os.path.isdir(args.model):
        models = [ path for path in Path(args.model).rglob('*.xml')]
        if args.precision:
            models = [
                model for model in models 
                if args.precision.lower() in str(model).lower()
            ]
    else:
        raise ValueError(
            '{} is not a valid directory nor xml file.'.format(args.model)
        )

    if args.precision is None:
        args.precision = "fp32*"

    # exec(open(args.source).read())
    # subprocess.Popen('source {}'.format(args.source), shell=True)
    # ld_path = Path(__file__).parent.absolute() / 'bin/lib'
    openvino_path = "/opt/intel/openvino_2020.2.120/deployment_tools"
    ld_paths = [
        "{}/bin/lib".format(os.getcwd()),
        '{}/inference_engine/external/tbb/lib/'.format(openvino_path),
        '{}/inference_engine/lib/intel64/'.format(openvino_path),
        '{}/ngraph/lib/'.format(openvino_path),
    os.environ['LD_LIBRARY_PATH'] = ':'.join(ld_paths)

    cpu_count = psutil.cpu_count(logical=False)

    if args.config is not None:
        configs = pd.read_csv(args.config)
    else:
        default_config = [[cpu_count, 1, 1, 1]]
        configs = pd.DataFrame(default_config, columns=['cores', 'streams', 'requests', 'batch'])

    #System monitors
    cpu_monitor = ResourceMonitor()
    rapl_monitor = RAPLMonitor()
    pcm_monitor = PCMMonitor()

    proc = psutil.Process()
    benchmark_stats = []
    benchmark_metrics = ['Count', 'Duration', 'Latency', 'Throughput']

    configs_per_model = len(configs)
    total_runs = len(models) * configs_per_model
    

    with tqdm(total=total_runs, desc="Total runs") as pbar:
        for model in tqdm(models, desc='Models to run', total=len(models)):
            model_name = model.stem
            for _, cores, nstreams, nireq, batch in tqdm(configs.itertuples(), desc='Runs with {}'.format(model_name), total=configs_per_model, leave=False):
                if not isinstance(cores, int):
                    cores = cpu_count
                if not isinstance(nstreams, int):
                    nstreams = cpu_count
                if not isinstance(nireq, int):
                    nireq = nstreams
                if not isinstance(batch, int):
                    batch = 1

                proc.cpu_affinity(list(np.arange(0, cores)))

                pcm_monitor.start(interval=1.0)
                cpu_monitor.start(interval=1.0)
                rapl_monitor.start(interval=1.0)

                subproc = subprocess.Popen(
                    [args.bin, 
                    '-m', str(model),
                    '-d', args.device,
                    '-nstreams', str(nstreams),
                    '-nireq', str(nireq),
                    '-b', str(batch),
                    '-t', str(args.t)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

                out, err = subproc.communicate()

                cpu_monitor.stop(checkpoint=True)
                rapl_monitor.stop(checkpoint=True)
                pcm_monitor.stop(checkpoint=True)
                
                stats = [model_name, cores, nstreams, nireq, batch, args.precision]
                for line in out.decode('utf-8').split('\n')[-10:]:
                    if not any(metric in line for metric in benchmark_metrics):
                        continue

                    metric, value = line.split(':')
                    stats.append(float(value.strip().split(' ')[0]))


                benchmark_stats.append(stats)

                # print(j)
                # progress = i*configs_per_model + j
                pbar.update(1)
                
    benchmark_metrics = ['Model', 'CPUs', 'streams', 'requests', 'batch_size', 'precision'] + benchmark_metrics
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

