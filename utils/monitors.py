import subprocess
import os
import signal
import argparse
import logging
import time
import collections
import xmltodict
from threading import Thread

import pandas as pd
import numpy as np

# Utilization data
import psutil

# Power measurements from local RAPL interface 
import rapl


class NVMonitor:
    def __init__(self):
        self.thread = None
        self.sampling = False
        self.samples = []
        self.result = None
        self.avg = None
        self.checkpoints = None
        self.product_name = None

    def start(self, interval=1.0, keep=False):
        self.interval = interval
        self.sampling = True

        if not keep:
            self.result = None
            self.avg = None

        self.thread = Thread(
            target=self.nv_monitor, 
            args=(self.interval,),
            daemon=True)
        
        self.thread.start()

    def nv_monitor(self, interval):
        data = []
        while self.sampling:
            time.sleep(interval)
            
            p = subprocess.Popen(['nvidia-smi', '-q', '-x'], stdout=subprocess.PIPE)
            out, error = p.communicate()

            xml = dict(xmltodict.parse(out.decode('utf-8')))
            xml = dict(xml['nvidia_smi_log'])

            gpu = dict(xml['gpu'][0])

            categories = ['utilization', 'power_readings', 'encoder_stats', 'temperature']

            # add single metrics 
            headers = ['timestamp', 'duration', 'fan_speed']
            stats = [time.time(), interval, int(gpu['fan_speed'].split(' ')[0])]

            self.product_name = gpu['product_name']
            
            for cat in categories:
                for subcat, stat in gpu[cat].items():
                    headers.append(subcat)
                    s = gpu[cat][subcat].split(' ')[0]
                    try:
                        s = float(s)
                    except ValueError:
                        s = np.nan
                        pass
                    stats.append(s)

            print(stats)
            data.append(stats)
            
        self.result = pd.DataFrame(data, columns=headers)

    def stop(self, checkpoint=True):
        self.sampling = False
        self.thread.join()

        if checkpoint:
            self.checkpoint()

    def average(self):
        if self.result is None:
            logging.warning("Average cannot be computed while monitor is running. Please, call stop() first.")
            return None

        self.avg = pd.DataFrame(self.result.drop('timestamp', axis=1).mean().dropna()).T
        self.avg.insert(0, 'product_name', self.product_name)
        return self.avg

    def to_csv(filename):
        self.result.to_csv(filename, sep=',', index=False)
    
    def checkpoint(self):
        self.avg = None
        _ = self.average()

        if self.checkpoints is None:
            self.checkpoints = self.avg
        else:
            self.checkpoints = self.checkpoints.append(self.avg, ignore_index=True)


class IPMIMonitor:
    def __init__(self):
        self.thread = None
        self.sampling = False
        self.samples = []
        self.result = None
        self.avg = None
        self.checkpoints = None

    def start(self, interval, full=False, keep=False):
        self.interval = interval
        self.sampling = True

        if not keep:
            self.result = None
            self.avg = None

        self.thread = Thread(
            target=self.ipmi_monitor, 
            args=(self.interval, full),
            daemon=True)
        
        self.thread.start()

    def get_full_ipmi_data(self, output):
        data = []
        timestamp = time.time()
        for row in output.split('\n'):
            if 'discrete' in row:
                continue
            sensor, value, unit = [field.strip() for field in row.split('|')[:3]]
            data.append([timestamp, sensor, value, unit])
    
        return data

    def get_power_ipmi_data(self, reading):
        data = []
        timestamp = time.time()
        for row in output.split('\n'):
            if not 'Instantaneous' in row:
                continue
            
            value, unit = row.split(':')[1].strip().split(' ')
            data.append([timestamp, value, unit])

        return data

    def get_ipmi_data(self, full=False):
        data = []
        
        if full:
            opts = ['sensor']
        else:
            opts = ['dcmi', 'power', 'reading']

        cmd = ['ipmitool'] + opts

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        
        out, err = proc.communicate()

        if full:
            return self.get_full_ipmi_data(out.decode('utf-8'))
            
        return self.get_power_ipmi_data(out.decode('utf-8'))

    def ipmi_monitor(self, interval, full=False):
        data = []
        while self.sampling:
            time.sleep(interval)
            
            sample = get_ipmi_data()
            data.append(sample)

        if full:
            columns = ['timestamp', 'sensor', 'value', 'unit']
        else:
            columns = ['timestamp', 'power', 'unit']
        self.result = pd.DataFrame(data, columns=columns)

    def stop(self):
        self.sampling = False
        self.thread.join()
    
    def to_csv(filename):
        self.result.to_csv(filename, sep=',', index=False)

    def checkpoint(self):
        self.avg = None
        _ = self.average()

        if self.checkpoints is None:
            self.checkpoints = self.avg
        else:
            self.checkpoints = self.checkpoints.append(self.avg, ignore_index=True)


class PCMMonitor:
    def __init__(self):
        self.sampling = False
        self.process = None
        self.result = None
        self.avg = None
        self.checkpoints = None

    def start(self, interval=1.0):
        self.interval = interval
        self.pcm_monitor(interval)
        self.result = None
        self.avg = None

    def pcm_monitor(self, interval):
        p = subprocess.Popen(
            ['/tf/workspace/py-sysmon/utils/pcm.x', '-csv'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.process = p
        self.sampling = True

    def stop(self, checkpoint=True):
        os.kill(self.process.pid, signal.SIGTERM)
        self.sampling = False

        out, err = self.process.communicate()
        stats = out.decode('utf-8').split('\n')

        headers = []
        prev_level = ""
        for level,metric in zip(stats[0].split(','), stats[1].split(',')):
            if level != "":
                prev_level = level

                prev_level = prev_level.replace(' ', '')
                prev_level = prev_level.replace('(', '_')
                prev_level = prev_level.replace(')', '')
            header = '{}-{}'.format(prev_level, metric)
            headers.append(header)
            
        data = []
        for sample in stats[2:]:
            data.append(sample.split(','))

        df = pd.DataFrame(data, columns=headers)
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    print('{} impossible to determine'.format(col))

        self.result = df

        if checkpoint:
            self.checkpoint()

    def average(self, percpu=True):
        if self.result is None:
            logging.warning("Average cannot be computed while monitor is running. Please, call stop() first.")
            return None

        self.avg = pd.DataFrame(self.result.mean(numeric_only=True).dropna()).T

        return self.avg

    def to_csv(filename):
        self.result.to_csv(filename, sep=',', index=False)

    def checkpoint(self):
        if self.sampling:
            logging.warning("PCMMonitor can't be checkpointed while running. Please, call stop() first.")

        self.avg = None
        _ = self.average()

        if self.checkpoints is None:
            self.checkpoints = self.avg
        else:
            self.checkpoints = self.checkpoints.append(self.avg, ignore_index=True)


class RAPLMonitor:
    def __init__(self):
        # self.interval = interval
        self.thread = None
        self.sampling = False
        self.samples = []
        self.result = None
        self.avg = None
        self.checkpoints = None

    def start(self, interval=1.0, keep=False):
        self.interval = interval
        self.sampling = True

        if not keep:
            self.result = None
            self.avg = None

        self.thread = Thread(
            target=self.rapl_monitor, 
            args=(self.interval,),
            daemon=True)
        
        self.thread.start()

    def get_sample_data(self, diff):
        data = [time.time()]
        data.append(diff.duration)
        for domain in diff.domains:
            data.append(diff.domains[domain].values['energy_uj'])
            for sd in diff.domains[domain].subdomains:
                data.append(diff.domains[domain].subdomains[sd].values['energy_uj'])
        
        return data

    def rapl_monitor(self, interval):
        prev_sample = rapl.RAPLMonitor.sample()
        data = []
        while self.sampling:
            time.sleep(interval)
            
            cur_sample = rapl.RAPLMonitor.sample()
            diff = cur_sample - prev_sample
            prev_sample = cur_sample
            
            sample_data = self.get_sample_data(diff)
            data.append(sample_data)

        columns = ['timestamp', 'duration']
        for domain in diff.domains:
            columns.append(domain)
            for sd in diff.domains[domain].subdomains:
                columns.append(sd)
        self.result = pd.DataFrame(data, columns=columns)

    def stop(self, checkpoint=True):
        self.sampling = False
        self.thread.join()

        if checkpoint:
            self.checkpoint()

    def average(self):
        if self.result is None:
            logging.warning("Average cannot be computed while monitor is running. Please, call stop() first.")
            return None

        self.avg = pd.DataFrame(self.result.drop('timestamp', axis=1).mean().dropna()).T
        return self.avg

    def to_csv(filename):
        self.result.to_csv(filename, sep=',', index=False)
    
    def checkpoint(self):
        self.avg = None
        _ = self.average()

        if self.checkpoints is None:
            self.checkpoints = self.avg
        else:
            self.checkpoints = self.checkpoints.append(self.avg, ignore_index=True)


class ResourceMonitor:
    def __init__(self):
        # self.interval = interval
        self.thread = None
        self.sampling = False
        self.result = None
        self.avg = None
        self.checkpoints = None
        self.cpu_count = psutil.cpu_count(logical=True)

    def start(self, interval, sensors=False):
        self.interval = interval
        self.get_sensors = sensors
        self.sampling = True
        self.result = None
        self.avg = None

        self.thread = Thread(
            target=self.monitor_resources, 
            args=(self.interval,),
            daemon=True)
        
        self.thread.start()

    def get_cpu_stats(self):
        data = []
        timestamp = time.time()
        
        cpu_utils = psutil.cpu_percent(percpu=True)
        cpu_times = psutil.cpu_times_percent(percpu=True)
        cpu_freqs = psutil.cpu_freq(percpu=True)

        for cpu_id in range(self.cpu_count):
            data_cpu = [
                timestamp, 
                cpu_id, 
                cpu_utils[cpu_id]
            ]

            data_cpu += list(cpu_times[cpu_id])
            data_cpu += list(cpu_freqs[cpu_id])
        
            data.append(data_cpu)

        return data

    def get_cpu_headers(self):
        headers = ['timestamp', 'cpu_id', 'util']

        cpu_times = psutil.cpu_times_percent(percpu=False)
        times = ['time_{}'.format(time) for time in cpu_times._fields]

        cpu_freqs = psutil.cpu_freq(percpu=False)
        freqs = ['freq_{}'.format(freq) for freq in cpu_freqs._fields]

        headers += times
        headers += freqs

        return headers

    def get_sensors_stats(self):
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
        else:
            temps = {}
        if hasattr(psutil, "sensors_fans"):
            fans = psutil.sensors_fans()
        else:
            fans = {}
        if hasattr(psutil, "sensors_battery"):
            battery = psutil.sensors_battery()
        else:
            battery = None

        if not any((temps, fans, battery)):
            logger.warning("can't read any temperature, fans or battery info")
            return

        # Temperatures.
        temp_sensors = []
        for name in temps.keys():
            temp_name = 'temp_{}'.format(name)
            # temp_sensors.append(name)
            for entry in temps[name]:
                for field in entry._fields:
                    temp_sensors.append(getattr(entry, field))
        
        # Fans.
        fan_sensors = []
        for name in fans.keys():
            fan_name = 'fan_{}'.format(name)
            # fan_sensors.append(name)
            for entry in fans[name]:
                for field in entry._fields:
                    fan_sensors.append(getattr(entry, field))
        
        # Battery.
        battery_sensors = []
        if battery:
            battery_sensors = [battery.power_plugged]
        
        all_sensors = [time.time()] + temp_sensors + fan_sensors + battery_sensors
        return all_sensors

    def get_sensors_headers(self):
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
        else:
            temps = {}
        if hasattr(psutil, "sensors_fans"):
            fans = psutil.sensors_fans()
        else:
            fans = {}
        if hasattr(psutil, "sensors_battery"):
            battery = psutil.sensors_battery()
        else:
            battery = None

        if not any((temps, fans, battery)):
            logger.warning("can't read any temperature, fans or battery info")
            return

        # Temperatures.
        temp_sensors = []
        for name in temps.keys():
            temp_name = 'temp_{}'.format(name)
            # temp_sensors.append(name)
            for i, entry in enumerate(temps[name]):
                for field in entry._fields:
                    temp_sensors.append(f'{temp_name}_{i}.{field}')
        
        # Fans.
        fan_sensors = []
        for name in fans.keys():
            fan_name = 'fan_{}'.format(name)
            # fan_sensors.append(name)
            for i, entry in enumerate(fans[name]):
                for field in entry._fields:
                    fan_sensors.append(f'{fan_name}_{i}.{field}')
        
        # Battery.
        battery_sensors = []
        if battery:
            battery_sensors = ['battery.power_plugged']
        
        all_sensors = ['timestamp'] + temp_sensors + fan_sensors + battery_sensors
        return all_sensors

    def monitor_resources(self, interval):
        data = []
        data_sensors = []
        _ = self.get_cpu_stats() # First reading must be discarded
        
        if self.get_sensors:
            _ = self.get_sensors_stats()
        
        while self.sampling:
            time.sleep(interval)
            
            sample_data = self.get_cpu_stats()
            data += sample_data

            if self.get_sensors:
                sample_data = self.get_sensors_stats()
                data_sensors += sample_data

        columns = self.get_cpu_headers()    
        self.result = pd.DataFrame(data, columns=columns)

        if self.get_sensors:
            columns = self.get_sensors_headers()
            self.result_sensors = pd.DataFrame(data_sensors, columns=columns)

    def stop(self, checkpoint=True):
        self.sampling = False   
        self.thread.join()

        if checkpoint:
            self.checkpoint()

    def average(self, percpu=True):
        if self.result is None:
            logging.warning("Average cannot be computed while monitor is running. Please, call stop() first.")
            return None

        avg = self.result

        pkg_avg = pd.DataFrame(avg.groupby('timestamp').mean())
        pkg_avg = pkg_avg.reset_index().drop(['timestamp', 'cpu_id'], axis=1)

        headers = list('pkg_' + pkg_avg.columns.values)
        pkg_avg = pkg_avg.mean().values.tolist()
        self.avg = pd.DataFrame([pkg_avg], columns=headers)
        
        if percpu:
            cpus = self.result.cpu_id.unique()
            avg_data = []
            
            for cpu in cpus:
                avg_cpu = self.result[self.result.cpu_id == cpu]
                avg_cpu = avg_cpu.drop(['timestamp', 'cpu_id'], axis=1)

                headers += list('cpu{}_'.format(cpu) + avg_cpu.columns.values)
                avg_data += avg_cpu.mean().values.tolist()

            data = pkg_avg + avg_data
            self.avg = pd.DataFrame([data], columns=headers)

        return self.avg
        
    def to_csv(filename):
        self.result.to_csv(filename, sep=',', index=False)

    def checkpoint(self):
        self.avg = None
        _ = self.average()

        if self.checkpoints is None:
            self.checkpoints = self.avg
        else:
            self.checkpoints = self.checkpoints.append(self.avg, ignore_index=True)

def setup():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def check():
    thread_count = psutil.cpu_count(logical=True)
    cpu_count = psutil.cpu_count(logical=False)
    if cpu_count < thread_count:
        logging.warning("Hyper-Threading is ENABLED.")



def get_mem_stats():
    virtual_memory = psutil.virtual_memory()
    print(virtual_memory)

def get_sensors_stats():
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
        else:
            temps = {}
        if hasattr(psutil, "sensors_fans"):
            fans = psutil.sensors_fans()
        else:
            fans = {}
        if hasattr(psutil, "sensors_battery"):
            battery = psutil.sensors_battery()
        else:
            battery = None

        if not any((temps, fans, battery)):
            print("can't read any temperature, fans or battery info")
            return

        names = set(list(temps.keys()) + list(fans.keys()))
        for name in names:
            print(name)
            # Temperatures.
            if name in temps:
                print("    Temperatures:")
                for entry in temps[name]:
                    print("        %-20s %s°C (high=%s°C, critical=%s°C)" % (
                        entry.label or name, entry.current, entry.high,
                        entry.critical))
            # Fans.
            if name in fans:
                print("    Fans:")
                for entry in fans[name]:
                    print("        %-20s %s RPM" % (
                        entry.label or name, entry.current))

        # Battery.
        if battery:
            print("Battery:")
            print("    charge:     %s%%" % round(battery.percent, 2))
            if battery.power_plugged:
                print("    status:     %s" % (
                    "charging" if battery.percent < 100 else "fully charged"))
                print("    plugged in: yes")
            else:
                print("    left:       %s" % secs2hours(battery.secsleft))
                print("    status:     %s" % "discharging")
                print("    plugged in: no")


def do_something():
    time.sleep(1)

def main():
    setup()
    check()

    
    # do_something()
    # get_cpu_stats()
    get_mem_stats()
    get_sensors_stats()

    # meter.end()
    # print(report.data.head())
    # print(meter.result)



if __name__ == '__main__':
    main()
