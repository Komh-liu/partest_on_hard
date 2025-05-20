# hardware_monitor.py
import psutil
import pynvml
import numpy as np
from threading import Event, Thread
from typing import Dict, List, Any
import time

class HardwareMonitor:
    def __init__(self):
        # CPU信息
        self.cpu_cores = psutil.cpu_count(logical=False)
        self.cpu_threads = psutil.cpu_count(logical=True)
        
        # 内存信息
        self.mem_total = psutil.virtual_memory().total
        
        # GPU信息
        self.gpu_count = 0
        self.gpu_handles = []
        try:
            pynvml.nvmlInit()
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.gpu_count)]
        except pynvml.NVMLError:
            pass
        
        # 监控控制
        self._stop_event = Event()
        self.metrics_log = []

    def _get_cuda_compute_capability(self) -> str:
        """获取CUDA计算能力"""
        if self.gpu_count == 0:
            return ""
        try:
            cc_major = pynvml.nvmlDeviceGetCudaComputeCapability(self.gpu_handles[0])[0]
            cc_minor = pynvml.nvmlDeviceGetCudaComputeCapability(self.gpu_handles[0])[1]
            return f"{cc_major}{cc_minor}"
        except:
            return ""

    def start_monitoring(self, interval: float = 0.1):
        """启动后台监控线程"""
        self._stop_event.clear()
        self.metrics_log.clear()
        self.monitor_thread = Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.start()

    def stop_monitoring(self):
        """停止监控"""
        self._stop_event.set()
        if self.monitor_thread.is_alive():
            self.monitor_thread.join()

    def _monitor_loop(self, interval: float):
        """监控循环（含详细CPU负载指标）"""
        while not self._stop_event.is_set():
            metrics = {
                'timestamp': time.time(),
                'cpu_usage': psutil.cpu_percent(interval=0.1),
                'cpu_per_core': psutil.cpu_percent(interval=0.1, percpu=True),
                'cpu_ctx_switches': psutil.cpu_stats().ctx_switches,
                'cpu_interrupts': psutil.cpu_stats().interrupts,
                'mem_usage': psutil.virtual_memory().used,
                'gpu_usage': [],
                'gpu_mem': []
            }

            # 监控GPU
            if self.gpu_count > 0:
                for handle in self.gpu_handles:
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        metrics['gpu_usage'].append(util.gpu)
                        metrics['gpu_mem'].append(mem_info.used)
                    except pynvml.NVMLError:
                        pass

            self.metrics_log.append(metrics)
            time.sleep(interval)

    def generate_report(self, task_name: str) -> Dict[str, Any]:
        """生成包含负载均衡分析的完整报告"""
        if not self.metrics_log:
            return {}

        # CPU分析
        cpu_usage = np.array([m['cpu_usage'] for m in self.metrics_log])
        avg_cpu = np.mean(cpu_usage)
        max_cpu = np.max(cpu_usage)

        # 核心级负载均衡分析
        core_metrics = []
        if 'cpu_per_core' in self.metrics_log[0]:
            core_usage_data = [m['cpu_per_core'] for m in self.metrics_log]
            for core_idx in range(len(core_usage_data[0])):
                single_core_usage = [sample[core_idx] for sample in core_usage_data]
                core_metrics.append({
                    "core_id": core_idx,
                    "avg_usage": f"{np.mean(single_core_usage):.1f}%",
                    "std_usage": f"{np.std(single_core_usage):.1f}%",
                    "max_usage": f"{np.max(single_core_usage):.1f}%",
                    "min_usage": f"{np.min(single_core_usage):.1f}%"
                })

        # CPU竞争指标
        total_ctx_switches = self.metrics_log[-1]['cpu_ctx_switches'] - self.metrics_log[0]['cpu_ctx_switches']
        total_interrupts = self.metrics_log[-1]['cpu_interrupts'] - self.metrics_log[0]['cpu_interrupts']

        # 内存分析
        max_mem = max(m['mem_usage'] for m in self.metrics_log)

        # GPU分析
        avg_gpu, max_gpu = 0.0, 0.0
        if self.gpu_count > 0:
            gpu_usage = [u for m in self.metrics_log for u in m['gpu_usage']]
            avg_gpu = np.mean(gpu_usage) if gpu_usage else 0.0
            max_gpu = np.max(gpu_usage) if gpu_usage else 0.0

        return {
            'task_name': task_name,
            'hardware': {
                'cpu_cores': self.cpu_cores,
                'cpu_threads': self.cpu_threads,
                'gpu_count': self.gpu_count,
                'cuda_cc': self._get_cuda_compute_capability(),
                'total_mem': f"{self.mem_total / 1024**3:.1f} GB"
            },
            'metrics': {
                'avg_cpu': f"{avg_cpu:.1f}%",
                'max_cpu': f"{max_cpu:.1f}%",
                'cpu_load_balance': core_metrics,
                'cpu_contention': {
                    'total_ctx_switches': total_ctx_switches,
                    'total_interrupts': total_interrupts
                },
                'max_mem': f"{max_mem / 1024**3:.2f} GB",
                'avg_gpu': f"{avg_gpu:.1f}%" if self.gpu_count else "N/A",
                'max_gpu': f"{max_gpu:.1f}%" if self.gpu_count else "N/A"
            }
        }
