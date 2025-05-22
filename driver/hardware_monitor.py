import psutil
import pynvml
import numpy as np
from threading import Event, Thread
from typing import Dict, List, Any, Optional
import time

class HardwareMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.cpu_cores = psutil.cpu_count(logical=False)
        self.cpu_threads = psutil.cpu_count(logical=True)
        self.mem_total = psutil.virtual_memory().total
        
        # GPU 相关初始化
        self.gpu_count = 0
        self.gpu_handles = []
        try:
            pynvml.nvmlInit()
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.gpu_count)]
        except pynvml.NVMLError:
            pass
        
        self._stop_event = Event()
        self.metrics_log = []
        self._monitoring_thread = None
        
    def start_monitoring(self):
        """开始监控硬件资源使用情况"""
        self._stop_event.clear()
        self._monitoring_thread = Thread(target=self._monitor_loop)
        self._monitoring_thread.start()
        
    def stop_monitoring(self):
        """停止监控"""
        if self._monitoring_thread is not None:
            self._stop_event.set()
            self._monitoring_thread.join()
            
    def generate_report(self, task_type: str, phase_times: Optional[Dict] = None) -> Dict:
        """生成监控报告"""
        report = {
            "hardware": {
                "cpu_cores": self.cpu_cores,
                "cpu_threads": self.cpu_threads,
                "memory_total": self.mem_total,
                "gpu_count": self.gpu_count
            },
            "metrics": self._calculate_metrics(phase_times),
            "task_type": task_type
        }
        
        if phase_times:
            report["time_window"] = {
                "start": phase_times["bfs_start"],
                "end": phase_times["bfs_end"],
                "duration_ms": (phase_times["bfs_end"] - phase_times["bfs_start"]) * 1000
            }
            
        return report

    def _monitor_loop(self):
        """监控循环"""
        while not self._stop_event.is_set():
            metrics = {
                "timestamp": time.time(),
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": self.process.memory_percent(),
                "gpu_usage": []
            }
            
            # 获取GPU使用情况
            if self.gpu_count > 0:
                for handle in self.gpu_handles:
                    try:
                        info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        metrics["gpu_usage"].append(info.gpu)
                    except pynvml.NVMLError:
                        metrics["gpu_usage"].append(0)
                        
            self.metrics_log.append(metrics)
            time.sleep(0.1)  # 采样间隔
            
    def _calculate_metrics(self, phase_times: Optional[Dict] = None) -> Dict[str, float]:
        """计算监控指标"""
        if not self.metrics_log:
            return {}
            
        metrics = {}
        
        # 如果指定了时间窗口，只计算该窗口内的指标
        if phase_times:
            window_metrics = [
                m for m in self.metrics_log 
                if phase_times["bfs_start"] <= m["timestamp"] <= phase_times["bfs_end"]
            ]
        else:
            window_metrics = self.metrics_log
            
        if window_metrics:
            # CPU使用率
            metrics["avg_cpu_usage"] = np.mean([m["cpu_usage"] for m in window_metrics])
            metrics["max_cpu_usage"] = np.max([m["cpu_usage"] for m in window_metrics])
            
            # 内存使用率
            metrics["avg_memory_usage"] = np.mean([m["memory_usage"] for m in window_metrics])
            metrics["max_memory_usage"] = np.max([m["memory_usage"] for m in window_metrics])
            
            # GPU使用率(如果有)
            if self.gpu_count > 0:
                gpu_usage = [m["gpu_usage"] for m in window_metrics if m["gpu_usage"]]
                if gpu_usage:
                    metrics["avg_gpu_usage"] = np.mean([u[0] for u in gpu_usage])
                    metrics["max_gpu_usage"] = np.max([u[0] for u in gpu_usage])
                    
        return metrics
