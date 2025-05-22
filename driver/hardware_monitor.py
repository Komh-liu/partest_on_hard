import psutil
import pynvml
import numpy as np
from threading import Event, Thread
from typing import Dict, List, Any, Optional
import time
import subprocess
import os
import re

class HardwareMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.pid = self.process.pid
        self.cpu_cores = psutil.cpu_count(logical=False)
        self.cpu_threads = psutil.cpu_count(logical=True)
        self.mem_total = psutil.virtual_memory().total
        
        # 缓存性能监控初始化
        self.perf_available = self._check_perf_availability()
        self.perf_process = None
        self.cache_metrics = {"L1_miss": 0, "LLC_miss": 0, "instructions": 0}
        
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
        
        # 记录初始状态的上下文切换数
        if hasattr(self.process, 'num_ctx_switches'):
            self.initial_ctx_switches = self.process.num_ctx_switches()
        else:
            self.initial_ctx_switches = None
    
    def _check_perf_availability(self):
        """检查perf工具是否可用"""
        try:
            result = subprocess.run(["perf", "--version"], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE,
                                  text=True)
            return result.returncode == 0
        except (FileNotFoundError, PermissionError):
            return False
    
    def start_monitoring(self):
        """开始监控硬件资源使用情况"""
        self._stop_event.clear()
        self._monitoring_thread = Thread(target=self._monitor_loop)
        self._monitoring_thread.start()
        
        # 如果perf可用，启动perf统计
        if self.perf_available:
            try:
                cmd = [
                    "perf", "stat", "-e", 
                    "L1-dcache-load-misses,LLC-load-misses,instructions",
                    "-p", str(self.pid)
                ]
                self.perf_process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
            except Exception as e:
                print(f"Perf monitoring error: {e}")
                self.perf_process = None
        
    def stop_monitoring(self):
        """停止监控"""
        if self._monitoring_thread is not None:
            self._stop_event.set()
            self._monitoring_thread.join()
            
        # 停止perf并获取结果
        if self.perf_process is not None:
            try:
                self.perf_process.terminate()
                _, stderr = self.perf_process.communicate(timeout=1.0)
                self._parse_perf_output(stderr)
            except Exception as e:
                print(f"Error stopping perf: {e}")
            finally:
                self.perf_process = None
            
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
            "task_type": task_type,
            "cache_metrics": self.cache_metrics
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
                "cpu_per_core": psutil.cpu_percent(percpu=True),
                "memory_usage": self.process.memory_percent(),
                "gpu_usage": [],
                "gpu_memory": [],
                "ctx_switches": {}
            }
            
            # 获取CPU负载均衡指标
            metrics["cpu_load_balance"] = {
                "std_dev": np.std(metrics["cpu_per_core"]),
                "max_diff": max(metrics["cpu_per_core"]) - min(metrics["cpu_per_core"]) 
                             if metrics["cpu_per_core"] else 0
            }
            
            # 获取上下文切换信息
            if hasattr(self.process, 'num_ctx_switches'):
                current_ctx = self.process.num_ctx_switches()
                metrics["ctx_switches"]["voluntary"] = current_ctx.voluntary
                metrics["ctx_switches"]["involuntary"] = current_ctx.involuntary
                
                # 计算每秒上下文切换次数（如果有历史数据）
                if len(self.metrics_log) > 0:
                    last_metric = self.metrics_log[-1]
                    if "ctx_switches" in last_metric:
                        time_diff = metrics["timestamp"] - last_metric["timestamp"]
                        if time_diff > 0:
                            vol_rate = (current_ctx.voluntary - 
                                      last_metric["ctx_switches"]["voluntary"]) / time_diff
                            invol_rate = (current_ctx.involuntary - 
                                        last_metric["ctx_switches"]["involuntary"]) / time_diff
                            metrics["ctx_switches"]["vol_rate_per_sec"] = vol_rate
                            metrics["ctx_switches"]["invol_rate_per_sec"] = invol_rate
            
            # 获取GPU使用情况
            if self.gpu_count > 0:
                for handle in self.gpu_handles:
                    try:
                        # GPU计算利用率
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        metrics["gpu_usage"].append(util.gpu)
                        
                        # GPU内存使用情况
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        metrics["gpu_memory"].append({
                            "used": mem_info.used,
                            "total": mem_info.total,
                            "percent": (mem_info.used / mem_info.total) * 100
                        })
                        
                        # SM流处理器利用率（部分GPU支持）
                        try:
                            sm_util = pynvml.nvmlDeviceGetUtilizationRates(handle).sm
                            if "gpu_sm" not in metrics:
                                metrics["gpu_sm"] = []
                            metrics["gpu_sm"].append(sm_util)
                        except:
                            pass
                        
                        # 显存带宽使用（当前值，非绝对带宽）
                        try:
                            throughput = pynvml.nvmlDeviceGetPcieThroughput(
                                handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
                            )
                            if "gpu_pcie_throughput" not in metrics:
                                metrics["gpu_pcie_throughput"] = []
                            metrics["gpu_pcie_throughput"].append(throughput)
                        except:
                            pass
                        
                    except pynvml.NVMLError:
                        metrics["gpu_usage"].append(0)
                        metrics["gpu_memory"].append({"used": 0, "total": 0, "percent": 0})
                        
            self.metrics_log.append(metrics)
            time.sleep(0.1)  # 采样间隔
    
    def _parse_perf_output(self, perf_output):
        """解析perf工具输出，提取缓存性能指标"""
        if not perf_output:
            return
        
        # 匹配L1 缓存未命中
        l1_match = re.search(r'([\d,]+)\s+L1-dcache-load-misses', perf_output)
        if l1_match:
            self.cache_metrics["L1_miss"] = int(l1_match.group(1).replace(',', ''))
            
        # 匹配最后级缓存未命中
        llc_match = re.search(r'([\d,]+)\s+LLC-load-misses', perf_output)
        if llc_match:
            self.cache_metrics["LLC_miss"] = int(llc_match.group(1).replace(',', ''))
            
        # 匹配指令数
        instr_match = re.search(r'([\d,]+)\s+instructions', perf_output)
        if instr_match:
            self.cache_metrics["instructions"] = int(instr_match.group(1).replace(',', ''))
            
        # 计算缓存命中率（如果有指令数据）
        if self.cache_metrics["instructions"] > 0:
            # 估算L1缓存命中率（近似值）
            self.cache_metrics["L1_hit_ratio"] = max(0, 1 - (
                self.cache_metrics["L1_miss"] / self.cache_metrics["instructions"]
            ))
            
            # 估算LLC命中率（从L1未命中中）
            if self.cache_metrics["L1_miss"] > 0:
                self.cache_metrics["LLC_hit_ratio"] = max(0, 1 - (
                    self.cache_metrics["LLC_miss"] / self.cache_metrics["L1_miss"]
                ))
            
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
            
            # CPU负载均衡性分析
            metrics["avg_cpu_load_balance_std"] = np.mean(
                [m["cpu_load_balance"]["std_dev"] for m in window_metrics]
            )
            metrics["avg_cpu_load_balance_max_diff"] = np.mean(
                [m["cpu_load_balance"]["max_diff"] for m in window_metrics]
            )
            
            # 上下文切换（如果有数据）
            ctx_data = [m.get("ctx_switches", {}) for m in window_metrics]
            if ctx_data and ctx_data[0]:
                # 计算整个窗口期间的上下文切换次数
                if "voluntary" in ctx_data[0] and "voluntary" in ctx_data[-1]:
                    metrics["ctx_switch_voluntary_total"] = (
                        ctx_data[-1]["voluntary"] - ctx_data[0]["voluntary"]
                    )
                    metrics["ctx_switch_involuntary_total"] = (
                        ctx_data[-1]["involuntary"] - ctx_data[0]["involuntary"]
                    )
                
                # 计算每秒平均上下文切换率
                vol_rates = [d.get("vol_rate_per_sec", 0) for d in ctx_data if "vol_rate_per_sec" in d]
                invol_rates = [d.get("invol_rate_per_sec", 0) for d in ctx_data if "invol_rate_per_sec" in d]
                
                if vol_rates:
                    metrics["ctx_switch_voluntary_per_sec"] = np.mean(vol_rates)
                if invol_rates:
                    metrics["ctx_switch_involuntary_per_sec"] = np.mean(invol_rates)
            
            # 内存使用率
            metrics["avg_memory_usage"] = np.mean([m["memory_usage"] for m in window_metrics])
            metrics["max_memory_usage"] = np.max([m["memory_usage"] for m in window_metrics])
            
            # GPU使用率(如果有)
            if self.gpu_count > 0:
                # 基本GPU使用率
                gpu_usage = [m["gpu_usage"] for m in window_metrics if m["gpu_usage"]]
                if gpu_usage:
                    metrics["avg_gpu_usage"] = np.mean([np.mean(u) for u in gpu_usage if u])
                    metrics["max_gpu_usage"] = np.max([np.max(u) for u in gpu_usage if u])
                
                # GPU内存使用率
                gpu_mem = [m["gpu_memory"] for m in window_metrics if m["gpu_memory"]]
                if gpu_mem:
                    metrics["avg_gpu_memory_percent"] = np.mean(
                        [np.mean([g["percent"] for g in batch]) for batch in gpu_mem if batch]
                    )
                    metrics["max_gpu_memory_percent"] = np.max(
                        [np.max([g["percent"] for g in batch]) for batch in gpu_mem if batch]
                    )
                
                # SM流处理器利用率（如果有）
                gpu_sm = [m.get("gpu_sm", []) for m in window_metrics if m.get("gpu_sm")]
                if gpu_sm and any(gpu_sm):
                    metrics["avg_gpu_sm_usage"] = np.mean([np.mean(u) for u in gpu_sm if u])
                    metrics["max_gpu_sm_usage"] = np.max([np.max(u) for u in gpu_sm if u])
                
                # PCIe传输带宽（如果有）
                gpu_pcie = [m.get("gpu_pcie_throughput", []) for m in window_metrics 
                           if m.get("gpu_pcie_throughput")]
                if gpu_pcie and any(gpu_pcie):
                    metrics["avg_gpu_pcie_throughput"] = np.mean(
                        [np.mean(u) for u in gpu_pcie if u]
                    )
                    metrics["max_gpu_pcie_throughput"] = np.max(
                        [np.max(u) for u in gpu_pcie if u]
                    )
                    
        return metrics
