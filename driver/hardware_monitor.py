import psutil
import pynvml
import numpy as np
from threading import Event, Thread
from typing import Dict, List, Any, Optional
import time

class HardwareMonitor:
    def __init__(self):
        # 获取当前进程
        self.process = psutil.Process()
        
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
            return f"{cc_major}.{cc_minor}"
        except:
            return ""

    def start_monitoring(self, interval: float = 0.1):
        """启动后台监控线程"""
        self._stop_event.clear()
        self.metrics_log.clear()
        self.monitor_thread = Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True  # 设为守护线程，防止主进程结束时线程仍在运行
        self.monitor_thread.start()

    def stop_monitoring(self):
        """停止监控"""
        self._stop_event.set()
        if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)  # 添加超时防止无限等待

    def _get_process_memory(self) -> int:
        """获取当前进程及其子进程的内存使用情况（RSS，以字节为单位）"""
        try:
            # 获取当前进程内存
            mem_info = self.process.memory_info().rss
            
            # 获取子进程内存（如果存在）
            try:
                children = self.process.children(recursive=True)
                for child in children:
                    try:
                        mem_info += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass  # 忽略已经结束或无权访问的进程
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
            return mem_info
        except Exception as e:
            print(f"内存监控错误: {e}")
            return 0

    def _monitor_loop(self, interval: float):
        """监控循环（含详细CPU负载指标）"""
        while not self._stop_event.is_set():
            try:
                # 获取进程内存使用
                mem_usage = self._get_process_memory()
                
                metrics = {
                    'timestamp': time.time(),
                    'cpu_usage': psutil.cpu_percent(interval=0.1),
                    'cpu_per_core': psutil.cpu_percent(interval=0.1, percpu=True),
                    'cpu_ctx_switches': psutil.cpu_stats().ctx_switches,
                    'cpu_interrupts': psutil.cpu_stats().interrupts,
                    'mem_usage': mem_usage,  # 进程内存使用（RSS）
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
            except Exception as e:
                print(f"监控循环错误: {e}")
            
            time.sleep(interval)

    def generate_report(self, task_name: str, phase_times: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        生成包含负载均衡分析的完整报告
        
        Args:
            task_name: 任务名称
            phase_times: 可选的时间范围，格式为 {'bfs_start': 时间戳, 'bfs_end': 时间戳}
        """
        if not self.metrics_log:
            return {"error": "没有收集到监控数据"}

        # 如果提供了时间范围，筛选指定时间范围内的监控数据
        filtered_metrics = self.metrics_log
        if phase_times and 'bfs_start' in phase_times and 'bfs_end' in phase_times:
            start_time = phase_times['bfs_start']
            end_time = phase_times['bfs_end']
            filtered_metrics = [
                m for m in self.metrics_log 
                if start_time <= m['timestamp'] <= end_time
            ]
            if not filtered_metrics:
                print("警告: 在指定时间范围内未找到监控数据，使用全部数据")
                filtered_metrics = self.metrics_log

        # CPU分析
        cpu_usage = np.array([m['cpu_usage'] for m in filtered_metrics])
        avg_cpu = np.mean(cpu_usage) if len(cpu_usage) > 0 else 0
        max_cpu = np.max(cpu_usage) if len(cpu_usage) > 0 else 0

        # 核心级负载均衡分析
        core_metrics = []
        if filtered_metrics and 'cpu_per_core' in filtered_metrics[0] and len(filtered_metrics[0]['cpu_per_core']) > 0:
            core_usage_data = [m['cpu_per_core'] for m in filtered_metrics]
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
        total_ctx_switches = 0
        total_interrupts = 0
        if len(filtered_metrics) > 1:
            total_ctx_switches = filtered_metrics[-1]['cpu_ctx_switches'] - filtered_metrics[0]['cpu_ctx_switches']
            total_interrupts = filtered_metrics[-1]['cpu_interrupts'] - filtered_metrics[0]['cpu_interrupts']

        # 内存分析 - 使用十进制单位 (GB)
        mem_usage_values = [m['mem_usage'] for m in filtered_metrics]
        max_mem = max(mem_usage_values) if mem_usage_values else 0
        avg_mem = np.mean(mem_usage_values) if mem_usage_values else 0
        
        # 最终内存 - 监控结束时的内存值
        final_mem = filtered_metrics[-1]['mem_usage'] if filtered_metrics else 0

        # GPU分析
        avg_gpu, max_gpu = 0.0, 0.0
        avg_gpu_mem, max_gpu_mem = 0.0, 0.0
        if self.gpu_count > 0 and filtered_metrics:
            # GPU使用率
            gpu_usage = [u for m in filtered_metrics for u in m['gpu_usage']]
            avg_gpu = np.mean(gpu_usage) if gpu_usage else 0.0
            max_gpu = np.max(gpu_usage) if gpu_usage else 0.0
            
            # GPU内存
            if 'gpu_mem' in filtered_metrics[0] and filtered_metrics[0]['gpu_mem']:
                gpu_mem_usage = [m['gpu_mem'][0] for m in filtered_metrics if m['gpu_mem']]
                avg_gpu_mem = np.mean(gpu_mem_usage) / 1e9 if gpu_mem_usage else 0.0  # 转换为GB
                max_gpu_mem = np.max(gpu_mem_usage) / 1e9 if gpu_mem_usage else 0.0   # 转换为GB

        # 添加时间窗口信息
        time_window = {}
        if phase_times and 'bfs_start' in phase_times and 'bfs_end' in phase_times:
            time_window = {
                'start': phase_times['bfs_start'],
                'end': phase_times['bfs_end'],
                'duration_ms': int((phase_times['bfs_end'] - phase_times['bfs_start']) * 1000)
            }

        return {
            'task_name': task_name,
            'time_window': time_window,
            'hardware': {
                'cpu_cores': self.cpu_cores,
                'cpu_threads': self.cpu_threads,
                'gpu_count': self.gpu_count,
                'cuda_cc': self._get_cuda_compute_capability(),
                'total_mem': f"{self.mem_total / 1e9:.1f} GB"  # 十进制单位
            },
            'metrics': {
                'avg_cpu': f"{avg_cpu:.1f}%",
                'max_cpu': f"{max_cpu:.1f}%",
                'cpu_load_balance': core_metrics,
                'cpu_contention': {
                    'total_ctx_switches': total_ctx_switches,
                    'total_interrupts': total_interrupts
                },
                'memory': {
                    'max_mem': f"{max_mem / 1e9:.2f} GB",  # 十进制单位
                    'avg_mem': f"{avg_mem / 1e9:.2f} GB",  # 十进制单位
                    'final_mem': f"{final_mem / 1e9:.2f} GB",  # 十进制单位
                    'type': "RSS (进程实际占用物理内存)"
                },
                'gpu': {
                    'avg_usage': f"{avg_gpu:.1f}%" if self.gpu_count else "N/A",
                    'max_usage': f"{max_gpu:.1f}%" if self.gpu_count else "N/A",
                    'avg_mem': f"{avg_gpu_mem:.2f} GB" if self.gpu_count else "N/A",
                    'max_mem': f"{max_gpu_mem:.2f} GB" if self.gpu_count else "N/A"
                }
            }
        }

    def get_current_stats(self) -> Dict[str, Any]:
        """返回当前资源使用状态的快照"""
        mem_usage = self._get_process_memory()
        
        # 基本CPU信息
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        
        # GPU信息
        gpu_stats = []
        if self.gpu_count > 0:
            for i, handle in enumerate(self.gpu_handles):
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_stats.append({
                        "index": i,
                        "usage": f"{util.gpu}%",
                        "memory": f"{mem_info.used / 1e9:.2f} GB / {mem_info.total / 1e9:.2f} GB"
                    })
                except pynvml.NVMLError:
                    gpu_stats.append({"index": i, "error": "无法读取GPU信息"})
        
        return {
            "cpu": {
                "total": f"{cpu_percent:.1f}%",
                "per_core": [f"{p:.1f}%" for p in cpu_per_core]
            },
            "memory": {
                "process": f"{mem_usage / 1e9:.2f} GB",  # 进程内存（GB）
                "system_total": f"{psutil.virtual_memory().total / 1e9:.1f} GB",  # 系统总内存
                "system_used": f"{psutil.virtual_memory().used / 1e9:.1f} GB"  # 系统已用内存
            },
            "gpu": gpu_stats
        }
