import json
import subprocess
import os
import tempfile
import shutil
import time
import re
import matplotlib.pyplot as plt
from hardware_monitor import HardwareMonitor
import sys

def json_serializable(obj):
    """将对象转换为 JSON 可序列化的形式"""
    if isinstance(obj, (list, dict, str, int, float, bool, type(None))):
        return obj
    return str(obj)

def make_json_serializable(data):
    """递归地将数据结构转换为 JSON 可序列化的形式"""
    if isinstance(data, dict):
        return {k: make_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [make_json_serializable(item) for item in data]
    else:
        return json_serializable(data)

def list_files_in_directory(directory):
    """列出指定目录中的所有文件和文件夹"""
    files = os.listdir(directory)
    print(f"目录内容 ({directory}):")
    for file in files:
        print(f"  {file}")

def extract_and_compile(metadata, current_dir, temp_dir, monitor_mode):
    framework = metadata['framework']
    task_type = metadata['task_type']

    # 初始化硬件监控
    monitor = HardwareMonitor()

    # 创建头文件路径
    if framework == 'Serial':
        header_file_name = 'single_thread_impl.h'
    elif framework == 'OpenMP':
        header_file_name = 'openmp_impl.h'
    elif framework == 'CUDA':
        header_file_name = 'cuda_impl.cu'
    elif framework == 'MPI':
        header_file_name = 'mpi_impl.h'
    elif framework == 'TBB':
        header_file_name = 'tbb_impl.h'
    else:
        header_file_name = 'single_thread_impl.h'

    header_file_path = os.path.join(temp_dir, header_file_name)

    # 使用头文件保护机制
    code_content = metadata['code'].strip().replace("```cpp", "").replace("```", "")
    protected_code = f"#ifndef {header_file_name.replace('.', '_').upper()}\n#define {header_file_name.replace('.', '_').upper()}\n{code_content}\n#endif"

    # 写入头文件
    with open(header_file_path, 'w') as header_file:
        header_file.write(protected_code)

    # 设置测试文件夹路径
    relative_test_folder_path = task_type
    absolute_test_folder_path = os.path.join(current_dir, relative_test_folder_path)
    
    if not os.path.exists(absolute_test_folder_path):
        print(f"文件夹 {absolute_test_folder_path} 不存在，跳过此任务。")
        return

    # 复制测试文件夹
    temp_test_folder_path = os.path.join(temp_dir, relative_test_folder_path)
    shutil.copytree(absolute_test_folder_path, temp_test_folder_path)

    # 设置头文件包含
    include_line = f'#include "{header_file_name}"'

    # 确定主文件路径
    main_cpp_path = os.path.join(temp_test_folder_path, 'main.cu' if framework == 'CUDA' else 'main.cpp')
    if not os.path.exists(main_cpp_path):
        print(f"文件 {main_cpp_path} 不存在，跳过此任务。")
        return

    # 修改主文件
    with open(main_cpp_path, 'r') as main_file:
        lines = main_file.readlines()

    # 插入头文件包含
    new_lines = []
    inserted = False
    if len(lines) > 1:
        if lines[1].strip() == include_line.strip():
            new_lines = lines
            inserted = True
        else:
            new_lines = lines[:1] + [include_line + '\n'] + lines[1:]
            inserted = True
    else:
        if len(lines) == 0:
            new_lines = [include_line + '\n']
        elif len(lines) == 1:
            new_lines = [lines[0], include_line + '\n']
        else:
            new_lines = [include_line + '\n'] + lines
        inserted = True

    if inserted:
        with open(main_cpp_path, 'w') as main_file:
            main_file.writelines(new_lines)

    # 开始监控
    monitor.start_monitoring()

    # 设置编译命令
    if framework == 'OpenMP':
        compile_command = f"g++ -std=c++17 {main_cpp_path} -o {os.path.join(temp_dir, 'main')} -I{temp_dir} -fopenmp -DUSE_OPENMP"
    elif framework == 'CUDA':
        compile_command = f"nvcc -std=c++17 {main_cpp_path} -o {os.path.join(temp_dir, 'main')} -I{temp_dir} -lcudart -DUSE_CUDA"
    elif framework == 'MPI':
        compile_command = f"mpicxx -std=c++17 {main_cpp_path} -o {os.path.join(temp_dir, 'main')} -I{temp_dir} -DUSE_MPI"
    elif framework == 'TBB':
        compile_command = f"g++ -std=c++17 {main_cpp_path} -o {os.path.join(temp_dir, 'main')} -I{temp_dir} -ltbb -DUSE_TBB"
    else:
        compile_command = f"g++ -std=c++17 {main_cpp_path} -o {os.path.join(temp_dir, 'main')} -I{temp_dir}"

    # 编译
    print(f"编译命令: {compile_command}")
    compile_result = subprocess.run(compile_command, shell=True, capture_output=True, text=True, cwd=temp_dir)
    
    if compile_result.returncode != 0:
        print("编译失败！")
        print(compile_result.stderr)
        log_content = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {framework} - {task_type} - 编译失败 - 运行时长: N/A"
        with open('log.txt', 'a') as log_file:
            log_file.write(log_content + '\n')
        monitor.stop_monitoring()
        return

    parent_path = os.path.dirname(current_dir)
    # 运行测试代码
    input_file = os.path.join(parent_path, 'dataset', task_type, 'data.txt')
    output_file = os.path.join(parent_path, 'driver', task_type, 'result.txt')
    run_command = f"./{os.path.basename(os.path.join(temp_dir, 'main'))} {input_file} {output_file}"
    start_time = time.time()

    try:
        run_result = subprocess.run(run_command, shell=True, capture_output=True, text=True, cwd=temp_dir, timeout=300)  # 设置超时时间为300秒（5分钟）
        output = run_result.stdout
        
        # 提取BFS核心代码执行时间戳
        phase_times = None
        start_match = re.search(r'\[METRICS\] BFS_TIME_START=(\d+)', output)
        end_match = re.search(r'\[METRICS\] BFS_TIME_END=(\d+)', output)
        
        if start_match and end_match:
            # 提取并转换为秒级时间戳
            phase_times = {
                "bfs_start": int(start_match.group(1)) / 1000.0,
                "bfs_end": int(end_match.group(1)) / 1000.0
            }
            print(f"核心代码时间窗口: {phase_times['bfs_start']} - {phase_times['bfs_end']} (持续时间: {(phase_times['bfs_end']-phase_times['bfs_start'])*1000:.1f}ms)")
        else:
            print("未检测到时间戳标记，将使用完整运行时间分析")
            
    except subprocess.TimeoutExpired:
        print("测试代码运行超时！")
        log_content = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {framework} - {task_type} - 运行超时 - 运行时长: {int((time.time() - start_time) * 1000)}ms"
        with open("log.txt", 'a') as log_file:
            log_file.write(log_content + '\n')
        monitor.stop_monitoring()
        shutil.rmtree(temp_dir)
        return

    end_time = time.time()
    runtime = int((end_time - start_time) * 1000)  # 转换为毫秒

    # 停止监控并生成报告
    monitor.stop_monitoring()
    report = monitor.generate_report(task_type, phase_times)

    # 检查运行结果
    if run_result.returncode != 0:
        print("测试代码运行失败！")
        print("错误信息：")
        print(run_result.stderr)
        log_content = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {framework} - {task_type} - 运行失败 - 运行时长: {runtime}ms"
    else:
        print("测试代码运行成功！")
        print("输出结果：")
        print(run_result.stdout)
        # 提取运行时间和验证成功与否的信息
        time_match = re.search(r"Time: (\d+)ms", run_result.stdout)
        success_match = re.search(r"验证成功", run_result.stdout)
        time_info = time_match.group(1) if time_match else "N/A"
        success_info = "验证成功" if success_match else "验证失败"
        log_content = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {framework} - {task_type} - 运行成功 - 运行时间: {time_info}ms - {success_info}"

    with open("log.txt", 'a') as log_file:
        log_file.write(log_content + '\n')
        if monitor_mode:
            # 追加监控数据
            for key, value in report['metrics'].items():
                log_file.write(f'  {key}: {value}\n')
            
            # 如果存在BFS时间窗口信息，也记录下来
            if 'time_window' in report and report['time_window']:
                log_file.write(f"  核心代码执行时段: {report['time_window']['duration_ms']}ms\n")

    # 生成可视化报告
    '''
    if monitor_mode:
        try:
            generate_detailed_report(report, task_type, monitor)
        except Exception as e:
            print(f"生成报告时出错: {str(e)}")
    '''
    # 清理临时文件
    shutil.rmtree(temp_dir)

def generate_detailed_report(report: dict, task_name: str, monitor: HardwareMonitor):
    """生成详细报告"""
    if monitor.metrics_log:
        # 绘制CPU使用率图
        plt.figure(figsize=(10, 5))
        timestamps = [float(m['timestamp'] - monitor.metrics_log[0]['timestamp']) for m in monitor.metrics_log]
        plt.plot(timestamps, [float(m['cpu_usage']) for m in monitor.metrics_log], label='CPU Usage (%)')
        
        # 添加BFS时间窗口标记
        if 'time_window' in report and report['time_window']:
            start_rel = float(report['time_window']['start'] - monitor.metrics_log[0]['timestamp'])
            end_rel = float(report['time_window']['end'] - monitor.metrics_log[0]['timestamp'])
            plt.axvline(x=start_rel, color='r', linestyle='--', label='BFS Start')
            plt.axvline(x=end_rel, color='g', linestyle='--', label='BFS End')
            
        plt.xlabel("Time (seconds)")
        plt.ylabel("CPU Usage (%)")
        plt.title(f"CPU Utilization - {task_name}")
        plt.legend()
        plt.tight_layout()
        # plt.savefig(f"{task_name}_cpu_usage.png")
        plt.close()

        # 如果有GPU，绘制GPU使用率图
        if report['hardware']['gpu_count'] > 0:
            plt.figure(figsize=(10, 5))
            gpu_data = [float(u['gpu_usage'][0]) if u['gpu_usage'] else 0.0 for u in monitor.metrics_log]
            plt.plot(timestamps, gpu_data, label='GPU Usage (%)', color='orange')
            
            if 'time_window' in report and report['time_window']:
                plt.axvline(x=start_rel, color='r', linestyle='--', label='BFS Start')
                plt.axvline(x=end_rel, color='g', linestyle='--', label='BFS End')
                
            plt.xlabel("Time (seconds)")
            plt.ylabel("GPU Usage (%)")
            plt.title(f"GPU Utilization - {task_name}")
            plt.legend()
            plt.tight_layout()
            # plt.savefig(f"{task_name}_gpu_usage.png")
            plt.close()

        # 保存JSON报告
        try:
            serializable_report = make_json_serializable(report)
            with open(f"{task_name}_report.json", 'w') as f:
                print()
                #json.dump(serializable_report, f, indent=2)
        except Exception as e:
            print(f"保存JSON报告时出错: {str(e)}")

def main():
    # 检查是否有'm'参数
    monitor_mode = '-m' in sys.argv

    # 定义JSON文件路径
    json_file_path = 'output.json'
    
    # 获取当前脚本所在的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 读取JSON文件
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print("Error: output.json file not found.")
        return
    except json.JSONDecodeError:
        print("Error: output.json is not a valid JSON file.")
        return

    # 检查任务
    if not data.get('tasks'):
        print("Error: No tasks found in output.json.")
        return

    # 处理每个任务
    for task in data['tasks']:
        metadata = task['metadata']
        temp_dir = tempfile.mkdtemp()
        extract_and_compile(metadata, current_dir, temp_dir, monitor_mode)

if __name__ == "__main__":
    main()
