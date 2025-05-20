import json
import subprocess
import os
import tempfile
import shutil
import time
import re  # 导入正则表达式模块
import matplotlib.pyplot as plt
from hardware_monitor import HardwareMonitor

def list_files_in_directory(directory):
    """列出指定目录中的所有文件和文件夹"""
    files = os.listdir(directory)
    print(f"目录内容 ({directory}):")
    for file in files:
        print(f"  {file}")

def extract_and_compile(metadata, current_dir, temp_dir):
    framework = metadata['framework']
    task_type = metadata['task_type']  # 获取任务类型

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
    else:
        header_file_name = 'single_thread_impl.h'

    header_file_path = os.path.join(temp_dir, header_file_name)

    # 使用头文件保护机制避免重复定义
    code_content = metadata['code'].strip().replace("```cpp", "").replace("```", "")
    protected_code = f"#ifndef {header_file_name.replace('.', '_').upper()}\n#define {header_file_name.replace('.', '_').upper()}\n{code_content}\n#endif // {header_file_name.replace('.', '_').upper()}\n"

    # 将目标代码写入头文件
    with open(header_file_path, 'w') as header_file:
        header_file.write(protected_code)

    # 定义相对路径的测试文件夹
    relative_test_folder_path = task_type

    # 计算测试文件夹的绝对路径
    absolute_test_folder_path = os.path.join(current_dir, relative_test_folder_path)
    # 检查文件夹是否存在
    if not os.path.exists(absolute_test_folder_path):
        print(f"文件夹 {absolute_test_folder_path} 不存在，跳过此任务。")
        return

    # 复制整个f"{task_type}"文件夹到临时文件夹
    temp_test_folder_path = os.path.join(temp_dir, relative_test_folder_path)
    shutil.copytree(absolute_test_folder_path, temp_test_folder_path)

    # 根据framework选择头文件包含
    include_line = f'#include "{header_file_name}"'

    # 修改main.cpp以包含正确的头文件
    if framework == 'CUDA':
        main_cpp_path = os.path.join(temp_test_folder_path, 'main.cu')
    else:
        main_cpp_path = os.path.join(temp_test_folder_path, 'main.cpp')
    if not os.path.exists(main_cpp_path):
        print(f"文件 {main_cpp_path} 不存在，跳过此任务。")
        return

    with open(main_cpp_path, 'r') as main_file:
        lines = main_file.readlines()
    new_lines = []
    inserted = False
    # 检查是否已经存在于第二行
    if len(lines) > 1:
        line_2 = lines[1].strip()
        if line_2 == include_line.strip():
            # 第二行已经是目标行，不插入
            new_lines = lines
            inserted = True
        else:
            # 插入到第二行（索引为1）
            new_lines = lines[:1] + [include_line + '\n'] + lines[1:]
            inserted = True
    else:
        # 包括文件行数不足的情况（< 2 行），则插入到第二行
        if len(lines) == 0:
            # 文件为空的情况下直接插入
            new_lines = [include_line + '\n']
        elif len(lines) == 1:
            new_lines = [lines[0], include_line + '\n']
        else:
            # 正常情况（为防止不出错）
            new_lines = [include_line + '\n'] + lines
        inserted = True
    # 如果插入完成，写入文件
    if inserted:
        with open(main_cpp_path, 'w') as main_file:
            main_file.writelines(new_lines)

    # 开始硬件监控
    monitor.start_monitoring()

    # 根据框架调整编译命令
    if framework == 'OpenMP':
        compile_command = f"g++ -std=c++17 {main_cpp_path} -o {os.path.join(temp_dir, 'main')} -I{temp_dir} -fopenmp -DUSE_OPENMP"
        print("g++ OpenMP编译")
    elif framework == 'CUDA':
        main_cu_path = os.path.join(temp_test_folder_path, 'main.cu')
        compile_command = f"nvcc -std=c++17 {main_cu_path} -o {os.path.join(temp_dir, 'main')} -I{temp_dir} -lcudart -DUSE_CUDA"
        print("nvcc CUDA编译")
    elif framework == 'MPI':
        compile_command = f"mpicxx -std=c++17 {main_cpp_path} -o {os.path.join(temp_dir, 'main')} -I{temp_dir} -DUSE_MPI"
        print("mpicxx MPI编译")
    else:
        print("g++编译")
        # print(main_cpp_path)
        compile_command = f"g++ -std=c++17 {main_cpp_path} -o {os.path.join(temp_dir, 'main')} -I{temp_dir}"

    print(f"编译命令: {compile_command}")
    compile_result = subprocess.run(compile_command, shell=True, capture_output=True, text=True, cwd=temp_dir)
    print("编译结果输出:", compile_result.stdout)
    print("编译错误输出:", compile_result.stderr)

    # 检查编译是否成功
    if compile_result.returncode != 0:
        print("编译失败！")
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
            print("未检测到BFS时间戳标记，将使用完整运行时间分析")
            
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
        # 追加监控数据
        for key, value in report['metrics'].items():
            log_file.write(f'  {key}: {value}\n')
        
        # 如果存在BFS时间窗口信息，也记录下来
        if 'time_window' in report and report['time_window']:
            log_file.write(f"  核心代码执行时段: {report['time_window']['duration_ms']}ms\n")

    # 生成可视化报告
    generate_detailed_report(report, task_type, monitor)

    # 清理临时文件夹
    shutil.rmtree(temp_dir)

def generate_detailed_report(report: dict, task_name: str, monitor: HardwareMonitor):
    """生成可视化报告，并保存结构化数据"""
    if monitor.metrics_log:
        # 可视化 CPU 使用率曲线
        plt.figure(figsize=(10, 5))
        timestamps = [m['timestamp'] - monitor.metrics_log[0]['timestamp'] for m in monitor.metrics_log]
        plt.plot(timestamps, [m['cpu_usage'] for m in monitor.metrics_log], label='CPU Usage (%)')
        
        # 如果有BFS时间窗口，在图表中标记
        if 'time_window' in report and report['time_window']:
            start_rel = report['time_window']['start'] - monitor.metrics_log[0]['timestamp']
            end_rel = report['time_window']['end'] - monitor.metrics_log[0]['timestamp']
            plt.axvline(x=start_rel, color='r', linestyle='--', label='BFS Start')
            plt.axvline(x=end_rel, color='g', linestyle='--', label='BFS End')
            
        plt.xlabel("Time (seconds)")
        plt.ylabel("CPU Usage (%)")
        plt.title(f"CPU Utilization - {task_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{task_name}_cpu_usage.png")
        plt.close()

        # 可视化 GPU 使用率曲线（如果有 GPU）
        if report['hardware']['gpu_count'] > 0:
            plt.figure(figsize=(10, 5))
            gpu_data = [u['gpu_usage'][0] if u['gpu_usage'] else 0 for u in monitor.metrics_log]
            plt.plot(timestamps, gpu_data, label='GPU Usage (%)', color='orange')
            
            # 如果有BFS时间窗口，在图表中标记
            if 'time_window' in report and report['time_window']:
                plt.axvline(x=start_rel, color='r', linestyle='--', label='BFS Start')
                plt.axvline(x=end_rel, color='g', linestyle='--', label='BFS End')
                
            plt.xlabel("Time (seconds)")
            plt.ylabel("GPU Usage (%)")
            plt.title(f"GPU Utilization - {task_name}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{task_name}_gpu_usage.png")
            plt.close()

        # 保存结构化评估报告
        with open(f"{task_name}_report.json", 'w') as f:
            json.dump(report, f, indent=2)

        print(f"详细报告保存完毕: {task_name}_cpu_usage.png, {task_name}_gpu_usage.png, {task_name}_report.json")
    else:
        print(f"未能获取监控数据，跳过生成报告。")

# 定义JSON文件路径
json_file_path = 'output.json'

# 获取当前脚本所在的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 读取JSON文件内容
with open(json_file_path, 'r') as file:
    try:
        data = json.load(file)
    except FileNotFoundError:
        print("Error: output.json file not found.")
        exit(1)
    except json.JSONDecodeError:
        print("Error: output.json is not a valid JSON file.")
        exit(1)

# 遍历所有任务
if not data.get('tasks'):
    print("Error: No tasks found in output.json.")
    exit(1)

for task in data['tasks']:
    metadata = task['metadata']
    # 创建临时文件夹
    temp_dir = tempfile.mkdtemp()
    # 提取和编译
    extract_and_compile(metadata, current_dir, temp_dir)
