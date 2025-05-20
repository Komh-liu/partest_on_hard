import os
import random
import numpy as np
import networkx as nx
import argparse
import json
import glob
import concurrent.futures
from datetime import datetime

def generate_batch_array_sum_tests(base_path, batch_id=None, num_tests=10, configs=None):
    """
    批量生成数组求和测试数据，可以测试不同规模、不同数据分布
    
    参数:
        base_path: 项目的根路径
        batch_id: 批次ID
        num_tests: 要生成的测试数量
        configs: 测试配置列表，如果为None，将自动生成
    """
    if batch_id is None:
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\n===== 生成数组求和批量测试 [{batch_id}] =====")
    
    # 确保dataset目录存在
    dataset_dir = os.path.join(base_path, 'dataset', 'array_sum')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
    
    # 如果配置为空，自动生成多种配置
    if configs is None:
        configs = []
        
        # 不同大小的随机数组
        sizes = [1000, 10000, 100000, 1000000, 10000000]
        for size in sizes:
            configs.append({
                "size": size,
                "data_type": "random",
                "value_range": (-1000000, 1000000),
                "description": f"随机整数数组 (大小: {size})"
            })
        
        # 不同分布类型
        distributions = ["uniform", "normal", "skewed", "zeros", "all_same"]
        for dist in distributions:
            configs.append({
                "size": 100000,
                "data_type": dist,
                "value_range": (-1000000, 1000000),
                "description": f"{dist}分布数组 (大小: 100000)"
            })
        
        # 极端值测试
        configs.append({
            "size": 10000,
            "data_type": "random",
            "value_range": (-9223372036854775807, 9223372036854775807),  # 接近long long最大值
            "description": "极大值范围测试"
        })
        
        # 边界测试
        configs.append({
            "size": 1,
            "data_type": "random",
            "value_range": (-100, 100),
            "description": "单元素数组"
        })
        
        # 随机选择配置来达到所需的测试数量
        if len(configs) > num_tests:
            configs = random.sample(configs, num_tests)
        else:
            # 不足则补充随机配置
            while len(configs) < num_tests:
                size = random.choice([5000, 20000, 50000, 200000, 500000])
                data_type = random.choice(["random", "uniform", "normal", "skewed"])
                configs.append({
                    "size": size,
                    "data_type": data_type,
                    "value_range": (-1000000, 1000000),
                    "description": f"随机生成配置 {len(configs)+1}"
                })
    
    # 导入测试数据生成函数
    # 这里假设已经有一个更新后的数据生成脚本
    from advanced_test_generator import create_array_sum_data
    
    # 批量测试元数据
    batch_metadata = {
        "batch_id": batch_id,
        "task": "array_sum",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_tests": len(configs),
        "tests": []
    }
    
    # 生成所有测试
    for i, config in enumerate(configs):
        test_id = f"{batch_id}_array_{i+1}"
        print(f"\n[{i+1}/{len(configs)}] 生成测试: {config['description']}")
        
        try:
            # 生成测试数据
            test_id, file_path, result_path = create_array_sum_data(
                base_path,
                test_id=test_id,
                size=config["size"],
                data_type=config["data_type"],
                value_range=config["value_range"],
                backup=False
            )
            
            # 添加到批量测试元数据
            test_metadata = {
                "test_id": test_id,
                "description": config["description"],
                "size": config["size"],
                "data_type": config["data_type"],
                "value_range": config["value_range"],
                "file_path": file_path,
                "result_path": result_path
            }
            batch_metadata["tests"].append(test_metadata)
            
        except Exception as e:
            print(f"生成测试 {test_id} 时出错: {e}")
    
    # 保存批量测试元数据
    metadata_path = os.path.join(base_path, 'dataset', f"array_sum_batch_{batch_id}.json")
    with open(metadata_path, 'w') as f:
        json.dump(batch_metadata, f, indent=2)
    
    print(f"\n数组求和批量测试生成完毕，共 {len(batch_metadata['tests'])} 个测试")
    print(f"批量测试元数据保存至: {metadata_path}")
    
    return batch_id, batch_metadata

def generate_batch_matrix_multiply_tests(base_path, batch_id=None, num_tests=10, configs=None):
    """
    批量生成矩阵乘法测试数据，可以测试不同矩阵大小、类型和密度
    
    参数:
        base_path: 项目的根路径
        batch_id: 批次ID
        num_tests: 要生成的测试数量
        configs: 测试配置列表，如果为None，将自动生成
    """
    if batch_id is None:
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\n===== 生成矩阵乘法批量测试 [{batch_id}] =====")
    
    # 确保dataset目录存在
    dataset_dir = os.path.join(base_path, 'dataset', 'matrix_multiply')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
    
    # 如果配置为空，自动生成多种配置
    if configs is None:
        configs = []
        
        # 不同大小的矩阵
        sizes = [50, 100, 300, 500, 1000]
        for size in sizes:
            # 随着矩阵变大，降低密度
            density = min(0.1, 1000 / (size * size))
            configs.append({
                "size": size,
                "density": density,
                "matrix_type": "random",
                "value_range": (-10, 10),
                "description": f"随机稀疏矩阵 (大小: {size}×{size}, 密度: {density:.4f})"
            })
        
        # 不同类型的矩阵
        matrix_types = ["diagonal", "block", "symmetric", "banded"]
        for mtype in matrix_types:
            configs.append({
                "size": 300,
                "density": 0.05,
                "matrix_type": mtype,
                "value_range": (-10, 10),
                "description": f"{mtype}类型矩阵 (大小: 300×300)"
            })
        
        # 极端密度测试
        configs.append({
            "size": 100,
            "density": 0.8,
            "matrix_type": "random",
            "value_range": (-10, 10),
            "description": "高密度矩阵 (大小: 100×100, 密度: 0.8)"
        })
        
        configs.append({
            "size": 500,
            "density": 0.001,
            "matrix_type": "random",
            "value_range": (-10, 10),
            "description": "极低密度矩阵 (大小: 500×500, 密度: 0.001)"
        })
        
        # 随机选择配置来达到所需的测试数量
        if len(configs) > num_tests:
            configs = random.sample(configs, num_tests)
        else:
            # 不足则补充随机配置
            while len(configs) < num_tests:
                size = random.choice([150, 250, 400, 600, 800])
                density = min(0.1, 1000 / (size * size))
                matrix_type = random.choice(["random", "diagonal", "symmetric", "banded"])
                configs.append({
                    "size": size,
                    "density": density,
                    "matrix_type": matrix_type,
                    "value_range": (-10, 10),
                    "description": f"随机生成配置 {len(configs)+1}"
                })
    
    # 导入测试数据生成函数
    from advanced_test_generator import create_matrix_multiply_data
    
    # 批量测试元数据
    batch_metadata = {
        "batch_id": batch_id,
        "task": "matrix_multiply",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_tests": len(configs),
        "tests": []
    }
    
    # 生成所有测试
    for i, config in enumerate(configs):
        test_id = f"{batch_id}_matrix_{i+1}"
        print(f"\n[{i+1}/{len(configs)}] 生成测试: {config['description']}")
        
        try:
            # 生成测试数据
            test_id, file_path, result_path = create_matrix_multiply_data(
                base_path,
                test_id=test_id,
                n=config["size"],
                density=config["density"],
                matrix_type=config["matrix_type"],
                value_range=config["value_range"],
                backup=False
            )
            
            # 添加到批量测试元数据
            test_metadata = {
                "test_id": test_id,
                "description": config["description"],
                "size": config["size"],
                "density": config["density"],
                "matrix_type": config["matrix_type"],
                "value_range": config["value_range"],
                "file_path": file_path,
                "result_path": result_path
            }
            batch_metadata["tests"].append(test_metadata)
            
        except Exception as e:
            print(f"生成测试 {test_id} 时出错: {e}")
    
    # 保存批量测试元数据
    metadata_path = os.path.join(base_path, 'dataset', f"matrix_multiply_batch_{batch_id}.json")
    with open(metadata_path, 'w') as f:
        json.dump(batch_metadata, f, indent=2)
    
    print(f"\n矩阵乘法批量测试生成完毕，共 {len(batch_metadata['tests'])} 个测试")
    print(f"批量测试元数据保存至: {metadata_path}")
    
    return batch_id, batch_metadata

def generate_batch_graph_bfs_tests(base_path, batch_id=None, num_tests=10, configs=None):
    """
    批量生成图的BFS遍历测试数据，可以测试不同图结构和规模
    
    参数:
        base_path: 项目的根路径
        batch_id: 批次ID
        num_tests: 要生成的测试数量
        configs: 测试配置列表，如果为None，将自动生成
    """
    if batch_id is None:
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\n===== 生成图的BFS遍历批量测试 [{batch_id}] =====")
    
    # 确保dataset目录存在
    dataset_dir = os.path.join(base_path, 'dataset', 'graph_bfs')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
    
    # 如果配置为空，自动生成多种配置
    if configs is None:
        configs = []
        
        # 不同大小的随机图
        sizes = [100, 500, 1000, 5000, 10000]
        for size in sizes:
            # 随着图变大，降低边概率
            edge_prob = min(0.1, 10.0 / size)
            configs.append({
                "num_vertices": size,
                "edge_probability": edge_prob,
                "graph_type": "random",
                "start_vertex": 1,
                "description": f"随机图 (顶点数: {size}, 边概率: {edge_prob:.6f})"
            })
        
        # 不同类型的图
        graph_types = ["tree", "small_world", "scale_free", "complete"]
        for gtype in graph_types:
            # 完全图使用较小的顶点数
            if gtype == "complete":
                size = 50
                edge_prob = 1.0
            else:
                size = 500
                edge_prob = 0.01
            
            configs.append({
                "num_vertices": size,
                "edge_probability": edge_prob,
                "graph_type": gtype,
                "start_vertex": 1,
                "description": f"{gtype}类型图 (顶点数: {size})"
            })
        
        # 不同起始顶点的测试
        start_vertices = [1, 5, 10, 50, 100]
        for sv in start_vertices[1:]:  # 跳过1，因为之前已经有1作为起点的测试
            configs.append({
                "num_vertices": 300,
                "edge_probability": 0.05,
                "graph_type": "random",
                "start_vertex": sv,
                "description": f"不同起始顶点 (起点: {sv}, 顶点数: 300)"
            })
        
        # 随机选择配置来达到所需的测试数量
        if len(configs) > num_tests:
            configs = random.sample(configs, num_tests)
        else:
            # 不足则补充随机配置
            while len(configs) < num_tests:
                size = random.choice([200, 400, 600, 1500, 3000])
                edge_prob = min(0.1, 10.0 / size)
                graph_type = random.choice(["random", "tree", "small_world", "scale_free"])
                start_vertex = random.randint(1, min(size // 10, 100))
                configs.append({
                    "num_vertices": size,
                    "edge_probability": edge_prob,
                    "graph_type": graph_type,
                    "start_vertex": start_vertex,
                    "description": f"随机生成配置 {len(configs)+1}"
                })
    
    # 导入测试数据生成函数
    from advanced_test_generator import create_graph_bfs_data
    
    # 批量测试元数据
    batch_metadata = {
        "batch_id": batch_id,
        "task": "graph_bfs",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_tests": len(configs),
        "tests": []
    }
    
    # 生成所有测试
    for i, config in enumerate(configs):
        test_id = f"{batch_id}_graph_{i+1}"
        print(f"\n[{i+1}/{len(configs)}] 生成测试: {config['description']}")
        
        try:
            # 生成测试数据
            test_id, file_path, result_path = create_graph_bfs_data(
                base_path,
                test_id=test_id,
                num_vertices=config["num_vertices"],
                edge_probability=config["edge_probability"],
                graph_type=config["graph_type"],
                start_vertex=config["start_vertex"],
                backup=False
            )
            
            # 添加到批量测试元数据
            test_metadata = {
                "test_id": test_id,
                "description": config["description"],
                "num_vertices": config["num_vertices"],
                "edge_probability": config["edge_probability"],
                "graph_type": config["graph_type"],
                "start_vertex": config["start_vertex"],
                "file_path": file_path,
                "result_path": result_path
            }
            batch_metadata["tests"].append(test_metadata)
            
        except Exception as e:
            print(f"生成测试 {test_id} 时出错: {e}")
    
    # 保存批量测试元数据
    metadata_path = os.path.join(base_path, 'dataset', f"graph_bfs_batch_{batch_id}.json")
    with open(metadata_path, 'w') as f:
        json.dump(batch_metadata, f, indent=2)
    
    print(f"\n图的BFS遍历批量测试生成完毕，共 {len(batch_metadata['tests'])} 个测试")
    print(f"批量测试元数据保存至: {metadata_path}")
    
    return batch_id, batch_metadata

def generate_all_test_batches(base_path, num_tests=10, use_parallel=True):
    """
    生成所有任务的测试批次
    
    参数:
        base_path: 项目的根路径
        num_tests: 每个任务生成的测试数量
        use_parallel: 是否并行生成不同任务的测试
    """
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 确保所有目录存在
    for task in ['array_sum', 'matrix_multiply', 'graph_bfs']:
        dataset_dir = os.path.join(base_path, 'dataset', task)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir, exist_ok=True)
        
        driver_dir = os.path.join(base_path, 'driver', task)
        if not os.path.exists(driver_dir):
            os.makedirs(driver_dir, exist_ok=True)
    
    # 定义任务函数映射
    batch_generators = {
        'array_sum': generate_batch_array_sum_tests,
        'matrix_multiply': generate_batch_matrix_multiply_tests,
        'graph_bfs': generate_batch_graph_bfs_tests
    }
    
    results = {}
    
    if use_parallel:
        # 并行生成测试
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(3, os.cpu_count())) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(
                    batch_generators[task], 
                    base_path, 
                    f"{batch_id}_{task}", 
                    num_tests
                ): task for task in batch_generators
            }
            
            # 获取结果
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    batch_id, batch_metadata = future.result()
                    results[task] = {
                        'batch_id': batch_id,
                        'batch_metadata': batch_metadata
                    }
                except Exception as e:
                    print(f"生成 {task} 任务批量测试时出错: {e}")
    else:
        # 串行生成测试
        for task, generator in batch_generators.items():
            try:
                task_batch_id, batch_metadata = generator(
                    base_path, 
                    f"{batch_id}_{task}", 
                    num_tests
                )
                results[task] = {
                    'batch_id': task_batch_id,
                    'batch_metadata': batch_metadata
                }
            except Exception as e:
                print(f"生成 {task} 任务批量测试时出错: {e}")
    
    # 创建总体批次元数据
    global_metadata = {
        "global_batch_id": batch_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tasks": {},
        "total_tests": 0
    }
    
    for task, result in results.items():
        global_metadata["tasks"][task] = {
            "batch_id": result["batch_id"],
            "num_tests": len(result["batch_metadata"]["tests"]),
            "metadata_path": f"dataset/{task}_batch_{result['batch_id']}.json"
        }
        global_metadata["total_tests"] += len(result["batch_metadata"]["tests"])
    
    # 保存总体批次元数据
    metadata_path = os.path.join(base_path, 'dataset', f"global_batch_{batch_id}.json")
    with open(metadata_path, 'w') as f:
        json.dump(global_metadata, f, indent=2)
    
    print(f"\n所有批量测试生成完毕，全局批次ID: {batch_id}")
    print(f"总共生成 {global_metadata['total_tests']} 个测试")
    print(f"全局元数据保存至: {metadata_path}")
    
    return batch_id, global_metadata

def run_batch_tests(base_path, batch_id=None, task=None):
    """
    运行批量测试
    
    参数:
        base_path: 项目的根路径
        batch_id: 批次ID，如果为None则使用最新的批次
        task: 任务类型，如果为None则运行所有任务
    """
    # 如果没有指定批次ID，寻找最新的批次
    if batch_id is None:
        pattern = os.path.join(base_path, 'dataset', "global_batch_*.json")
        batch_files = sorted(glob.glob(pattern), reverse=True)
        if not batch_files:
            print("未找到任何批量测试元数据")
            return
        
        with open(batch_files[0], 'r') as f:
            global_metadata = json.load(f)
            batch_id = global_metadata["global_batch_id"]
            print(f"使用最新的批次: {batch_id}")
    else:
        # 加载指定批次
        metadata_path = os.path.join(base_path, 'dataset', f"global_batch_{batch_id}.json")
        try:
            with open(metadata_path, 'r') as f:
                global_metadata = json.load(f)
        except FileNotFoundError:
            print(f"未找到批次 {batch_id} 的元数据")
            return
    
    # 确定要运行的任务
    tasks_to_run = []
    if task is not None:
        if task in global_metadata["tasks"]:
            tasks_to_run.append(task)
        else:
            print(f"批次 {batch_id} 中未找到任务 {task}")
            return
    else:
        tasks_to_run = list(global_metadata["tasks"].keys())
    
    # 创建运行结果目录
    results_dir = os.path.join(base_path, 'results', f"batch_{batch_id}")
    os.makedirs(results_dir, exist_ok=True)
    
    # TODO: 实际的测试运行逻辑
    # 这部分需要根据你的driver.py的具体实现来完成
    # 以下是一个简单的模拟示例
    
    results = {
        "batch_id": batch_id,
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tasks": {}
    }
    
    for task_name in tasks_to_run:
        print(f"\n===== 运行 {task_name} 任务 =====")
        task_metadata_path = os.path.join(
            base_path, 
            'dataset', 
            f"{task_name}_batch_{global_metadata['tasks'][task_name]['batch_id']}.json"
        )
        
        with open(task_metadata_path, 'r') as f:
            task_metadata = json.load(f)
        
        task_results = []
        
        for test in task_metadata["tests"]:
            print(f"运行测试: {test['description']}")
            
            # 创建符号链接，使driver.py能够找到正确的文件
            data_link = os.path.join(base_path, 'driver', task_name, 'data.txt')
            result_link = os.path.join(base_path, 'driver', task_name, 'result.txt')
            
            if os.path.exists(data_link):
                os.remove(data_link)
            if os.path.exists(result_link):
                os.remove(result_link)
            
            try:
                # 根据操作系统，创建符号链接或复制文件
                if os.name == 'nt':  # Windows
                    import shutil
                    shutil.copy2(test['file_path'], data_link)
                    shutil.copy2(test['result_path'], result_link)
                else:  # Linux/Mac
                    os.symlink(test['file_path'], data_link)
                    os.symlink(test['result_path'], result_link)
                
                # 运行测试
                # 在实际实现中，应该调用你的driver.py或类似的测试运行脚本
                # subprocess.run(["python", "driver.py"], cwd=os.path.join(base_path, 'driver'))
                
                # 模拟测试结果
                test_result = {
                    "test_id": test["test_id"],
                    "description": test["description"],
                    "success": True,
                    "runtime_ms": random.randint(10, 1000),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                task_results.append(test_result)
                print(f"  成功 - 运行时间: {test_result['runtime_ms']}ms")
                
            except Exception as e:
                print(f"  失败 - 错误: {e}")
                task_results.append({
                    "test_id": test["test_id"],
                    "description": test["description"],
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
            finally:
                # 清理符号链接
                if os.path.exists(data_link):
                    os.remove(data_link)
                if os.path.exists(result_link):
                    os.remove(result_link)
        
        # 保存任务结果
        results["tasks"][task_name] = {
            "total_tests": len(task_results),
            "successful_tests": sum(1 for r in task_results if r["success"]),
            "test_results": task_results
        }
    
    # 保存总体结果
    result_path = os.path.join(results_dir, "batch_results.json")
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n批量测试运行完毕，结果保存至: {result_path}")
    
    # 打印摘要
    print("\n===== 测试摘要 =====")
    for task_name, task_result in results["tasks"].items():
        print(f"{task_name}: {task_result['successful_tests']}/{task_result['total_tests']} 成功")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='批量生成并行测试数据')
    parser.add_argument('--path', type=str, default='E:/third/Partest_on_hard', 
                        help='项目根目录路径')
    parser.add_argument('--mode', type=str, choices=['array', 'matrix', 'graph', 'all', 'run'], 
                        default='all', help='运行模式')
    parser.add_argument('--num_tests', type=int, default=10, 
                        help='每个任务生成的测试数量')
    parser.add_argument('--batch_id', type=str, default=None,
                        help='批次ID（仅在运行模式下使用）')
    parser.add_argument('--task', type=str, choices=['array_sum', 'matrix_multiply', 'graph_bfs'],
                        default=None, help='指定要运行的任务（仅在运行模式下使用）')
    
    args = parser.parse_args()
    
    # 获取项目根目录
    base_path = args.path.replace('\\', '/')
    
    if args.mode == 'all':
        generate_all_test_batches(base_path, args.num_tests)
    elif args.mode == 'array':
        generate_batch_array_sum_tests(base_path, num_tests=args.num_tests)
    elif args.mode == 'matrix':
        generate_batch_matrix_multiply_tests(base_path, num_tests=args.num_tests)
    elif args.mode == 'graph':
        generate_batch_graph_bfs_tests(base_path, num_tests=args.num_tests)
    elif args.mode == 'run':
        run_batch_tests(base_path, args.batch_id, args.task)