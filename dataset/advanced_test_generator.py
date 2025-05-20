import os
import random
import numpy as np
import networkx as nx
import shutil
import argparse
import json
from datetime import datetime

def create_array_sum_data(base_path, test_id=None, size=100000, data_type='random', value_range=(-1000000, 1000000), 
                          backup=True):
    """
    生成数组求和任务的测试数据
    
    参数:
        base_path: 项目的根路径
        test_id: 测试ID，用于生成唯一文件名
        size: 数组大小
        data_type: 数据类型 (random, uniform, normal, skewed, zeros, all_same)
        value_range: 数值范围
        backup: 是否备份原始结果文件
    """
    # 确保dataset目录存在
    dataset_dir = os.path.join(base_path, 'dataset', 'array_sum')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
    
    # 确保driver目录存在
    driver_dir = os.path.join(base_path, 'driver', 'array_sum')
    if not os.path.exists(driver_dir):
        os.makedirs(driver_dir, exist_ok=True)
    
    # 生成文件名
    if test_id is None:
        test_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = f"array_sum_{test_id}"
    file_path = os.path.join(dataset_dir, f"{filename}.txt")
    result_path = os.path.join(driver_dir, f"result_{test_id}.txt")
    
    # 备份原始结果文件
    if backup and os.path.exists(result_path):
        shutil.copy2(result_path, result_path + '.bak')
        print(f"已备份原始结果文件: {result_path}.bak")
    
    # 根据数据类型生成数组
    if data_type == 'random':
        data = np.random.randint(value_range[0], value_range[1], size=size, dtype=np.int64)
    elif data_type == 'uniform':
        data = np.random.uniform(value_range[0], value_range[1], size=size).astype(np.int64)
    elif data_type == 'normal':
        mean = (value_range[1] + value_range[0]) / 2
        std = (value_range[1] - value_range[0]) / 6  # 99.7%的数据在范围内
        data = np.random.normal(mean, std, size=size).astype(np.int64)
    elif data_type == 'skewed':
        # 创建偏斜分布，大多数值较小，少数值较大
        data = np.random.exponential(1000, size=size).astype(np.int64)
        # 确保在范围内
        data = np.clip(data, value_range[0], value_range[1])
    elif data_type == 'zeros':
        # 大多数元素为0，少数非0
        data = np.zeros(size, dtype=np.int64)
        num_nonzero = size // 100  # 1%的元素非0
        indices = np.random.choice(size, num_nonzero, replace=False)
        data[indices] = np.random.randint(value_range[0], value_range[1], size=num_nonzero)
    elif data_type == 'all_same':
        # 所有元素相同
        value = np.random.randint(value_range[0], value_range[1])
        data = np.full(size, value, dtype=np.int64)
    else:
        raise ValueError(f"不支持的数据类型: {data_type}")
    
    # 计算正确的和（用于验证）
    correct_sum = np.sum(data)
    
    # 写入数据文件
    with open(file_path, 'w') as f:
        for num in data:
            f.write(f"{num}\n")  # 每行一个数字
    
    # 同时创建一个普通data.txt文件用于兼容旧代码
    with open(os.path.join(dataset_dir, "data.txt"), 'w') as f:
        for num in data:
            f.write(f"{num}\n")
    
    # 更新result.txt文件
    with open(result_path, 'w') as f:
        f.write(f"{correct_sum}\n")
    
    # 同时创建一个普通result.txt文件用于兼容旧代码
    with open(os.path.join(driver_dir, "result.txt"), 'w') as f:
        f.write(f"{correct_sum}\n")
    
    # 创建元数据文件（测试数据描述）
    metadata = {
        "test_id": test_id,
        "task": "array_sum",
        "size": size,
        "data_type": data_type,
        "value_range": value_range,
        "expected_sum": int(correct_sum),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(dataset_dir, f"{filename}_meta.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"数组求和测试数据已生成: {file_path}")
    print(f"数组大小: {size}")
    print(f"数据类型: {data_type}")
    print(f"测试ID: {test_id}")
    print(f"正确答案: {correct_sum}")
    
    return test_id, file_path, result_path

def create_matrix_multiply_data(base_path, test_id=None, n=500, density=0.05, matrix_type='random',
                                value_range=(-10, 10), backup=True):
    """
    生成矩阵乘法任务的测试数据
    
    参数:
        base_path: 项目的根路径
        test_id: 测试ID，用于生成唯一文件名
        n: 矩阵维度
        density: 矩阵非零元素的密度
        matrix_type: 矩阵类型 (random, diagonal, block, symmetric, banded)
        value_range: 数值范围
        backup: 是否备份原始结果文件
    """
    # 确保dataset目录存在
    dataset_dir = os.path.join(base_path, 'dataset', 'matrix_multiply')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
    
    # 确保driver目录存在
    driver_dir = os.path.join(base_path, 'driver', 'matrix_multiply')
    if not os.path.exists(driver_dir):
        os.makedirs(driver_dir, exist_ok=True)
    
    # 生成文件名
    if test_id is None:
        test_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = f"matrix_multiply_{test_id}"
    file_path = os.path.join(dataset_dir, f"{filename}.txt")
    result_path = os.path.join(driver_dir, f"result_{test_id}.txt")
    
    # 备份原始结果文件
    if backup and os.path.exists(result_path):
        shutil.copy2(result_path, result_path + '.bak')
        print(f"已备份原始结果文件: {result_path}.bak")
    
    # 根据矩阵类型生成矩阵
    A = np.zeros((n, n), dtype=int)
    
    if matrix_type == 'random':
        # 随机稀疏矩阵
        num_nonzero = int(n * n * density)
        for _ in range(num_nonzero):
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            A[i, j] = random.randint(value_range[0], value_range[1])
    
    elif matrix_type == 'diagonal':
        # 对角线矩阵
        for i in range(n):
            A[i, i] = random.randint(value_range[0], value_range[1])
            # 加一些非对角线元素保持一定稀疏性
            num_off_diag = int(n * density) - 1
            for _ in range(num_off_diag):
                j = random.randint(0, n-1)
                if i != j:  # 避免重复设置对角线元素
                    A[i, j] = random.randint(value_range[0], value_range[1])
    
    elif matrix_type == 'block':
        # 分块矩阵（左上角和右下角有值）
        block_size = n // 2
        # 左上块
        for i in range(block_size):
            for j in range(block_size):
                if random.random() < density * 2:  # 增加密度以确保足够的非零元素
                    A[i, j] = random.randint(value_range[0], value_range[1])
        # 右下块
        for i in range(block_size, n):
            for j in range(block_size, n):
                if random.random() < density * 2:
                    A[i, j] = random.randint(value_range[0], value_range[1])
    
    elif matrix_type == 'symmetric':
        # 对称矩阵
        for i in range(n):
            for j in range(i, n):  # 只填上三角
                if random.random() < density * 2:  # 增加密度
                    value = random.randint(value_range[0], value_range[1])
                    A[i, j] = value
                    A[j, i] = value  # 对称位置
    
    elif matrix_type == 'banded':
        # 带状矩阵
        band_width = max(1, int(n * density))  # 带宽
        for i in range(n):
            for j in range(max(0, i - band_width), min(n, i + band_width + 1)):
                if random.random() < 0.7:  # 在带内有70%概率有值
                    A[i, j] = random.randint(value_range[0], value_range[1])
    
    else:
        raise ValueError(f"不支持的矩阵类型: {matrix_type}")
    
    # 确保矩阵有足够的非零元素
    if np.count_nonzero(A) < 1:
        # 至少添加一个非零元素
        i, j = random.randint(0, n-1), random.randint(0, n-1)
        A[i, j] = random.randint(max(1, value_range[0]), value_range[1])
    
    # 计算结果矩阵 C，其中 C[i][j] = sum(A[i][k] * A[j][k])
    C = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * A[j, k]
    
    # 写入输入数据文件（三元组格式: i j value）
    with open(file_path, 'w') as f:
        f.write(f"{n} {n}\n")  # 矩阵维度
        for i in range(n):
            for j in range(n):
                if A[i, j] != 0:
                    f.write(f"{i} {j} {A[i, j]}\n")
    
    # 同时创建一个普通data.txt文件用于兼容旧代码
    with open(os.path.join(dataset_dir, "data.txt"), 'w') as f:
        f.write(f"{n} {n}\n")  # 矩阵维度
        for i in range(n):
            for j in range(n):
                if A[i, j] != 0:
                    f.write(f"{i} {j} {A[i, j]}\n")
    
    # 写入结果文件（三元组格式: i j value）
    with open(result_path, 'w') as f:
        for i in range(n):
            for j in range(n):
                if C[i, j] != 0:
                    f.write(f"{i} {j} {C[i, j]}\n")
    
    # 同时创建一个普通result.txt文件用于兼容旧代码
    with open(os.path.join(driver_dir, "result.txt"), 'w') as f:
        for i in range(n):
            for j in range(n):
                if C[i, j] != 0:
                    f.write(f"{i} {j} {C[i, j]}\n")
    
    # 创建元数据文件
    metadata = {
        "test_id": test_id,
        "task": "matrix_multiply",
        "size": n,
        "matrix_type": matrix_type,
        "density": density,
        "value_range": value_range,
        "nonzero_count": int(np.count_nonzero(A)),
        "result_nonzero_count": int(np.count_nonzero(C)),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(dataset_dir, f"{filename}_meta.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"矩阵乘法测试数据已生成: {file_path}")
    print(f"矩阵维度: {n}×{n}")
    print(f"矩阵类型: {matrix_type}")
    print(f"非零元素密度: {density}")
    print(f"测试ID: {test_id}")
    print(f"输入矩阵非零元素数量: {np.count_nonzero(A)}")
    print(f"结果矩阵非零元素数量: {np.count_nonzero(C)}")
    
    return test_id, file_path, result_path

def create_graph_bfs_data(base_path, test_id=None, num_vertices=1000, edge_probability=0.01, 
                         graph_type='random', start_vertex=1, backup=True):
    """
    生成图的BFS遍历任务的测试数据
    
    参数:
        base_path: 项目的根路径
        test_id: 测试ID，用于生成唯一文件名
        num_vertices: 图的顶点数量
        edge_probability: 任意两个顶点之间有边的概率
        graph_type: 图的类型 (random, tree, small_world, scale_free, complete)
        start_vertex: BFS起始顶点
        backup: 是否备份原始结果文件
    """
    # 确保dataset目录存在
    dataset_dir = os.path.join(base_path, 'dataset', 'graph_bfs')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
    
    # 确保driver目录存在
    driver_dir = os.path.join(base_path, 'driver', 'graph_bfs')
    if not os.path.exists(driver_dir):
        os.makedirs(driver_dir, exist_ok=True)
    
    # 生成文件名
    if test_id is None:
        test_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = f"graph_bfs_{test_id}"
    file_path = os.path.join(dataset_dir, f"{filename}.txt")
    result_path = os.path.join(driver_dir, f"result_{test_id}.txt")
    
    # 备份原始结果文件
    if backup and os.path.exists(result_path):
        shutil.copy2(result_path, result_path + '.bak')
        print(f"已备份原始结果文件: {result_path}.bak")
    
    # 根据图类型生成图
    if graph_type == 'random':
        # 随机图
        G = nx.erdos_renyi_graph(num_vertices, edge_probability, directed=True)
    
    elif graph_type == 'tree':
        # 树
        G = nx.DiGraph()
        G.add_nodes_from(range(1, num_vertices + 1))
        
        # 为每个节点（除了根节点）添加一条指向父节点的边
        for i in range(2, num_vertices + 1):
            parent = random.randint(1, i-1)
            G.add_edge(parent, i)
        
        # 添加一些随机边使其成为DAG
        extra_edges = int(num_vertices * edge_probability)
        for _ in range(extra_edges):
            u = random.randint(1, num_vertices)
            v = random.randint(1, num_vertices)
            if u != v and not G.has_edge(u, v) and not G.has_edge(v, u):
                G.add_edge(u, v)
    
    elif graph_type == 'small_world':
        # 小世界网络
        # 创建无向环图，然后随机重连一些边
        k = int(num_vertices * edge_probability * 10)  # 平均度数
        k = max(2, min(k, num_vertices - 1))  # 确保合理的k值
        p = 0.1  # 重连概率
        
        # 创建无向小世界图
        G_undirected = nx.watts_strogatz_graph(num_vertices, k, p)
        
        # 转换为有向图
        G = nx.DiGraph()
        # 添加从1开始的节点
        G.add_nodes_from(range(1, num_vertices + 1))
        
        # 为每条边随机选择方向
        for u, v in G_undirected.edges():
            # 确保节点编号在有效范围内
            if u+1 <= num_vertices and v+1 <= num_vertices:
                if random.random() < 0.5:
                    G.add_edge(u+1, v+1)  # +1使节点从1开始
                else:
                    G.add_edge(v+1, u+1)
    
    elif graph_type == 'scale_free':
        # 无标度网络 (Barabási-Albert模型)
        m = max(1, int(num_vertices * edge_probability * 5))  # 每个新节点连接的边数
        
        # 创建无向无标度图
        G_undirected = nx.barabasi_albert_graph(num_vertices, m)
        
        # 转换为有向图
        G = nx.DiGraph()
        G.add_nodes_from(range(num_vertices))
        
        # 为每条边随机选择方向
        for u, v in G_undirected.edges():
            if random.random() < 0.5:
                G.add_edge(u+1, v+1)  # +1使节点从1开始
            else:
                G.add_edge(v+1, u+1)
    
    elif graph_type == 'complete':
        # 完全图
        G = nx.complete_graph(num_vertices, create_using=nx.DiGraph)
        # 重新标记节点，使其从1开始
        G = nx.relabel_nodes(G, lambda x: x+1)
    
    else:
        raise ValueError(f"不支持的图类型: {graph_type}")
    
    # 确保所有顶点的编号是连续的，从1开始
    if graph_type not in ['tree', 'complete']:
        G = nx.convert_node_labels_to_integers(G, first_label=1)
    
    # 确保从起始顶点可以到达其他顶点
    # 先找出所有从起始顶点可到达的顶点
    reachable = set()
    if start_vertex in G:  # 确保起始顶点在图中
        queue = [start_vertex]
        while queue:
            current = queue.pop(0)
            reachable.add(current)
            for neighbor in G.neighbors(current):
                if neighbor not in reachable:
                    queue.append(neighbor)
    
    # 对于不可达的顶点，添加一条边从起始顶点连接到它
    unreachable_count = 0
    for node in range(1, num_vertices + 1):
        if node not in reachable and node in G and start_vertex in G:
            G.add_edge(start_vertex, node)
            unreachable_count += 1
    
    if unreachable_count > 0:
        print(f"添加了 {unreachable_count} 条边以确保图从顶点 {start_vertex} 是连通的")
    
    # 写入边列表到文件
    with open(file_path, 'w') as f:
        for edge in G.edges():
            f.write(f"{edge[0]} {edge[1]}\n")
    
    # 同时创建一个普通data.txt文件用于兼容旧代码
    with open(os.path.join(dataset_dir, "data.txt"), 'w') as f:
        for edge in G.edges():
            f.write(f"{edge[0]} {edge[1]}\n")
    
    # 计算从起始顶点开始的BFS遍历结果
    bfs_result = [-1] * (num_vertices + 1)  # +1是因为顶点从1开始编号
    
    # 执行BFS，但注意处理节点编号
    if start_vertex in G:  # 确保起始顶点在图中
        queue = [start_vertex]
        bfs_result[start_vertex] = 0  # 起始顶点距离为0
        
        visited = set([start_vertex])
        
        while queue:
            current = queue.pop(0)
            for neighbor in G.neighbors(current):
                if neighbor not in visited and 1 <= neighbor <= num_vertices:
                    visited.add(neighbor)
                    bfs_result[neighbor] = bfs_result[current] + 1
                    queue.append(neighbor)
    
    # 写入结果文件
    with open(result_path, 'w') as f:
        for i in range(1, num_vertices + 1):
            f.write(f"{bfs_result[i]}\n")
    
    # 同时创建一个普通result.txt文件用于兼容旧代码
    with open(os.path.join(driver_dir, "result.txt"), 'w') as f:
        for i in range(1, num_vertices + 1):
            f.write(f"{bfs_result[i]}\n")
    
    # 创建元数据文件
    metadata = {
        "test_id": test_id,
        "task": "graph_bfs",
        "vertices": num_vertices,
        "edges": G.number_of_edges(),
        "graph_type": graph_type,
        "edge_probability": edge_probability,
        "start_vertex": start_vertex,
        "max_distance": max(bfs_result[1:]) if max(bfs_result[1:]) >= 0 else -1,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(dataset_dir, f"{filename}_meta.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"图的BFS遍历测试数据已生成: {file_path}")
    print(f"图类型: {graph_type}")
    print(f"顶点数量: {num_vertices}")
    print(f"边数量: {G.number_of_edges()}")
    print(f"测试ID: {test_id}")
    print(f"起始顶点: {start_vertex}")
    print(f"最大BFS距离: {max(bfs_result[1:]) if max(bfs_result[1:]) >= 0 else '无法到达的顶点'}")
    
    return test_id, file_path, result_path

def create_performance_test_suite(base_path, suite_id=None):
    """
    创建不同规模的性能测试套件
    
    参数:
        base_path: 项目的根路径
        suite_id: 测试套件ID
    """
    if suite_id is None:
        suite_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建测试套件元数据
    suite_metadata = {
        "suite_id": suite_id,
        "name": f"Performance Test Suite {suite_id}",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tests": {
            "array_sum": [],
            "matrix_multiply": [],
            "graph_bfs": []
        }
    }
    
    # 数组求和测试套件
    array_sizes = [10000, 100000, 1000000, 10000000]
    array_types = ['random', 'uniform', 'skewed', 'all_same']
    
    print("\n======== 数组求和性能测试套件 ========")
    for i, size in enumerate(array_sizes):
        for j, data_type in enumerate(array_types):
            test_id = f"{suite_id}_array_{i+1}_{j+1}"
            
            try:
                test_id, file_path, result_path = create_array_sum_data(
                    base_path, 
                    test_id=test_id, 
                    size=size, 
                    data_type=data_type,
                    backup=False
                )
                
                # 添加到套件元数据
                suite_metadata["tests"]["array_sum"].append({
                    "test_id": test_id,
                    "size": size,
                    "data_type": data_type,
                    "file_path": file_path,
                    "result_path": result_path
                })
                
                print(f"测试 {i+1}.{j+1}: 数组大小 = {size}, 类型 = {data_type}")
            
            except Exception as e:
                print(f"生成数组求和测试 {test_id} 时出错: {e}")
    
    # 矩阵乘法测试套件
    matrix_sizes = [100, 300, 500, 1000]
    matrix_types = ['random', 'diagonal', 'symmetric', 'banded']
    
    print("\n======== 矩阵乘法性能测试套件 ========")
    for i, n in enumerate(matrix_sizes):
        for j, matrix_type in enumerate(matrix_types):
            test_id = f"{suite_id}_matrix_{i+1}_{j+1}"
            
            # 调整密度以保持非零元素数量合理
            density = min(0.1, 5000 / (n * n))  # 随着矩阵变大，降低密度
            
            try:
                test_id, file_path, result_path = create_matrix_multiply_data(
                    base_path, 
                    test_id=test_id, 
                    n=n, 
                    density=density,
                    matrix_type=matrix_type,
                    backup=False
                )
                
                # 添加到套件元数据
                suite_metadata["tests"]["matrix_multiply"].append({
                    "test_id": test_id,
                    "size": n,
                    "density": density,
                    "matrix_type": matrix_type,
                    "file_path": file_path,
                    "result_path": result_path
                })
                
                print(f"测试 {i+1}.{j+1}: 矩阵大小 = {n}×{n}, 类型 = {matrix_type}, 密度 = {density:.6f}")
            
            except Exception as e:
                print(f"生成矩阵乘法测试 {test_id} 时出错: {e}")
    
    # 图的BFS遍历测试套件
    graph_sizes = [100, 500, 2000, 5000]
    graph_types = ['random', 'tree', 'small_world', 'scale_free']
    
    print("\n======== 图的BFS遍历性能测试套件 ========")
    for i, num_vertices in enumerate(graph_sizes):
        for j, graph_type in enumerate(graph_types):
            test_id = f"{suite_id}_graph_{i+1}_{j+1}"
            
            # 根据节点数和图类型调整边概率
            if graph_type == 'random':
                edge_probability = min(0.1, 10.0 / num_vertices)
            elif graph_type == 'tree':
                edge_probability = 0.05
            elif graph_type == 'small_world':
                edge_probability = min(0.1, 5.0 / num_vertices)
            elif graph_type == 'scale_free':
                edge_probability = min(0.1, 3.0 / num_vertices)
            else:
                edge_probability = 0.01
            
            try:
                test_id, file_path, result_path = create_graph_bfs_data(
                    base_path, 
                    test_id=test_id, 
                    num_vertices=num_vertices, 
                    edge_probability=edge_probability,
                    graph_type=graph_type,
                    backup=False
                )
                
                # 添加到套件元数据
                suite_metadata["tests"]["graph_bfs"].append({
                    "test_id": test_id,
                    "vertices": num_vertices,
                    "edge_probability": edge_probability,
                    "graph_type": graph_type,
                    "file_path": file_path,
                    "result_path": result_path
                })
                
                print(f"测试 {i+1}.{j+1}: 顶点数 = {num_vertices}, 类型 = {graph_type}, 边概率 = {edge_probability:.6f}")
            
            except Exception as e:
                print(f"生成图的BFS遍历测试 {test_id} 时出错: {e}")
    
    # 保存测试套件元数据
    with open(os.path.join(base_path, 'dataset', f"perf_suite_{suite_id}.json"), 'w') as f:
        json.dump(suite_metadata, f, indent=2)
    
    print(f"\n性能测试套件 {suite_id} 已生成完毕！")
    print(f"套件元数据已保存至: {os.path.join(base_path, 'dataset', f'perf_suite_{suite_id}.json')}")
    
    return suite_id, suite_metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成并行测试数据')
    parser.add_argument('--path', type=str, default='E:/third/Partest_on_hard', 
                        help='项目根目录路径')
    parser.add_argument('--task', type=str, choices=['array', 'matrix', 'graph', 'all', 'perf'], 
                        default='all', help='指定要生成的测试任务')
    
    # 数组求和参数
    parser.add_argument('--array_size', type=int, default=100000, 
                        help='数组求和任务的数组大小')
    parser.add_argument('--array_type', type=str, 
                        choices=['random', 'uniform', 'normal', 'skewed', 'zeros', 'all_same'],
                        default='random', help='数组数据类型')
    
    # 矩阵乘法参数
    parser.add_argument('--matrix_size', type=int, default=500, 
                        help='矩阵乘法任务的矩阵维度')
    parser.add_argument('--matrix_density', type=float, default=0.05, 
                        help='矩阵非零元素的密度')
    parser.add_argument('--matrix_type', type=str,
                        choices=['random', 'diagonal', 'block', 'symmetric', 'banded'],
                        default='random', help='矩阵类型')
    
    # 图的BFS遍历参数
    parser.add_argument('--graph_size', type=int, default=1000, 
                        help='图的顶点数量')
    parser.add_argument('--graph_edge_prob', type=float, default=0.01, 
                        help='图中任意两点之间有边的概率')
    parser.add_argument('--graph_type', type=str,
                        choices=['random', 'tree', 'small_world', 'scale_free', 'complete'],
                        default='random', help='图的类型')
    
    # 通用参数
    parser.add_argument('--test_id', type=str, default=None,
                        help='指定测试ID（默认使用时间戳）')
    parser.add_argument('--no_backup', action='store_true', 
                        help='不备份原始结果文件')
    
    args = parser.parse_args()
    
    # 获取项目根目录
    base_path = args.path
    
    # 确保路径使用正斜杠
    base_path = base_path.replace('\\', '/')
    
    # 生成测试数据
    if args.task == 'array' or args.task == 'all':
        create_array_sum_data(
            base_path, 
            test_id=args.test_id, 
            size=args.array_size, 
            data_type=args.array_type,
            backup=not args.no_backup
        )
    
    if args.task == 'matrix' or args.task == 'all':
        create_matrix_multiply_data(
            base_path, 
            test_id=args.test_id,
            n=args.matrix_size, 
            density=args.matrix_density,
            matrix_type=args.matrix_type,
            backup=not args.no_backup
        )
    
    if args.task == 'graph' or args.task == 'all':
        create_graph_bfs_data(
            base_path, 
            test_id=args.test_id,
            num_vertices=args.graph_size, 
            edge_probability=args.graph_edge_prob,
            graph_type=args.graph_type,
            backup=not args.no_backup
        )
    
    if args.task == 'perf':
        create_performance_test_suite(base_path, suite_id=args.test_id)
    
    print("测试数据生成完毕！")