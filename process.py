def convert_edge_list(input_file, output_file):
    with open(input_file, 'r') as f:
        edges = [line.strip().split() for line in f if line.strip()]

    # 提取所有唯一节点并排序
    nodes = set()
    for u, v in edges:
        nodes.add(int(u))
        nodes.add(int(v))
    sorted_nodes = sorted(nodes)

    # 创建节点到连续序号的映射
    node_to_id = {node: idx for idx, node in enumerate(sorted_nodes)}

    # 转换边列表
    with open(output_file, 'w') as f:
        for u, v in edges:
            new_u = node_to_id[int(u)]
            new_v = node_to_id[int(v)]
            f.write(f"{new_u} {new_v}\n")

# 使用示例
convert_edge_list('input.txt', 'output.txt')