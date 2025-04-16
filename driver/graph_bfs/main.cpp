#include "graph_bfs.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <numeric>

std::vector<int> loadFileToVector(const std::string& filename) {
    std::vector<int> result;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return result;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int num;
        while (iss >> num) {
            result.push_back(num);
        }
    }
    file.close();
    return result;
}

Graph loadGraphFromFile(const std::string& filename) {
    std::ifstream file(filename);
    Graph graph;

    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return {0, 0, nullptr, nullptr};
    }

    // 第一次遍历文件，计算顶点数和边数
    int maxVertex = 0;
    int numEdges = 0;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int u, v;
        if (iss >> u >> v) {
            maxVertex = std::max({maxVertex, u, v});
            numEdges++;
        }
    }
    file.close();

    // 计算顶点数
    graph.numVertices = maxVertex + 1;
    graph.numEdges = numEdges;

    // 分配内存
    graph.offset = new int[graph.numVertices + 1]();
    graph.edges = new int[graph.numEdges];

    // 第二次遍历文件，计算偏移数组
    file.open(filename);
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int u, v;
        if (iss >> u >> v) {
            graph.offset[u + 1]++;
        }
    }
    file.close();

    // 计算前缀和
    std::partial_sum(graph.offset, graph.offset + graph.numVertices + 1, graph.offset);

    // 第三次遍历文件，填充边数组
    std::vector<int> edgeIndex(graph.numVertices + 1, 0);
    file.open(filename);
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int u, v;
        if (iss >> u >> v) {
            int idx = graph.offset[u] + edgeIndex[u];
            graph.edges[idx] = v;
            edgeIndex[u]++;
        }
    }
    file.close();

    return graph;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <result_file>" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    std::string result_file = argv[2];
    Graph graph = loadGraphFromFile(input_file);

    if (graph.numVertices > 0) {
        int bfs_start_vertex = 1;
        std::cout << "BFS starting from vertex " << bfs_start_vertex << ":\n";
        std::vector<int> bfs_result;
        std::vector<int> result = loadFileToVector(result_file);

        auto time_start = std::chrono::high_resolution_clock::now();
        bfs(graph, bfs_start_vertex, bfs_result);
        auto time_end = std::chrono::high_resolution_clock::now();

        // 清理内存
        delete[] graph.offset;
        delete[] graph.edges;

        std::cout << "Time: " 
                  << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count()
                  << "ms\n";
        if (result == bfs_result)
            std::cout << "验证成功";
    }
    return 0;
}    