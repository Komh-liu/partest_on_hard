#include "graph_bfs.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <ctime>
#include <iomanip>

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

// 将 BFS 结果保存到文件中
void saveBfsResultToFile(const std::vector<int>& bfs_result, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }
    for (int vertex : bfs_result) {
        file << vertex << "\n";
    }
    file.close();
}

// 生成带时间戳的文件名
std::string generateTimestampedFilename(const std::string& baseDir, const std::string& baseName) {
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << baseDir << "/" << baseName << "_"
       << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S") << ".txt";
    return ss.str();
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
        std::vector<int> bfs_result(graph.numVertices + 1, -1); // 初始化为 -1，表示未访问
        std::vector<int> result = loadFileToVector(result_file);

        auto time_start = std::chrono::high_resolution_clock::now();
        bfs(graph, bfs_start_vertex, bfs_result);
        auto time_end = std::chrono::high_resolution_clock::now();

        // 生成带时间戳的文件名
        std::string timestamped_result_file = generateTimestampedFilename(
            result_file.substr(0, result_file.find_last_of('/')), // 获取结果文件的目录
            "bfs_result" // 基础文件名
        );

        // 保存 BFS 结果到带时间戳的文件
        saveBfsResultToFile(bfs_result, timestamped_result_file);

        // 清理内存
        delete[] graph.offset;
        delete[] graph.edges;

        std::cout << "Time: " 
                  << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count()
                  << "ms\n";
        
        if (result == bfs_result)
            std::cout << "验证成功" << std::endl;
        else
            std::cout << "验证失败" << std::endl;
        // std::cout << "BFS 结果已保存到文件: " << timestamped_result_file << std::endl;
    }
    return 0;
}