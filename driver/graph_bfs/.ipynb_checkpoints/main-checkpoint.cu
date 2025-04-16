#include "graph_bfs.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>

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

CUDAGraph loadCUDAGraphFromFile(const std::string& filename) {
    std::ifstream file(filename);
    CUDAGraph graph;

    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return {0, 0, nullptr, nullptr};
    }

    std::vector<int> tempEdges;
    std::vector<int> tempOffset;
    tempOffset.push_back(0);
    int maxVertex = 0;

    // First pass: Calculate maxVertex and the number of neighbors for each vertex
    std::string line;
    while (std::getline(file, line)) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        int u, v;
        if (iss >> u >> v) {
            maxVertex = std::max({maxVertex, u, v});
            if (tempOffset.size() <= u + 1) tempOffset.resize(u + 2, 0);
            if (tempOffset.size() <= v + 1) tempOffset.resize(v + 2, 0);
            tempOffset[u + 1]++;
            tempOffset[v + 1]++; // Assuming undirected graph
        }
    }
    file.close();

    graph.numVertices = maxVertex + 1;
    graph.offset = new int[graph.numVertices + 1];
    std::partial_sum(tempOffset.begin(), tempOffset.end(), graph.offset);
    graph.numEdges = graph.offset[graph.numVertices];
    graph.edges = new int[graph.numEdges];

    std::vector<int> currentEdgeIndex(graph.numVertices, 0);

    // Second pass: Fill the edges array
    file.open(filename);
    if (!file.is_open()) {
        std::cerr << "无法重新打开文件: " << filename << std::endl;
        delete[] graph.offset;
        delete[] graph.edges;
        return {0, 0, nullptr, nullptr};
    }

    while (std::getline(file, line)) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        int u, v;
        if (iss >> u >> v) {
            int indexU = graph.offset[u] + currentEdgeIndex[u];
            if (indexU < graph.numEdges) {
                graph.edges[indexU] = v;
                currentEdgeIndex[u]++;
            }

            int indexV = graph.offset[v] + currentEdgeIndex[v];
            if (indexV < graph.numEdges) {
                graph.edges[indexV] = u;
                currentEdgeIndex[v]++;
            }
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
    CUDAGraph graph = loadCUDAGraphFromFile(input_file);

    if (graph.numVertices > 0) {
        int bfs_start_vertex = 0; // 修改变量名以避免冲突
        std::cout << "BFS starting from vertex " << bfs_start_vertex << ":\n";
        std::vector<int> bfs_result;
        std::vector<int> result = loadFileToVector(result_file);
        // 时间测量部分
        auto time_start = std::chrono::high_resolution_clock::now(); // 修改变量名
        bfs(graph, bfs_start_vertex,bfs_result);
        auto time_end = std::chrono::high_resolution_clock::now(); // 修改变量名

        // 清理内存
        delete[] graph.offset;
        delete[] graph.edges;

        // 输出运行时间
        std::cout << "Time: " 
                  << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count()
                  << "ms\n";
        if(result == bfs_result)
            std::cout << "验证成功";
    }
    return 0;
}