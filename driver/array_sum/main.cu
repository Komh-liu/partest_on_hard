#include "array_sum.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdlib> // 用于 exit()

std::vector<long long> load_array_from_file(const std::string& filename) {
    Array arr;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        exit(1);
    }

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        long long num;
        while (iss >> num) {
            arr.push_back(num);
        }
    }

    file.close();
    return arr;
}

long long load_result_from_file(const std::string& filename) {
    long long result;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        exit(1);
    }

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        iss >> result;
    }

    file.close();
    return result;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "用法: " << argv[0] << " <数组文件路径> <结果文件路径>" << std::endl;
        return 1;
    }

    // 从命令行参数获取文件路径
    std::string data_file_path = argv[1];
    std::string result_file_path = argv[2];

    // 从文件加载数组
    std::vector<long long> arr = load_array_from_file(data_file_path);
    long long result = load_result_from_file(result_file_path);

    // 记录开始时间戳并写入标准输出
    auto time_start = std::chrono::high_resolution_clock::now();
    auto start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        time_start.time_since_epoch()
    ).count();
    std::cout << "[METRICS] BFS_TIME_START=" << start_ms << std::endl << std::flush;

    // 调用CUDA加速的数组求和函数
    long long sum = array_sum(arr);
    std::cout << "数组的和是: " << sum << std::endl;

    // 记录结束时间戳并写入标准输出
    auto time_end = std::chrono::high_resolution_clock::now();
    auto end_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        time_end.time_since_epoch()
    ).count();
    std::cout << "[METRICS] BFS_TIME_END=" << end_ms << std::endl << std::flush;
    
    // 输出耗时和验证结果
    std::cout << "Time: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count()
            << "ms\n";

    if (result == sum)
        std::cout << "验证成功" << std::endl;
    else
        std::cout << "验证失败" << std::endl;

    return 0;
}
