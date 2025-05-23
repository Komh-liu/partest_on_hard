#include "matrix_multiply.h"
#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <string>
#include <ctime>
#include <filesystem>

// CUDA版本使用一维数组表示矩阵，这里需要加载为一维形式
Matrix load_matrix(const std::string& filename, int& N, int& M) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }

    int rows, cols;
    file >> rows >> cols; // 读取矩阵的行数和列数
    N = rows;
    M = cols;

    // 初始化为全零矩阵（一维表示）
    Matrix matrix(rows * cols, 0);

    // 读取三元组形式的非零元素
    int i, j, value;
    while (file >> i >> j >> value) {
        if (i >= 0 && i < rows && j >= 0 && j < cols) {
            matrix[i * cols + j] = value;
        } else {
            std::cerr << "Invalid matrix coordinates: (" << i << ", " << j << ")" << std::endl;
        }
    }

    return matrix;
}

// 将一维矩阵保存为三元组形式
void save_matrix(const Matrix& matrix, int rows, int cols, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }

    // 保存非零元素为三元组形式
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int index = i * cols + j;
            if (matrix[index] != 0) {
                file << i << " " << j << " " << matrix[index] << std::endl;
            }
        }
    }
}

// 比较两个文本文件是否相同
bool compare_text_files(const std::string& file1, const std::string& file2) {
    // 打开两个文件
    std::ifstream f1(file1);
    std::ifstream f2(file2);

    // 检查文件是否成功打开
    if (!f1.is_open() || !f2.is_open()) {
        std::cerr << "无法打开文件！" << std::endl;
        return false;
    }

    // 逐行读取并比较
    std::string line1, line2;
    bool same = true;

    while (true) {
        std::getline(f1, line1);
        std::getline(f2, line2);

        // 检查是否同时到达文件末尾
        if (f1.eof() && f2.eof()) {
            break;
        }

        // 如果一个文件到达末尾而另一个没有，行数不同
        if (f1.eof() != f2.eof()) {
            std::cerr << "文件行数不同！" << std::endl;
            same = false;
            break;
        }

        // 比较当前行
        if (line1 != line2) {
            std::cerr << "文件内容不同！" << std::endl;
            same = false;
            break;
        }
    }
    return same;
}

// 生成包含时间戳的文件名
std::string generate_filename_with_timestamp(const std::string& base_filename) {
    // 获取当前时间
    std::time_t now = std::time(nullptr);
    std::tm* now_tm = std::localtime(&now);

    // 格式化时间戳
    char timestamp_str[20];
    std::strftime(timestamp_str, sizeof(timestamp_str), "%Y%m%d_%H%M%S", now_tm);

    // 提取文件名（不包含路径）
    std::filesystem::path path(base_filename);
    std::string filename = path.filename().string();

    // 生成新文件名
    std::string new_filename = filename + "_" + timestamp_str + ".txt";

    // 返回新文件的完整路径
    return (path.parent_path() / new_filename).string();
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_matrix_file> <output_result_file>" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];

    // 加载矩阵（一维表示）
    int N, M;  // 矩阵的行数和列数
    Matrix A = load_matrix(input_file, N, M);
    Matrix result(M * M, 0);  // 初始化结果矩阵为全零（一维表示）

    // 记录开始时间戳并写入标准输出
    auto time_start = std::chrono::high_resolution_clock::now();
    auto start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        time_start.time_since_epoch()
    ).count();
    std::cout << "[METRICS] BFS_TIME_START=" << start_ms << std::endl << std::flush;

    // 执行矩阵乘法
    matrix_multiply(A, N, M, result);  // 使用CUDA特定的函数签名

    // 记录结束时间戳并写入标准输出
    auto time_end = std::chrono::high_resolution_clock::now();
    auto end_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        time_end.time_since_epoch()
    ).count();
    std::cout << "[METRICS] BFS_TIME_END=" << end_ms << std::endl << std::flush;

    // 输出耗时
    std::cout << "Time: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count()
            << "ms\n";

    // 生成包含时间戳的文件名
    std::string combined_file = generate_filename_with_timestamp(output_file);
    save_matrix(result, M, M, combined_file);  // 保存结果矩阵

    // 比较结果
    bool c_result = compare_text_files(combined_file, output_file);
    if(c_result)
       std::cout<<"验证成功"<<std::endl;
    else
        std::cout<<"验证失败"<<std::endl;
    
    return 0;
}
