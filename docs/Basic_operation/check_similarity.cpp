#include <iostream>
#include <unordered_set>
#include <algorithm>
#include <cmath>

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

// 计算两个字符串的相似性比例
double calculate_similarity(const std::string& str1, const std::string& str2) {
    int char_count1[256] = { 0 }; // 假设字符串由ASCII字符组成
    int char_count2[256] = { 0 };

    // 统计字符出现次数
    for (char c : str1) {
        char_count1[static_cast<unsigned char>(c)]++;
    }

    for (char c : str2) {
        char_count2[static_cast<unsigned char>(c)]++;
    }

    // 计算共同字符数量
    int common_count = 0;
    for (int i = 0; i < 256; ++i) {
        common_count += std::min(char_count1[i], char_count2[i]);
    }

    // 计算相似性比例
    double similarity_ratio = static_cast<double>(common_count) / std::max(str1.length(), str2.length());

    return similarity_ratio;
}

// 检查输入字符串与字符串列表中是否存在相似度大于指定阈值的字符串对
bool check_similarity(const std::string& input_str, const std::vector<std::string>& strings, double threshold) {
    // 遍历字符串列表
    for (const auto& str : strings) {
        // 计算输入字符串与当前列表字符串的相似性
        double similarity = calculate_similarity(input_str, str);
        std::cout << "similarity:" << similarity << std::endl;
        // 如果找到相似度大于阈值的字符串对，返回true
        if (similarity > threshold) {
            return true;
        }
    }

    // 没有找到相似度大于阈值的字符串对，返回false
    return false;
}

int main() {
    // 初始化一个包含多个示例字符串的向量
    std::vector<std::string> strings = {
        "ソフトルンクモルイル",
        "ラクテンカートサービス",
        "ラテフカートサービス2",
        "ENET150290",
        "Aカパジリカ",
        "SMBCレオパレス21",
        "セプンBKCIT0",
        "セプンBK0M40",
        "セブンBKOAFN",
        "セプンBKKA",
        "カメルカリ",
        "外国関係ヒラムクソウ",
        "简易保险21月"
    };

    // 设置相似性阈值
    double threshold = 0.3;

    // 输入的测试字符串
    std::string input_string = "(簡易保険4月)保険12,135111111111111111111111111111111111111111111111"; // 示例输入字符串

    // 检查输入字符串与列表中的字符串是否存在相似度大于阈值的情况
    bool is_similar = check_similarity(input_string, strings, threshold);

    // 输出结果
    if (is_similar) {
        std::cout << "输入字符串与列表中的某个字符串相似度超过 " << threshold << std::endl;
    }
    else {
        std::cout << "输入字符串与列表中的任何字符串相似度不超过 " << threshold << std::endl;
    }

    return 0;
}
