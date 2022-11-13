#include "D:\nextvpu\ncnn\ncnn_full_source_for_new\ncnn_full-source\build-vs2019\install\include\ncnn\net.h"
#include <iostream>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include<vector>
#include<queue>
#include <map>
#include <direct.h>
#include "opencv2/imgproc/imgproc.hpp" 
#include <stdlib.h>
#include <fstream>
#include <ctime>
#if NCNN_VULKAN
#include "D:\nextvpu\ncnn\ncnn_full_source_for_new\ncnn_full-source\build-vs2019\install\include\ncnn\gpu.h"
#endif // NCNN_VULKAN
#include <time.h>
#define GLFW_INCLUDE_VULKAN

using namespace std;
using namespace cv;
// 参数设置
int B01 = 0;  //测试数据的gt等级
int C02 = 1;  //测试数据的gt等级
int B01_count = 0; //预测青杂的数量
int C02_count = 0; //预测非青杂的数量
int weight = 400;
int height = 100;
std::vector<int> cls_scores;
// 颜色模型 分类
static int classifier_color(const cv::Mat& image, std::vector<int>& cls_scores)
{
	// 固定流程: 1.申明对象 2.使用GPU 3.加载模型网络结构与参数 4.预处理 5.模型推理   // 好像分类不需要6.存储类别  分割需要6.softmax
	// 第一步：申明对象
	ncnn::Net net;
	// 第二步:使用gpu
	net.opt.use_vulkan_compute = 1;
	// 第三步(1):加载模型网络结构
	net.load_param("D:\\nextvpu\\yanye\\yanye_color_ncnn\\model\\model_best_color_sim.param");
	// 第三步(2):加载模型网络结构
	net.load_model("D:\\nextvpu\\yanye\\yanye_color_ncnn\\model\\model_best_color_sim.bin");
	// 第四步：预处理(1) 格式的转换 因为我们是CV读取图像，在pytorch中我们是Image.open读取的，所以需要进行转换
	// 还要注意的一点是  这个地方的image是CV读取的image，还需要转换成ncnn的格式，所以具体的应该是CV(BGR)-RGB-ncnn
	cv::Mat rgbImage;
	cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);
	// 第四步：预处理(2) 转换成ncnn的格式
	ncnn::Mat input = ncnn::Mat::from_pixels_resize(rgbImage.data, ncnn::Mat::PIXEL_RGB, image.cols, image.rows, 100, 400);
	// 第四步：预处理(3) 均值和方差进行归一化
	const float mean_vals[3] = { 0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f };
	const float norm_vals[3] = { 1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f };
	input.substract_mean_normalize(mean_vals, norm_vals); // 可以获取到ncnn的input了
	// 第五步：模型推理 实例化Extractor
	ncnn::Extractor ex = net.create_extractor();
	ex.input("input", input);
	ncnn::Mat output;
	ex.extract("output", output);  // 得到ncnn的输出

	// 第六步：存储类别，将output中的值转化为cls_scores，存储类别
	double max = output[0];
	int Grade = 0;
	for (int j = 0; j < output.w; j++)
	{
		std::cout << j << " " << output[j] << std::endl;
		if (output[j] > max) {
			max = output[j];
			Grade = j;
		}
	}

	// cls_scores.push_back(Grade);

	// TO DO 这一块是对你模型获得的cls_scores进行逻辑判断
	// "0": "B01","1": "C02"


	if (Grade == B01)
	{
		B01_count += 1;
	}
	if (Grade == C02)
	{
		C02_count += 1;

	}

	std::cout << "grade:" << "----" << Grade << std::endl;
	std::cout << "B01的个数是：" << B01_count << std::endl;
	std::cout << "C02的个数是：" << C02_count << std::endl;
	std::cout << "------------------------" << std::endl;
	return 0;
}


int main()
{
	cv::String path = "D:\\nextvpu\\yanye\\yanye_color_ncnn\\test\\C02\\";  // B01和C02的测试集路径
	vector<cv::String> files; // 数据验证集的一张照片的路径
	cv::glob(path, files, false); // 每一张图片的完整路径
	// ------->全逻辑：是将图片遍历，缩放，颜色模型判断，保存合格的图像！！！<----------
	for (std::string file_name : files)
	{
		if (1)
		{
			// 第一步:遍历图像
			// std::cout << "files.size():" << files.size() << std::endl;
			std::cout << file_name << std::endl; // 打印图片的路径 D:\nextvpu\yanye\yanye_data\20221015\above\SW\test\292_up_B01.bmp
			cv::Mat img = cv::imread(file_name, 1); // 读取图片
			if (img.empty()) //判断图片是否为空
			{ 
				continue;
			}

			// 第二步:缩放成4000*100
			cv::Mat imgsize_out;
			cv::resize(img, imgsize_out, Size(height, weight), 0, 0, INTER_LINEAR); //imgsize_out.size 100 * 400
			
			// 第三步：颜色模型判断
			classifier_color(imgsize_out, cls_scores);

		}
	}
	return 0;

}