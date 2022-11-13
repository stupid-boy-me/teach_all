#include<stdio.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include  <opencv2\opencv.hpp>
#include "net.h"
#include "mat.h"
#include "benchmark.h"
#define GLFW_INCLUDE_VULKAN
#if NCNN_VULKAN
#include "D:\nextvpu\ncnn\ncnn_full_source_for_new\ncnn_full-source\build-vs2019\install\include\ncnn\gpu.h"
#endif // NCNN_VULKAN
#include "yanye_color_all_score_ncnn.id.h"
/*
	@brief 读取标签文件
	@param [input] strFileName 文件名
	@param [input] vecLabels 标签
*/
void read_labels(std::string strFileName, std::vector<std::string>& vecLabels)
{
	std::ifstream in(strFileName);

	if (in)
	{
		std::string line;
		while (std::getline(in, line))
		{
			// std::cout << line << std::endl;
			vecLabels.push_back(line);
		}
	}
	else
	{
		std::cout << "label file is not exit!!!" << std::endl;
	}
}
/*
	@brief squeezenet_v_1			预测单张图的类别
	@param [input] strImagePath		图片路径
*/
void forward_squeezenet_v_1(std::string strImagePath)
{
	// data
	std::string strLabelPath = "D:\\nextvpu\\yanye\\test_ten_animals_ncnn\\model\\label.txt";
	std::vector<std::string> vecLabel;
	read_labels(strLabelPath, vecLabel);
	cv::Mat rgbImage_color;
	// const float mean_vals[3] = { 104.f, 117.f, 123.f };
	const float mean_vals[3] = { 0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f };
	const float norm_vals[3] = { 1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f };
	cv::Mat matImage = cv::imread(strImagePath);
	cv::cvtColor(matImage, rgbImage_color, cv::COLOR_BGR2RGB);

	if (matImage.empty())
	{
		printf("image is empty!!!\n");
	}

	const int nImageWidth = matImage.cols;
	const int nImageHeight = matImage.rows;

	// input and output
	ncnn::Mat matIn;
	ncnn::Mat matOut;
	// net
	ncnn::Net net;
	net.load_param_bin("D:\\nextvpu\\yanye\\yanye_color_all_score_ncnn\\model\\weights_color_B01_C02_sim.param.bin");
	net.load_model("D:\\nextvpu\\yanye\\yanye_color_all_score_ncnn\\model\\weights_color_B01_C02_sim.bin");

	const int nNetInputWidth = 100;
	const int nNetInputHeight = 400;

	// time
	double dStart = ncnn::get_current_time();

	matIn = ncnn::Mat::from_pixels_resize(rgbImage_color.data, ncnn::Mat::PIXEL_RGB, nImageWidth, nImageHeight, nNetInputWidth, nNetInputHeight);
	// 数据预处理
	matIn.substract_mean_normalize(mean_vals, norm_vals);

	// forward
	// net.opt.use_vulkan_compute = 1;
	ncnn::Extractor ex = net.create_extractor();
	ex.set_light_mode(true);
	ex.input(weights_color_B01_C02_sim_param_id::BLOB_input, matIn);
	ex.extract(weights_color_B01_C02_sim_param_id::BLOB_output, matOut);

	printf("output_size: %d, %d, %d \n", matOut.w, matOut.h, matOut.c);
	// 添加softmax
	{
		ncnn::Layer* softmax = ncnn::create_layer("Softmax");

		ncnn::ParamDict pd;
		softmax->load_param(pd);

		softmax->forward_inplace(matOut, net.opt);

		delete softmax;
	}

	matOut = matOut.reshape(matOut.w * matOut.h * matOut.c);

	// cls 1000 class
	std::vector<float> cls_scores;
	cls_scores.resize(matOut.w);
	for (int i = 0; i < matOut.w; i++)
	{
		cls_scores[i] = matOut[i];
	}
	// return top class
	int top_class = 0;
	float max_score = 0.f;
	for (size_t i = 0; i < cls_scores.size(); i++)
	{
		float s = cls_scores[i];
		if (s > max_score)
		{
			top_class = i;
			max_score = s;
		}
	}
	double dEnd = ncnn::get_current_time();

	printf("%d  score: %f   spend time: %.2f ms\n", top_class, max_score, (dEnd - dStart));
	std::cout << vecLabel[top_class] << std::endl;
	cv::putText(matImage, vecLabel[top_class], cv::Point(5, 10), 1, 0.8, cv::Scalar(0, 0, 255), 1);
	cv::putText(matImage, " score:" + std::to_string(max_score), cv::Point(5, 20), 1, 0.8, cv::Scalar(0, 0, 255), 1);
	cv::putText(matImage, " time: " + std::to_string(dEnd - dStart) + "ms", cv::Point(5, 30), 1, 0.8, cv::Scalar(0, 0, 255), 1);
	//cv::imwrite("D:\\nextvpu\\yanye\\squeezenet_test\\ncnn_squeezenet_v_1_win-master\\ncnn_squeezenet_v_1_win-master\\images\\743_up_B01_output.bmp", matImage);
	//cv::imshow("result", matImage);
	//cv::waitKey(-1);


}

int main()
{
	//获取文件
	std::string path = "D:\\nextvpu\\yanye\\test_ten_animals_ncnn\\test_images\\dog\\";
	std::vector<cv::String> files;
	cv::glob(path, files, false);

	//遍历文件
	int count = -1;
	printf("hello ncnn");
	for (std::string file_name : files) {
		std::cout << "file_name" << file_name << std::endl;
		forward_squeezenet_v_1(file_name);
		std::cout << "=============================" << std::endl;

		//system("pause");
	}


}