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
// ��������
int B01 = 0;  //�������ݵ�gt�ȼ�
int C02 = 1;  //�������ݵ�gt�ȼ�
int B01_count = 0; //Ԥ�����ӵ�����
int C02_count = 0; //Ԥ������ӵ�����
int weight = 400;
int height = 100;
std::vector<int> cls_scores;
// ��ɫģ�� ����
static int classifier_color(const cv::Mat& image, std::vector<int>& cls_scores)
{
	// �̶�����: 1.�������� 2.ʹ��GPU 3.����ģ������ṹ����� 4.Ԥ���� 5.ģ������   // ������಻��Ҫ6.�洢���  �ָ���Ҫ6.softmax
	// ��һ������������
	ncnn::Net net;
	// �ڶ���:ʹ��gpu
	net.opt.use_vulkan_compute = 1;
	// ������(1):����ģ������ṹ
	net.load_param("D:\\nextvpu\\yanye\\yanye_color_ncnn\\model\\model_best_color_sim.param");
	// ������(2):����ģ������ṹ
	net.load_model("D:\\nextvpu\\yanye\\yanye_color_ncnn\\model\\model_best_color_sim.bin");
	// ���Ĳ���Ԥ����(1) ��ʽ��ת�� ��Ϊ������CV��ȡͼ����pytorch��������Image.open��ȡ�ģ�������Ҫ����ת��
	// ��Ҫע���һ����  ����ط���image��CV��ȡ��image������Ҫת����ncnn�ĸ�ʽ�����Ծ����Ӧ����CV(BGR)-RGB-ncnn
	cv::Mat rgbImage;
	cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);
	// ���Ĳ���Ԥ����(2) ת����ncnn�ĸ�ʽ
	ncnn::Mat input = ncnn::Mat::from_pixels_resize(rgbImage.data, ncnn::Mat::PIXEL_RGB, image.cols, image.rows, 100, 400);
	// ���Ĳ���Ԥ����(3) ��ֵ�ͷ�����й�һ��
	const float mean_vals[3] = { 0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f };
	const float norm_vals[3] = { 1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f };
	input.substract_mean_normalize(mean_vals, norm_vals); // ���Ի�ȡ��ncnn��input��
	// ���岽��ģ������ ʵ����Extractor
	ncnn::Extractor ex = net.create_extractor();
	ex.input("input", input);
	ncnn::Mat output;
	ex.extract("output", output);  // �õ�ncnn�����

	// ���������洢��𣬽�output�е�ֵת��Ϊcls_scores���洢���
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

	// TO DO ��һ���Ƕ���ģ�ͻ�õ�cls_scores�����߼��ж�
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
	std::cout << "B01�ĸ����ǣ�" << B01_count << std::endl;
	std::cout << "C02�ĸ����ǣ�" << C02_count << std::endl;
	std::cout << "------------------------" << std::endl;
	return 0;
}


int main()
{
	cv::String path = "D:\\nextvpu\\yanye\\yanye_color_ncnn\\test\\C02\\";  // B01��C02�Ĳ��Լ�·��
	vector<cv::String> files; // ������֤����һ����Ƭ��·��
	cv::glob(path, files, false); // ÿһ��ͼƬ������·��
	// ------->ȫ�߼����ǽ�ͼƬ���������ţ���ɫģ���жϣ�����ϸ��ͼ�񣡣���<----------
	for (std::string file_name : files)
	{
		if (1)
		{
			// ��һ��:����ͼ��
			// std::cout << "files.size():" << files.size() << std::endl;
			std::cout << file_name << std::endl; // ��ӡͼƬ��·�� D:\nextvpu\yanye\yanye_data\20221015\above\SW\test\292_up_B01.bmp
			cv::Mat img = cv::imread(file_name, 1); // ��ȡͼƬ
			if (img.empty()) //�ж�ͼƬ�Ƿ�Ϊ��
			{ 
				continue;
			}

			// �ڶ���:���ų�4000*100
			cv::Mat imgsize_out;
			cv::resize(img, imgsize_out, Size(height, weight), 0, 0, INTER_LINEAR); //imgsize_out.size 100 * 400
			
			// ����������ɫģ���ж�
			classifier_color(imgsize_out, cls_scores);

		}
	}
	return 0;

}