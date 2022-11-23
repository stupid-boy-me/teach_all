'''
题目:最大正方形
难度 中等
在一个由 0 和 1 组成的二维矩阵内，找到只包含 1 的最大正方形，并返回其面积。
我们的要求就是再mask上找到最大的正方形
'''
'''
https://leetcode.cn/problems/maximal-square/solution/by-liupengsay-zoii/
'''
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
class Solution:
    def maximalSquare(self, matrix):
        m, n = len(matrix), len(matrix[0])
        dp = [[0]*(n+1) for _ in range(m+1)]
        ans = 0
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == "1":
                    dp[i+1][j+1] = min(dp[i][j], dp[i+1][j], dp[i][j+1]) + 1
                    if dp[i+1][j+1] > ans:
                        ans = dp[i+1][j+1]
        return ans, ans**2

if __name__ == '__main__':
    '''将图像变成矩阵，
    x = Image.open(image_dir) #打开图片
    data = np.asarray(x)      #转换为矩阵
       将矩阵变成图像
    image = Image.fromarray(data)  #将之前的矩阵转换为图片
       '''
    with open(r"D:\nextvpu\yanye\yanye_xia\train_data_xia_deeplabv3\file\patch.txt","w") as f:
        model = Solution()
        picture_root = r"D:\nextvpu\yanye\yanye_xia\train_data_xia_deeplabv3\data\conv\\"
        filenames = [filename for filename in os.listdir(picture_root)]
        for file in tqdm(filenames):
            picture_path = os.path.join(picture_root,file)
            img = Image.open(picture_path)
            data = np.asarray(img)
            data_str = np.char.mod('%d',data)
            data_str_list = data_str.tolist()
            # print(data_str_list) # <class 'numpy.ndarray'> 2 <class 'list'>
            result = model.maximalSquare(data_str_list)
            print(result,file=f)
