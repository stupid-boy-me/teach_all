import os
import cv2
from tqdm import tqdm
import sys
#图片框的坐标
def all_dot(w,h,new_w,new_h):
    if w % new_w !=0 or h % new_h != 0:
        #print("原图无法充分裁剪")
        sys.exit()#退出程序
    dot_list=[]#保存裁剪图的左上角坐标
    x = int(w/new_w) #纵切
    y = int(h/new_h) #横切
    [x1,y1]=[0,0]#从原图左上角开始
    for k in range(int(x*y)):
        dot_list.append([x1,y1])
        x1 = x1 + new_w
        if (k+1) % x == 0:
            x1 = 0
            y1 = y1 + new_h
    #print("一共可以裁成%d张宽为%d,长为%d的图"%(len(dot_list),new_w,new_h))
    return dot_list

#画框
def plot_rectangle(image,new_w,new_h):
    # img = cv2.imread(img_dir,cv2.IMREAD_COLOR)#按彩色图读入
    w=image.shape[1]#图片宽
    h=image.shape[0]
    #print("原图宽为%d,高为%d"%(w,h),'\n')
    dots = all_dot(w,h,new_w,new_h)
    for i in range(len(dots)):#可以生成的图片框数量
        #img表示图像，两坐标分别左上(x0,y0)、右下坐标(x1,y1)，(0,0,255)为颜色，2为框粗
        cv2.rectangle(image,(dots[i][0],dots[i][1]),(dots[i][0]+new_w,dots[i][1]+new_h),(0,0,0), 1) # 1代表厚度
    return image

class_dir = r'C:\Users\yiguohuang\Desktop\target_dir\origin_picture\test_all_picture\\'

B03_output_dir = r'C:\Users\yiguohuang\Desktop\target_dir\origin_picture\test_all_picture_wangge\\'

# 输入你想要resize的图像尺寸。
size = 2000

for filename in tqdm(os.listdir(os.path.join(class_dir))):
    img_path = os.path.join(class_dir,filename)
    image = cv2.imread(img_path) # D:\nextvpu\yanye\yanye_data\data_norm\\B01\158_up_B01.bmp
    height, width = image.shape[0],image.shape[1]
    # 等比例缩放尺度。
    scale = height / size
    # 获得相应等比例的图像宽度。
    # width_size = int(width / scale)
    width_size = 500
    new_image = cv2.resize(image, (width_size, size))
    image_new = plot_rectangle(new_image, 50, 80)
    cv2.imwrite(os.path.join(B03_output_dir ,filename),image_new,[int(cv2.IMWRITE_JPEG_QUALITY),100])
