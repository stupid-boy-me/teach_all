from PIL import Image
import os
import random
from canny_dilate_GaussianBlur import canny_dilate_GaussianBlur

def handle_img(imgdir, imgFlodName, img_path):
    imgs = os.listdir(imgdir + imgFlodName)
    imgNum = len(imgs)
    print(imgNum)
    image_ori = os.listdir(img_path)
    image_Num = len(image_ori)
    print(image_Num)
    num = 1
    for i in range(imgNum):
        img1 = Image.open(imgdir + imgFlodName + "/" + imgs[i])
        print("img1",img1.size)

        for j in range(image_Num):
            oriImg = Image.open(img_path + "/" + image_ori[j])
            image = oriImg.size
            w_ratio = oriImg.size[0] / img1.size[0]
            h_ratio = oriImg.size[1] / img1.size[1]
            img = img1.resize((int(img1.size[0] * w_ratio) , int(img1.size[1] * h_ratio)))
            print("image",image)
            # oriImg.paste(img, (image[0]-102, image[1]-102))

            if image[0] < image[1]:
                oriImg.paste(img, (random.randint(0, image[0] -200), random.randint(0, image[0] -200)))
            else:
                oriImg.paste(img, (random.randint(0, image[1] -200), random.randint(0, image[1]-200 )))
            # oriImg.show()
            oriImg1 = oriImg.convert('RGB')
            oriImg1.save(r"D:\nextvpu\OCR\书本框检测-背景优化\data\OCR_picture2picture\picture_book_background" + "/" + str(num) + ".jpg")  # 保存合并的地址
            num += 1
if __name__ == '__main__':
    imgdir = r"D:\nextvpu\OCR\书本框检测-背景优化\data\OCR_picture2picture\\"  # 书的大地址
    imgFlodName = "picture_book" # 书的小地址
    image_path = r"D:\nextvpu\OCR\书本框检测-背景优化\data\OCR_picture2picture\picture_background"  # 背景的地址
    handle_img(imgdir, imgFlodName, image_path)
