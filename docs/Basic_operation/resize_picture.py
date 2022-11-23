# function: 更改图片尺寸大小

from PIL import Image

'''
filein: 输入图片
fileout: 输出图片
width: 输出图片宽度
height:输出图片高度
type:输出图片类型（png, gif, jpeg...）
'''


def ResizeImage(filein, fileout, width, height, type):
    img = Image.open(filein)
    out = img.resize((width, height), Image.ANTIALIAS)  # resize image with high-quality
    out.save(fileout, type)


if __name__ == "__main__":
    filein = r'D:\nextvpu\yanye\yanye_fcn\data\20220806\upPicture\0_up_level4.bmp'
    fileout = r'D:\nextvpu\yanye\yanye_fcn\data\resize_picture\testout.bmp'
    width = 6000
    height = 6000
    type = 'png'
    ResizeImage(filein, fileout, width, height, type)
