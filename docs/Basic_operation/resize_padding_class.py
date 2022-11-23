from PIL import Image
class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        target_size = [512, 512]
        img1 = sample['image']
        mask1 = sample['label']
        iw, ih = img1.size  # 原始图像的尺寸
        w, h = target_size  # 目标图像的尺寸
        scale = min(w / iw, h / ih)  # 转换的最小比例

        # 保证长或宽，至少一个符合目标图像的尺寸
        nw = int(iw * scale)
        nh = int(ih * scale)

        new_img = img1.resize((nw, nh), Image.BILINEAR)  # 缩小图像
        new_mask = mask1.resize((nw, nh), Image.NEAREST)  # 缩小图像

        img = Image.new('RGB', target_size, (0, 0, 0))  # 生成黑色图像
        mask = Image.new('L', target_size, (0))  # 生成黑色图像

        # // 为整数除法，计算图像的位置
        img.paste(new_img, ((w - nw) // 2, (h - nh) // 2))  # 将图像填充为中间图像，两侧为黑色的样式
        mask.paste(new_mask, ((w - nw) // 2, (h - nh) // 2))  # 将图像填充为中间图像，两侧为黑色的样式
        # return img, mask
        n = 0
        img.save("/home/facility/yiguo.huang/pytorch-deeplab-xception/output_result/{}.jpg".format(n))
        # n += 1
        return {'image': img, 'label': mask}
