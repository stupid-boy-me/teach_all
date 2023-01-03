import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from utils_grad_cam import GradCAM, show_cam_on_image, center_crop_img
from torchvision.utils import save_image
from model_resnet50 import resnet50
import os
import cv2
# from torchvision.models.feature_extraction import create_feature_extractor


def main():
    model = resnet50(num_classes=2)
    pretrain_weights_path = r"D:\yiguohuang\study\pilipala_w\grad_cam\model-1000_of_1000-0.0004053276206832379-0.9086294416243654.pth"
    model.load_state_dict(torch.load(pretrain_weights_path))

    target_layers = [model.layer4[2]]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dir_path = r"D:\nextvpu\yanye\yanye_data\yanye_data\B01\\"
    filenames = [filename for filename in os.listdir(dir_path)]
    for file in filenames:
        img_path = os.path.join(dir_path,file)
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path).convert('RGB')
        img = np.array(img, dtype=np.uint8)

        # [C, H, W]
        img_tensor = data_transform(img)
        # expand batch dimension
        # [C, H, W] -> [N, C, H, W]
        input_tensor = torch.unsqueeze(img_tensor, dim=0)

        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

        target_category = 0  # B01
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                          grayscale_cam,
                                      use_rgb=True)
        out_picture = os.path.join(r"D:\nextvpu\yanye\yanye_data\yanye_data\B01_grad\\","{}"+".jpg").format(file.split('.')[0])
        im = visualization[:, :, (2, 1, 0)]
        cv2.imwrite(out_picture,im)


if __name__ == '__main__':
    main()
