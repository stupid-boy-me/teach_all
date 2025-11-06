import torch
import argparse
import os
import sys
import onnx
import onnxsim

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import onnxmltools
# from onnxmltools.utils.float16_converter import convert_float_to_float16
from RealTime_Segment.models.fdmobilenet import FastSCNN , FDMobileNet



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="/algdata01/yiguo.huang/project_code/NextVpu/UFLDv2/liuxiao/FastSCNNSimpleInstanceSegENetCrossConvTestunsampleAugment_672_384_FDMobileNet/cpt_FDMobilenet_diceloss/FastSCNN_FDMobilenet_backbone_384_672_model.pth",
        help="path to model file",
        type=str,
    )
    parser.add_argument(
        "--accuracy", default="fp32", choices=["fp16", "fp32"], type=str
    )
    parser.add_argument('--size', default=(672, 384), help='size of original frame', type=tuple)
    parser.add_argument(
        "--dataset",
        default="citys",
        help="dataset",
        type=str,
    )
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')

    return parser.parse_args()


def convert(model, args):
    print("start convert...")
    model = model.cpu()
    images = torch.ones((1, 3, args.size[1], args.size[0]))# .cuda()
    onnx_path = args.model_path[:-4] + ".onnx"
    with torch.no_grad():
        torch.onnx.export(
        model,
        images,
        onnx_path,
        # export_params=True,  # 存储训练参数
        opset_version=11,   # ONNX 算子集版本
        # do_constant_folding=True,  # 优化常量
        input_names=['input'],  # 输入名称
        output_names=['instance_pred'],  # 输出名称
        # dynamic_axes={
        #     'input': {0: 'batch_size', 2: 'height', 3: 'width'},  # 动态维度
        #     'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        # }
        )
        print("Export ONNX successful. Model is saved at", onnx_path)
        

        return onnx_path


def optimize_model(model_file: str):
    onnx_model = onnx.load(model_file)
    onnx_model, status = onnxsim.simplify(onnx_model)
    assert status, "failed to simplify"
    onnx.save(onnx_model, model_file.replace(".onnx", "-sim.onnx"))
    print(f"Simplified ONNX model saved to {model_file.replace('.onnx', '-sim.onnx')}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = get_args()

    
    net = FastSCNN(in_channels=3, num_classes=2)
    net = FDMobileNet(2)
    model_weight = args.model_path
    net.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    net.eval()
    output_path = convert(net, args)
    optimize_model(output_path)
