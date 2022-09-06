import argparse
import os

import torch
from torch import nn
from torch.utils.mobile_optimizer import optimize_for_mobile

from spiritvision import get_models_dir, get_data_dir
from ai import resnet_learner, make_data_loader


def main(arguments):
    arch = int(arguments.arch)

    data_loader = make_data_loader(get_data_dir(), batch_size=9)
    learn = resnet_learner(data_loader, arch)

    load_path = os.path.join(get_models_dir(), f"resnet{arch}_model")
    learn.load(load_path)

    quantized = nn.Sequential(learn.model, nn.Softmax()).to('cpu')
    quantized.eval()

    model_int8 = torch.quantization.quantize_dynamic(quantized,
                                                     {nn.Sequential},
                                                     dtype=torch.qint8)

    dummy_input = torch.rand(1, 3, 224, 224)
    traced_script_module = torch.jit.trace(model_int8, dummy_input)

    original_model_name = load_path.split("\\")[-1]
    optimized_model_path = os.path.join(get_models_dir(), f"optimized_torchscript_{original_model_name}.ptl")

    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    traced_script_module_optimized._save_for_lite_interpreter(optimized_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', required=True, help="ResNet architecture. Choose from 18, 34 or 50")

    args = parser.parse_args()
    main(args)
