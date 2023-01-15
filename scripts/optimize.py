import argparse
import os

import torch
import torchvision
from torch import nn
from torch.utils.mobile_optimizer import optimize_for_mobile

from helpers import get_models_dir, make_data_loader, get_data_dir, get_resnet_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',
                        required=True,
                        help="ResNet architecture. Choose from 18, 34 or 50")
    parser.add_argument('--platform',
                        required=True,
                        help="Platform to optimize for. Choose 'w' for 'web' or 'm' for 'mobile'")

    args = parser.parse_args()

    arch = int(args.arch)
    platform = args.platform

    data_loader = make_data_loader(get_data_dir(), batch_size=9)
    learn = get_resnet_model(data_loader, arch)

    load_path = os.path.join(get_models_dir(), f"resnet{arch}_model.pth")

    learn.load(load_path)
    original_model_name = load_path.split("\\")[-1]

    # Load the model's parameters from the saved file
    params = torch.load(load_path)

    # Instantiate the model architecture
    model = torchvision.models.resnet18()

    # Load the parameters into the model
    model.load_state_dict(params)

    # Move the model to the CPU
    model.to("cpu")

    if platform.lower() == 'm':

        # quantized = nn.Sequential(learn.model, nn.Softmax()).to('cpu')
        quantized = nn.Sequential(model, nn.Softmax()).to('cpu')
        quantized.eval()

        model_int8 = torch.quantization.quantize_dynamic(quantized,
                                                         {nn.Sequential},
                                                         dtype=torch.qint8)

        dummy_input = torch.rand(1, 3, 224, 224)
        traced_script_module = torch.jit.trace(model_int8, dummy_input)

        optimized_model_path = os.path.join(get_models_dir(), f"{original_model_name}_mobile_optimized.ptl")

        traced_script_module_optimized = optimize_for_mobile(traced_script_module)
        traced_script_module_optimized._save_for_lite_interpreter(optimized_model_path)


    elif platform.lower() == 'w':
        # Set model to run on CPU
        model.to("cpu")

        # Optimize model for inference
        # This will remove unnecessary operations and reduce the model size
        # It's not necessary if you are not going to use your model on the web
        model.eval()

        # pip install torch-qnnpack
        # THIS WILL NOT WORK IN WINDOWS
        torch.backends.quantized.engine = 'qnnpack'
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

        optimized_model_path = os.path.join(get_models_dir(), f"{original_model_name}_web_optimized.pth")

        # Save the optimized model
        torch.save(model.state_dict(), optimized_model_path)

    else:
        raise RuntimeError(f'No platform selected. Choose either "w" or "m".')


if __name__ == "__main__":
    main()
