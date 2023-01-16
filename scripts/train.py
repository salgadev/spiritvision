import argparse

from torchvision import models


from helpers import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', required=True, help="ResNet architecture. Choose from 18, 34 or 50")

    args = parser.parse_args()

    arch = int(args.arch)

    # batch size of 9 because of small dataset
    data_loader = make_data_loader(get_processed_data_dir(), batch_size=9)

    # Instantiate the model architecture
    if arch == 18:
        model = models.resnet18(models.ResNet18_Weights.DEFAULT)
    elif arch == 34:
        model = models.resnet34(models.ResNet34_Weights.DEFAULT)
    elif arch == 50:
        model = models.resnet50(models.ResNet50_Weights.DEFAULT)
    else:
        raise RuntimeError(f'Choose a valid ResNet architecture.')

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    now = datetime.now()
    timestamp = str(now.strftime("%Y%m%d_%H-%M-%S"))

    fname = f'resnet{arch}_{timestamp}.pth'
    model_path = os.path.join(get_models_dir(), fname)
    torch.save(model.state_dict(), model_path)
    print(f"Model state dictionary saved to {model_path}.")

    interp = ClassificationInterpretation(model, data_loader)
    interp.save_confusion_matrix_plot(model_path)


if __name__ == "__main__":
    main()
