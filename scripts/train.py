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

    # TRAINING LOOP
    if arch == 34:
        model = models.resnet34(pretrained=True)
    elif arch == 50:
        model = models.resnet50(pretrained=True)
    else:
        model = models.resnet18(pretrained=True)

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # model.fine_tune(10)

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

    # previous way
    #  save_path = os.path.join(get_models_dir(), f"resnet{arch}_model")
    # model.save(save_path)

    # TODO: plot with my "custom class"
    """
    interp = ClassificationInterpretation(model=,
                                          test_data_loader=data_loader,
                                          class_names=data_loader.vocab)
    interp.plot_confusion_matrix()

    # interp = ClassificationInterpretation.from_learner(model)
    # cm = interp.confusion_matrix()
    # cm_png_path = os.path.join(get_root_dir(), f"confusion_matrix_ResNet{arch}.png")

    # save_confusion_matrix_plot(confusion_matrix=cm, labels=data_loader.vocab, path=cm_png_path)
    """


if __name__ == "__main__":
    main()
