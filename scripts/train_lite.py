import torch
from torchvision import models

from datetime import datetime
import os

from scripts.helpers import make_data_loader, get_processed_data_dir, get_models_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # batch size of 9 because of small dataset
    data_loader = make_data_loader(get_processed_data_dir(), batch_size=9)

    # Instantiate the model architecture
    model = models.resnet18(models.ResNet18_Weights.DEFAULT)

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

    fname = f'resnet{18}_{timestamp}.pth'
    model_path = os.path.join(get_models_dir(), fname)
    torch.save(model.state_dict(), model_path)
    print(f"Model state dictionary saved to {model_path}.")


if __name__ == "__main__":
    main()
