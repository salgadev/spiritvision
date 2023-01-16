from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import torchvision

from torchvision import transforms

# probably should stay like this
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_root_dir():
    """
    The 'scripts' folder is hardcoded since the root is expected to be its parent
    """
    current_dir = Path(__file__)
    return [p for p in current_dir.parents if p.parts[-1] == 'scripts'][0].parent


def get_data_dir():
    return os.path.join(get_root_dir(), "data")


def get_processed_data_dir():
    return os.path.join(get_root_dir(), "data", "processed")


def get_models_dir():
    return os.path.join(get_root_dir(), "models")


def get_scripts_dir():
    return os.path.join(get_root_dir(), "scripts")


def save_to_model_folder(model, title):
    now = datetime.now()
    timestamp = str(now.strftime("%Y%m%d_%H-%M-%S"))

    fname = f'{title}-{timestamp}.pth'
    torch.save(model.state_dict(), os.path.join(get_models_dir(), fname))
    return fname


def make_data_loader(data_path, batch_size):
    # the next two lines represent imagenette_stats
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(460),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    dataset = torchvision.datasets.ImageFolder(data_path, data_transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader


class ClassificationInterpretation:
    def __init__(self, model, dataloader):
        self.confusion_matrix = None
        self.model = model
        self.dataloader = dataloader
        self.class_names = dataloader.dataset.classes
        self.num_classes = len(self.class_names)
        self.generate_confusion_matrix()

    def generate_confusion_matrix(self):
        self.model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                y_true.append(labels.cpu().numpy())
                y_pred.append(preds.cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        self.confusion_matrix = confusion_matrix(y_true, y_pred)

    def save_confusion_matrix_plot(self, path):
        # TODO: Use seaborn or plotly
        cm = self.confusion_matrix
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.class_names)
        disp.plot(cmap=plt.cm.Blues, values_format='g')

        plt.title("Confusion Matrix")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.grid(False)

        plt.imshow(cm, interpolation='nearest', cmap="Blues")
        saving_path = path.replace('.pth', '.PNG')
        plt.savefig(saving_path, dpi=120)
        print(f'Saved {saving_path} successfully!')
