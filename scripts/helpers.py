import itertools
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import torchvision
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from torchvision import transforms

# probably should stay like this
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClassificationInterpretation:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader
        self.class_names = dataloader.dataset.classes
        self.num_classes = len(self.class_names)

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

    def plot_confusion_matrix(self, normalize=False):
        if not hasattr(self, 'confusion_matrix'):
            self.generate_confusion_matrix()

        if normalize:
            self.confusion_matrix = self.confusion_matrix.astype('float') / self.confusion_matrix.sum(axis=1)[:,
                                                                            np.newaxis]

        plt.figure(figsize=(10, 10))
        plt.imshow(self.confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(self.class_names))
        plt.xticks(tick_marks, self.class_names, rotation=45)
        plt.yticks(tick_marks, self.class_names)

        fmt = '.2f' if normalize else 'd'
        thresh = self.confusion_matrix.max() / 2.
        for i, j in itertools.product(range(self.confusion_matrix.shape[0]), range(self.confusion_matrix.shape[1])):
            plt.text(j, i, format(self.confusion_matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if self.confusion_matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()


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


def save_confusion_matrix_plot(confusion_matrix, labels, path):
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, values_format='g')

    plt.title("Confusion Matrix")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.grid(False)

    plt.imshow(confusion_matrix, interpolation='nearest', cmap="Blues")
    plt.savefig(path, dpi=120)
