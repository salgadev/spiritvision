"""
See https://github.com/pytorch/serve/issues/667 for ResNet reference
"""
from torchvision.models.resnet import ResNet, Bottleneck


class ImageClassifier(ResNet):
    def __init__(self):
        # Parameters for ResNet 50
        super(ImageClassifier, self).__init__(Bottleneck, [3, 4, 6, 3])
