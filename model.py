import torch
import torch.nn as nn
import torchvision.models as models


class ResNetMultiLabelClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetMultiLabelClassifier, self).__init__()
        resnet = models.resnet50(pretrained=False)
        resnet.conv1 = nn.Conv2d(
            9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )  # 18,101
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.sigmoid(x)
