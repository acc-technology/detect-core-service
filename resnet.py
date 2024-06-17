import torch
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet18_Weights

# ResNet 模型的配置
resnet_models = [
    {
        "path": "../model/resnet_ML_model0516-50-2_best.pth",
        "channels": [0, 1, 2, 3, 4, 5, 6, 7, 8],  # 使用的通道
        "type": "端部盲道错误",  # 模型对应的类型
    },
    {
        "path": "../model/resnet_ML_model0516-50-3_best.pth",
        "channels": [0, 1, 2, 3, 4, 5, 6, 7, 8],  # 使用的通道
        "type": "交叉处盲道错误",  # 模型对应的类型
    },
    {
        "path": "../model/resnet_ML_model0516-50-4_best.pth",
        "channels": [0, 1, 2, 3, 4, 5, 6, 7, 8],  # 使用的通道
        "type": "柱状物占用",  # 模型对应的类型
    },
    {
        "path": "../model/resnet_ML_model0516-50-6_best.pth",
        "channels": [0, 1, 2, 3, 4, 5, 6, 7, 8],  # 使用的通道
        "type": "非机动车占用",  # 模型对应的类型
    },
    {
        "path": "../model/resnet_ML_model0516-50-7_best.pth",
        "channels": [0, 1, 2, 3, 4, 5, 6, 7, 8],  # 使用的通道
        "type": "窨井盖占用",  # 模型对应的类型
    },
    {
        "path": "../model/resnet_ML_model0617-50-podao_best.pth",
        "channels": [0, 1, 2, 9],  # 使用的通道
        "type": "缘石坡道高差过高",  # 模型对应的类型
    },
    # 添加更多模型配置
]


# 自定义ResNet模型以支持更多通道输入
class CustomResNet(torch.nn.Module):
    def __init__(self, num_input_channels, num_classes):
        super(CustomResNet, self).__init__()
        resnet = (
            models.resnet50(weights=ResNet50_Weights.DEFAULT)
            if num_input_channels == 9
            else models.resnet18(weights=ResNet18_Weights)
        )
        resnet.conv1 = torch.nn.Conv2d(
            num_input_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        self.features = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.sigmoid(x)


def classify_with_resnet_models(multi_channel_image):
    classifications = []

    for model_cfg in resnet_models:
        channels = model_cfg["channels"]
        resnet_input = multi_channel_image[:, :, channels]
        resnet_input = (
            torch.tensor(resnet_input, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

        model = CustomResNet(num_input_channels=len(channels), num_classes=1)
        model.load_state_dict(torch.load(model_cfg["path"], map_location="cpu"))
        model.eval()

        with torch.no_grad():
            output = model(resnet_input)
            predicted_class = output.item()  # 判断输出是否大于0.5
            classifications.append((model_cfg["type"], predicted_class))

    return classifications
