import numpy as np
from ultralytics import YOLO
import os

# YOLOv8 模型的配置
yolo_models = [
    {
        "path": "../model/best_yolo_seg0304.pt",
        "class_to_channel": {0: 3, 1: 4, 2: 5},
        "type": "ground",
    },
    {
        "path": "../model/yolov8x-seg.pt",
        "class_to_channel": {1: 6, 3: 6, 2: 7, 5: 7, 7: 7, 10: 8},
        "type": "general",
    },
    {
        "path": "../model/best_podao_0617.pt",
        "class_to_channel": {0: 9},
        "type": "ramp",
    },
    # 添加更多模型配置
]


def apply_yolo_models(image, filename):
    image_array = np.array(image)
    h, w, _ = image_array.shape
    multi_channel_image = np.zeros((h, w, 10), dtype=np.uint8)  # 最多有9个通道

    # 将原始RGB图像放到前3个通道
    multi_channel_image[:, :, :3] = image_array

    for model_cfg in yolo_models:
        model = YOLO(model_cfg["path"])
        results = model(image_array)

        for result in results:
            result.save(filename=os.path.join("segments", model_cfg["type"], filename))
            if result.masks is not None:  # 检查 result.masks 是否为 None
                masks = result.masks.data.numpy()
                for box, mask in zip(result.boxes, masks):
                    cls_index = int(box.cls)  # 确保 cls_index 是整数类型
                    channel_index = model_cfg["class_to_channel"].get(cls_index)
                    if channel_index is not None:
                        # 对同一通道的多个掩码取并集
                        multi_channel_image[:, :, channel_index] = np.maximum(
                            multi_channel_image[:, :, channel_index], mask * 255
                        )

    return multi_channel_image
