import torch
from PIL import Image
from ultralytics import YOLO
import os
import torch
from model import ResNetMultiLabelClassifier
from model2 import ResNetMultiLabelClassifier2
import numpy as np


# 调用yolo生成图像txt
def gen_txt(
    yolo_path,
    yolo_path2,
    original_image_path,
    image_path,
    new_image_path,
    label_path,
    width=640,
    height=480,
):  # path要包含文件名
    # Load a model
    surface_model = YOLO(yolo_path)  #  YOLOv8n model
    instance_model = YOLO(yolo_path2)  # pretrained YOLOv8n model

    img = Image.open(original_image_path).convert("RGB")
    if img.width < img.height:
        img = img.transpose(Image.ROTATE_90)
    img = img.resize((width, height))
    img.save(image_path)

    result = surface_model([image_path])[0]
    boxes = result.boxes
    masks = result.masks
    agg_masks = torch.zeros(6, height, width)
    if boxes != None:
        for i in range(len(boxes)):
            cls = int(boxes[i].cls)
            data = masks[i].data[0]
            agg_masks[cls] = torch.max(agg_masks[cls], data.to("cpu"))

    result.save(filename=new_image_path)  # save to disk

    result = instance_model([image_path])[0]
    boxes = result.boxes
    masks = result.masks
    if boxes != None:
        for i in range(len(boxes)):
            cls = int(boxes[i].cls)
            cls = (
                3
                if cls == 1 or cls == 3
                else (4 if cls == 2 or cls == 5 or cls == 7 else 5 if cls == 10 else -1)
            )
            if cls != -1:
                if masks != None:
                    data = masks[i].data[0]
                    agg_masks[cls] = torch.max(agg_masks[cls], data)
    with open(label_path, "w") as f:
        for x in range(width):
            for y in range(height):
                r, g, b = img.getpixel((x, y))
                c1, c2, c3, c4, c5, c6 = agg_masks[:, y, x]
                f.write(
                    str(r)
                    + " "
                    + str(g)
                    + " "
                    + str(b)
                    + " "
                    + str(int(c1) * 255)
                    + " "
                    + str(int(c2) * 255)
                    + " "
                    + str(int(c3) * 255)
                    + " "
                    + str(int(c4) * 255)
                    + " "
                    + str(int(c5) * 255)
                    + " "
                    + str(int(c6) * 255)
                    + "\n"
                )


# 调用yolo生成图像txt
def gen_txt2(
    yolo_path,
    original_image_path,
    image_path,
    new_image_path,
    label_path,
    width=640,
    height=480,
):  # path要包含文件名
    # Load a model
    surface_model = YOLO(yolo_path)  #  YOLOv8n model

    img = Image.open(original_image_path).convert("RGB")
    if img.width < img.height:
        img = img.transpose(Image.ROTATE_90)
    img = img.resize((width, height))
    # img.save(image_path)

    result = surface_model([image_path])[0]
    boxes = result.boxes
    masks = result.masks
    agg_masks = torch.zeros(1, height, width)
    if boxes != None:
        for i in range(len(boxes)):
            cls = int(boxes[i].cls)
            data = masks[i].data[0]
            agg_masks[cls] = torch.max(agg_masks[cls], data.to("cpu"))

    result.save(filename=new_image_path)  # save to disk

    with open(label_path, "w") as f:
        for x in range(width):
            for y in range(height):
                r, g, b = img.getpixel((x, y))
                c1 = agg_masks[:, y, x]
                f.write(
                    str(r) + " " + str(g) + " " + str(b) + " " + str(int(c1)) + "\n"
                )


def gen_result(
    yolo_path,
    yolo_path2,
    image_name,
    original_image_path,
    resnet_path,
    num_classes=7,
    width=640,
    height=480,
):
    image_path = original_image_path + "image_trans/"
    new_image_path = original_image_path + "image_new/"

    if not os.path.exists(image_path):
        os.makedirs(image_path)
        os.makedirs(image_path + "txt/")
    if not os.path.exists(new_image_path):
        os.makedirs(new_image_path)

    original_image_path = original_image_path + image_name
    label_path = image_path + "txt/" + image_name.split(".")[0] + ".txt"
    image_path = image_path + image_name
    new_image_path = new_image_path + image_name
    gen_txt(
        yolo_path,
        yolo_path2,
        original_image_path,
        image_path,
        new_image_path,
        label_path,
        width=640,
        height=480,
    )
    # 创造
    doc_name = label_path
    inputs = np.loadtxt(doc_name)
    inputs = inputs.reshape((width, height, 9)).transpose((2, 0, 1))
    # Load the trained model
    model = ResNetMultiLabelClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(resnet_path, map_location="cpu"))
    model.eval()
    # Specify the device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    inputs = torch.tensor(inputs, dtype=torch.float32)
    inputs = inputs.unsqueeze(0)
    outputs = model(inputs)
    predictions = (outputs > 0.5).float()
    # print(predictions)
    errors_new = [
        "端部盲道缺失",
        "端部盲道错误",
        "交叉处盲道错误",
        "柱状物占用",
        "机动车占用",
        "非机动车占用",
        "窨井盖占用",
    ]
    res = []
    for i, item in enumerate(predictions[0]):
        if item:
            res.append(errors_new[i])
    return res


def gen_result2(
    yolo_path,
    image_name,
    original_image_path,
    resnet_path,
    num_classes=1,
    width=640,
    height=480,
):
    image_path = original_image_path + "image_trans/"
    new_image_path = original_image_path + "image_new2/"

    if not os.path.exists(image_path + "txt2/"):
        os.makedirs(image_path + "txt2/")
    if not os.path.exists(new_image_path):
        os.makedirs(new_image_path)

    original_image_path = original_image_path + image_name
    label_path = image_path + "txt2/" + image_name.split(".")[0] + ".txt"
    image_path = image_path + image_name
    new_image_path = new_image_path + image_name
    gen_txt2(
        yolo_path,
        original_image_path,
        image_path,
        new_image_path,
        label_path,
        width=640,
        height=480,
    )
    # 创造
    doc_name = label_path
    inputs = np.loadtxt(doc_name)
    inputs = inputs.reshape((width, height, 4)).transpose((2, 0, 1))
    # Load the trained model
    model = ResNetMultiLabelClassifier2(num_classes=num_classes)
    model.load_state_dict(torch.load(resnet_path, map_location="cpu"))
    model.eval()
    # Specify the device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    inputs = torch.tensor(inputs, dtype=torch.float32)
    inputs = inputs.unsqueeze(0)
    outputs = model(inputs)
    predictions = (outputs > 0.5).float()
    # print(predictions)
    errors_new = ["缘石坡道问题", "缘石坡道高差过高"]
    res = []
    for i, item in enumerate(predictions[0]):
        if item and i != 0:
            res.append(errors_new[i])
    return res


# yolo_path = "/home/lcj/lab/else/sam_model/others/yolo-seg/src/seg2/runs/segment/train3/weights/best.pt"
# image_name = "0000402.jpg"
# original_image_path = "./data/cocodatas/train/"
# resnet_path = "resnet_ML_model.pth"
# res = gen_result(
#     yolo_path,
#     image_name,
#     original_image_path,
#     resnet_path,
#     num_classes=7,
#     width=640,
#     height=480,
# )

# print(res)
