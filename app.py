from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from datetime import datetime
from werkzeug.utils import secure_filename
from PIL import Image as PILImage
import os
from config import Config
from models import db, Image
from utils import allowed_file, get_exif_data, get_lat_lon, process_image
from yolo import apply_yolo_models
from resnet import classify_with_resnet_models

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)
CORS(app)


# 辅助函数：生成统一的响应格式
def make_response(code, msg, data=None):
    return jsonify({"code": code, "msg": msg, "data": data}), 200


@app.route("/image/upload", methods=["POST"])
def upload_file():
    # 检查请求中是否有文件
    if "image" not in request.files:
        return make_response(400, "No file part")

    file = request.files["image"]

    # 如果用户没有选择文件，浏览器提交一个空的文件名
    if file.filename == "":
        return make_response(400, "No selected file")

    # 检查文件是否符合允许的扩展名
    if file and allowed_file(file.filename, app.config["ALLOWED_EXTENSIONS"]):
        # 添加时间戳到文件名
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = secure_filename(file.filename)
        filename = (
            f"{filename.rsplit('.', 1)[0]}_{timestamp}.{filename.rsplit('.', 1)[1]}"
        )
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # 读取EXIF数据以获取经纬度信息
        image = PILImage.open(file_path)
        exif_data = get_exif_data(image)
        latitude, longitude = get_lat_lon(exif_data)

        # 处理图像：转换为RGB、调整大小、旋转（如果需要）
        image = process_image(image)
        image.save(file_path)

        # 使用YOLOv8模型进行图像分割
        multi_channel_image = apply_yolo_models(image, filename)

        # 使用ResNet模型进行分类
        classifications = classify_with_resnet_models(multi_channel_image)

        # 创建新的Image对象并存储到数据库
        new_image = Image(
            filename=filename,
            latitude=latitude,
            longitude=longitude,
            classifications=classifications,  # 将分类结果存储为字符串
        )
        db.session.add(new_image)
        db.session.commit()

        data = {
            "id": new_image.id,  # 返回新创建的图片ID
            "filename": filename,
            "upload_time": int(new_image.upload_time.timestamp()),
            "latitude": latitude,
            "longitude": longitude,
            "classifications": classifications,
        }
        return make_response(200, "File uploaded successfully", data)
    else:
        return make_response(400, "File type not allowed")


# 获取图片列表接口
@app.route("/image/list", methods=["GET"])
def get_images():
    images = Image.query.order_by(Image.id.desc()).all()
    image_list = [
        {
            "id": image.id,
            "filename": image.filename,
            "upload_time": int(image.upload_time.timestamp()),
            "latitude": image.latitude,
            "longitude": image.longitude,
            "classifications": image.classifications,
        }
        for image in images
    ]
    return make_response(200, "Images retrieved successfully", image_list)


# 获取单张图片信息接口
@app.route("/image/get", methods=["GET"])
def get_image_info():
    image_id = request.args.get("id")
    image = Image.query.get(image_id)
    if image:
        image_info = {
            "id": image.id,
            "filename": image.filename,
            "upload_time": int(image.upload_time.timestamp()),
            "latitude": image.latitude,
            "longitude": image.longitude,
            "classifications": image.classifications,
        }
        return make_response(200, "Image info retrieved successfully", image_info)
    else:
        return make_response(404, "Image not found")


# 获取图像文件接口
@app.route("/image/fetch/<filename>", methods=["GET"])
def get_image(filename):
    type = request.args.get("type")
    directory = os.path.join("segments", type) if type else app.config["UPLOAD_FOLDER"]
    return send_from_directory(directory, filename)


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
