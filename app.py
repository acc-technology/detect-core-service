import os
from flask import Flask, request,jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from infer_api import gen_result
from get_gisinfo import get_location_from_image
import time

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////home/lcj/lab/else/detect-core-service/database.db'
db = SQLAlchemy(app)
# 创建数据库表格
class FileInfo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.Text, nullable=False)
    upload_time = db.Column(db.Integer, nullable=False)
    longitude = db.Column(db.Float, nullable=True)
    latitude = db.Column(db.Float, nullable=True)
    error_types = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f"<FileInfo id={self.id}, filename={self.filename}, upload_time={self.upload_time}, " \
               f"longitude={self.longitude}, latitude={self.latitude}, error_types={self.error_types}>"

# db.create_all()
with app.app_context():
    db.create_all()
cors = CORS(app)


@app.route("/image/upload", methods=["POST"])
def image_upload():
    image = request.files["image"]
    timestamp = int(time.time())
    filename_new = f"{timestamp}_{image.filename}"
    image.save("data/" + filename_new)
    # print(image)
    longitude,latitude=get_location_from_image("data/" + filename_new)
    # exit()
    # longitude = 0.0
    # latitude = 0.0
    # error_types_list=gen_result(
    #         "./best_yolo_seg0301.pt",
    #         image.filename,
    #         "../data/",
    #         "./resnet_ML_model.pth",
    #     )
    error_types_list=["errors1","errors2"]
    error_types=','.join(error_types_list)

    # # 将图片信息存入数据库
    new_file_info = FileInfo(filename=filename_new, upload_time=timestamp,
                                 longitude=longitude, latitude=latitude, error_types=error_types)
    db.session.add(new_file_info)
    db.session.commit()

    return {
        "code": 200,
        "data":"1"
    }
    # return {
    #     "code": 200,
    #     "data": gen_result(
    #         "./best_yolo_seg0301.pt",
    #         image.filename,
    #         "../data/",
    #         "./resnet_ML_model.pth",
    #     ),
    # }
@app.route("/image/list", methods=["GET"])
def list_images():
    images = FileInfo.query.all()
    image_list = []
    for image in images:
        image_info = {
            'id': image.id,
            'filename': image.filename,
            'upload_time': image.upload_time,
            'longitude': image.longitude,
            'latitude': image.latitude,
            'error_types': image.error_types.split(',')  # 将错误类型分割成列表
        }
        image_list.append(image_info)
    return jsonify(image_list)

@app.route("/image/get", methods=["GET"])
def get_image():
    image_id = request.args.get('id')
    image = FileInfo.query.filter_by(id=image_id).first()
    if image:
        image_info = {
            'id': image.id,
            'filename': image.filename,
            'upload_time': image.upload_time,
            'longitude': image.longitude,
            'latitude': image.latitude,
            'error_types': image.error_types.split(',')  # 将错误类型分割成列表
        }
        return jsonify(image_info)
    else:
        return jsonify({'error': 'Image not found'}), 404
    
@app.route("/test", methods=["POST"])
def test():
    return {
        "code": 200,
        "data": "1",
    }
