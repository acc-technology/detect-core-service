import os
from flask import Flask, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from infer_api import gen_result

app = Flask(__name__)
cors = CORS(app)
db = SQLAlchemy(app)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////" + os.path.join(
    app.root_path, "data.db"
)


class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    result = db.Column(db.String(100))


@app.route("/upload", methods=["POST"])
def upload():
    image = request.files["image"]
    image.save("../data/" + image.filename)
    return {
        "code": 200,
        "data": gen_result(
            "./best_yolo_seg0301.pt",
            image.filename,
            "../data/",
            "./resnet_ML_model.pth",
        ),
    }
