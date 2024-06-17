from PIL import ExifTags


def allowed_file(filename, allowed_extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def get_exif_data(image):
    exif_data = {}
    try:
        info = image._getexif()
        if info:
            for tag, value in info.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                exif_data[decoded] = value
    except AttributeError:
        pass
    return exif_data


def get_lat_lon(exif_data):
    lat = None
    lon = None

    if "GPSInfo" in exif_data:
        gps_info = exif_data["GPSInfo"]

        gps_lat = gps_info.get(2)
        gps_lon = gps_info.get(4)

        if gps_lat and gps_lon:
            lat = convert_to_degrees(gps_lat)
            lon = convert_to_degrees(gps_lon)

            if gps_info.get(1) == "S":
                lat = -lat
            if gps_info.get(3) == "W":
                lon = -lon

    return lat, lon


def convert_to_degrees(value):
    d, m, s = value
    return d + (m / 60.0) + (s / 3600.0)


def process_image(image):
    # 检查是否需要旋转
    if image.width < image.height:
        image = image.rotate(90, expand=True)

    # 转换为RGB格式
    image = image.convert("RGB")

    # 调整大小为640x480
    image = image.resize((640, 480))

    return image
