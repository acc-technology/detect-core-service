from PIL import Image
from PIL.ExifTags import TAGS


def get_geotagging(image):
    exif_data = image._getexif()
    if exif_data is not None:
        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)
            if tag_name == "GPSInfo":
                return value
    return None


def get_decimal_from_dms(dms):
    degrees = dms[0]
    minutes = dms[1]
    seconds = dms[2]
    return degrees + (minutes / 60.0) + (seconds / 3600.0)


# def dms_to_decimal(degrees, minutes, seconds):
#     decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
#     return decimal


def get_lat_lon(geotags):
    if geotags is None:
        return None, None
    lon_dms = geotags.get(4)
    lat_dms = geotags.get(2)
    if lon_dms is None or lat_dms is None:
        return None, None
    lon = get_decimal_from_dms(lon_dms)
    lat = get_decimal_from_dms(lat_dms)
    return lon, lat


def get_location_from_image(image_path):
    try:
        image = Image.open(image_path)
        # print(1)
        geotags = get_geotagging(image)
        lon, lat = get_lat_lon(geotags)
        return float(lon), float(lat)
    except Exception as e:
        print(f"Error: {e}")
        return 0.0, 0.0


# 示例用法
# image_path = "/home/lcj/lab/else/sam_model/blind_data_split2/train/images/0000503.jpg"
# latitude, longitude = get_location_from_image(image_path)
# if latitude is not None and longitude is not None:
#     print(f"经度: {longitude}, 纬度: {latitude}")
# else:
#     print("未找到位置信息。")
