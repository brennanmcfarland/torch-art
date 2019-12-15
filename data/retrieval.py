import os
import requests
from PIL import Image
from io import BytesIO


# pull the image from the api endpoint and save it if we don't have it, else load it from disk
def get_img_from_file_or_url(img_format='JPEG'):
    def _apply(filepath, url):
        img = from_file(filepath)
        if img is None:
            img = from_url(url)
            img.save(filepath, img_format)
        return img.convert('RGB')  # convert to rgb if not already (eg if grayscale)
    return _apply


def from_url(url):
    api_response = requests.get(url).content
    response_bytes = BytesIO(api_response)
    return Image.open(response_bytes)


def from_file(path):
    if os.path.exists(path):
        return Image.open(path)
    else:
        return None
