import os
import requests
from PIL import Image
from io import BytesIO


# pull the image from the api endpoint and save it if we don't have it, else load it from disk
def get_from_file_or_url(data_dir):
    def _apply(metadatum):
        # TODO: generalize indices?
        img_path = data_dir + metadatum[0] + '.jpg'
        img = from_file(img_path)
        if img is None:
            img = from_url(metadatum[-1])
            img.save(img_path, "JPEG")
        return img.convert('RGB')  # convert to rgb if not already (eg if grayscale)
    return _apply


# TODO: fix this up too
def get_label(metadatum):
    return metadatum[1]


def from_url(url):
    api_response = requests.get(url).content
    response_bytes = BytesIO(api_response)
    return Image.open(response_bytes)


def from_file(path):
    if os.path.exists(path):
        return Image.open(path)
    else:
        return None
