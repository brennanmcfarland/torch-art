import os
import requests
from PIL import Image
from io import BytesIO
import csv


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


def load_metadata(path, cols, class_cols=tuple(), valid_only=True):
    metadata = []
    # one dict for each class col
    class_to_index = [{}] * len(class_cols)
    index_to_class = [{}] * len(class_cols)
    next_indices = [0] * len(class_cols) # next index for a new class value
    with open(path, 'r', newline='', encoding="utf8") as metadata_file:
        reader = csv.reader(metadata_file)
        headers = next(reader)
        for row in reader:
            if len(row) != 0:
                metadatum = [row[c] for c in cols]
                # for all class cols, add their vals to the class_to_index and index_to_class dicts if not there already
                for c, class_col in enumerate(class_cols):
                    if not row[class_col] in class_to_index[c]:
                        class_to_index[c][row[class_col]] = next_indices[c]
                        index_to_class[c][next_indices[c]] = row[class_col]
                        next_indices[c] += 1
                if valid_only and '' in metadatum:
                    continue
                metadata.append(metadatum)
    len_metadata = len(metadata)
    # split off the headers
    return metadata, len_metadata, headers, class_to_index, index_to_class, next_indices[-1]