from io import BytesIO
import os
import torch
import torchvision
import torchvision.transforms as transforms
import csv
import requests
from PIL import Image


metadata_path = 'D:/HDD Data/CMAopenaccess/data.csv'
data_dir = 'D:/HDD Data/CMAopenaccess/images/'


def load_metadata(path, cols):
    metadata = []
    with open(path, 'r', newline='', encoding="utf8") as metadata_file:
        reader = csv.reader(metadata_file)
        for row in reader:
            metadatum = [row[c] for c in cols]
            metadata.append(metadatum)
    len_metadata = len(metadata)
    # split off the headers
    headers, metadata = metadata[0], metadata[1:]
    return metadata, len_metadata, headers


def get_image(metadatum):
    # TODO: generalize indices?

    # pull the image from the api endpoint and save it if we don't have it, else load it from disk
    img_path = data_dir + metadatum[0] + '.jpg'
    if os.path.exists(img_path):
        img = Image.open(img_path)
    else:
        api_response = requests.get(metadatum[-1]).content
        response_bytes = BytesIO(api_response)
        img = Image.open(response_bytes)
        img.save(img_path, "JPEG")
    return img


def get_label(metadatum):
    return metadatum[1]


# composes a closure to get the image and label for a metadatum given functions for getting each separately
def prepare_example(get_image, get_label):
    def _prepare(metadatum):
        return get_image(metadatum), get_label(metadatum)
    return _prepare


class PreparedDataset(torch.utils.data.Dataset):

    def __init__(self, metadata, metadata_len, prepare):
        self.metadata = metadata
        self.metadata_len = metadata_len
        self.prepare = prepare

    def __getitem__(self, item):
        return self.prepare(self.metadata[item])

    def __len__(self):
        return self.metadata_len


if __name__ == '__main__':
    COL_ID = 0
    COL_CREATION_DATE = 9
    COL_TYPE = 18
    COL_IMG_WEB = -4  # lower resolution makes for smaller dataset/faster training

    metadata, len_metadata, metadata_headers = load_metadata(
        metadata_path,
        cols=(COL_ID, COL_TYPE, COL_IMG_WEB)
    )
    # TODO: shuffle metadata
    dataset = PreparedDataset(metadata, len_metadata, prepare_example(get_image, get_label))
    print(metadata_headers)
    print(metadata[:10])
    print(len(dataset))
    img = dataset.__getitem__(0)
    print(img)
    #loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=6, pin_memory=True)