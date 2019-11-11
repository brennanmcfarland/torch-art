from io import BytesIO
import os
from typing import Any, Iterator

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as torchdata
import csv
import requests
from PIL import Image


metadata_path = 'D:/HDD Data/CMAopenaccess/data.csv'
data_dir = 'D:/HDD Data/CMAopenaccess/images/'


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
    return metadata, len_metadata, headers, class_to_index, index_to_class


# allows for variable-sized inputs in a batch
def custom_collate(batch):
    inputs, targets = [e[0] for e in batch], [e[1] for e in batch]
    targets = torch.LongTensor(targets)
    return [inputs, targets]

# composes a closure to get the input given a function for retrieving the image
# ie, formats the retrieved image as a tensor and returns it
def get_input(get_image):
    to_tensor = transforms.ToTensor()

    def _get(metadatum):
        img = get_image(metadatum)
        return to_tensor(img)
    return _get


def get_target(get_label):

    def _get(metadatum):
        label = get_label(metadatum)
        label = class_to_index[-1][label] # TODO: generalize index? rn it relies on target being last index
        return torch.nn.functional.one_hot(torch.tensor(label))
    return _get


# pull the image from the api endpoint and save it if we don't have it, else load it from disk
def get_image(metadatum):
    # TODO: generalize indices?
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


class PreparedDataset(torchdata.Dataset):

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

    metadata, len_metadata, metadata_headers, class_to_index, index_to_class = load_metadata(
        metadata_path,
        cols=(COL_ID, COL_TYPE, COL_IMG_WEB),
        class_cols=(COL_TYPE,)
    )
    print(class_to_index)
    print(index_to_class)
    # TODO: shuffle metadata
    dataset = PreparedDataset(
        metadata,
        len_metadata,
        prepare_example(
            get_input(get_image),
            get_target(get_label)
        )
    )
    print(metadata_headers)
    print(metadata[:10])
    print(len(dataset))
    img = dataset.__getitem__(0)
    print(img)
    loader = torchdata.DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate
    )

    dataiter: Iterator[Any] = iter(loader)
    x = dataiter.next()
