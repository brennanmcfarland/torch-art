from typing import Any, Iterator
import csv
import random

import torch
import torch.nn as nn
import torch.utils.data as torchdata
import torch.utils.tensorboard as tensorboard
import torch.optim as optim

import transforms.image_transforms as it
import data.retrieval as rt
from functional.core import pipe


# TODO: need to add functionality to add move feature maps out of VRAM - https://medium.com/syncedreview/how-to-train-a-very-large-and-deep-model-on-one-gpu-7b7edfe2d072
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
    return metadata, len_metadata, headers, class_to_index, index_to_class, next_indices[-1]


# allows for variable-sized inputs in a batch
def custom_collate(batch):
    inputs, targets = [e[0] for e in batch], [e[1] for e in batch]
    targets = torch.LongTensor(targets)
    return [inputs, targets]


def get_target(get_label):

    def _get(metadatum):
        label = get_label(metadatum)
        label = class_to_index[-1][label] # TODO: generalize index? rn it relies on target being last index
        return torch.tensor(label)
    return _get


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


# TODO: generalize
def define_network(num_classes):
    # TODO: shape inference (probably make it functional, maybe a subclass that calls it?)
    return nn.Sequential(
        nn.Conv2d(3, 16, 3),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 5),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 64, 6),
        nn.ReLU(),
        nn.Conv2d(64, 64, 5),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 4),
        nn.ReLU(),
        nn.Conv2d(128, 64, 4),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3),
        nn.ReLU(),
        nn.Flatten(),
        # TODO: need to figure out input size dynamically
        nn.Linear(153664, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes),
        nn.Softmax()
    )


# TODO: put most of these params in a compile closure?
def train(net, loader, loss_func, optimizer, device, epochs, print_interval=16):
    # TODO: abstract out tensorboard writer
    tensorboard_writer = tensorboard.SummaryWriter()

    steps_per_epoch = len(loader)
    for epoch in range(epochs):
        print('----BEGIN EPOCH ', epoch, '----')
        interval_avg_loss = 0.0
        for step, (inputs, gtruth) in enumerate(loader):

            inputs, gtruth = inputs.to(device, non_blocking=True), gtruth.to(device, non_blocking=True)
            optimizer.zero_grad() # reset the gradients to zero

            # run the inputs through the network and compute loss relative to gtruth
            outputs = net(inputs)
            loss = loss_func(outputs, gtruth)
            loss.backward()
            optimizer.step()

            # TODO: as above, abstract out tensorboard
            tensorboard_writer.add_scalar('loss', loss, epoch * steps_per_epoch + step)

            # print statistics
            interval_avg_loss += loss.item()
            if step % print_interval == 0:
                print('EPOCH ', epoch, ' STEP ', step, '/', steps_per_epoch, interval_avg_loss / print_interval)
                interval_avg_loss = 0
    print('TRAINING COMPLETE!')


# TODO: abstract out with train?
# TODO: also abstract out to be able to inject metrics
def test(net, loader, device):
    print('TESTING')
    correct, total = 0, 0
    with torch.no_grad():
        for (inputs, gtruth) in loader:
            inputs, gtruth = inputs.to(device), gtruth.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += gtruth.size(0)
            correct += (predicted == gtruth).sum().item()
    return correct / total


# TODO: obviously train and test data should be kept separate, make functions for it
# TODO: also add validation set capabilities
# TODO: should be easy to make a func that accepts a dataset or loader and splits it in two to avoid duplicating
# TODO: preprocessing code
if __name__ == '__main__':
    COL_ID = 0
    COL_CREATION_DATE = 9
    COL_TYPE = 18
    COL_IMG_WEB = -4  # lower resolution makes for smaller dataset/faster training

    metadata, len_metadata, metadata_headers, class_to_index, index_to_class, num_classes = load_metadata(
        metadata_path,
        cols=(COL_ID, COL_TYPE, COL_IMG_WEB),
        class_cols=(COL_TYPE,)
    )
    len_metadata = 31149 # TODO: either the dataset is corrupted/in a different format after this point or the endpoint was down last I tried
    metadata = metadata[:len_metadata]

    random.shuffle(metadata)

    print(class_to_index)
    print(index_to_class)
    # TODO: shuffle metadata
    dataset = PreparedDataset(
        metadata,
        len_metadata,
        prepare_example(
            pipe(rt.get_from_file_or_url(data_dir), it.random_fit_to((256, 256)), it.to_tensor()),
            get_target(rt.get_label)
        )
    )

    test_dataset = PreparedDataset(
        metadata[:256],
        256,
        prepare_example(
            pipe(rt.get_from_file_or_url(data_dir), it.random_fit_to((256, 256)), it.to_tensor()),
            get_target(rt.get_label)
        )
    )

    print(metadata_headers)
    print(metadata[:10])
    print(len(dataset))

    loader = torchdata.DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = torchdata.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    dataiter = iter(loader)
    demo_batch = dataiter.next()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using ", device)

    net = define_network(num_classes)
    net = net.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    accuracy = test(net, test_loader, device)
    print("pre-training accuracy: ", accuracy)

    train(net, loader, loss_func, optimizer, device, epochs=3)

    accuracy = test(net, test_loader, device)
    print("post-training accuracy: ", accuracy)
