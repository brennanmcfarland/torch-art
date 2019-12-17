from typing import Any, Iterator
import csv
import random
import functools
from copy import deepcopy

import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim

import shape_inference as sh
import transforms.image_transforms as it
import data.retrieval as rt
import data.handling as dt
import callbacks as cb
import metrics as mt
from functional.core import pipe
from utils import try_reduce_list, run_callbacks
from profiling import profile_cuda_memory_by_layer
from performance import optimize_cuda_for_fixed_input_size


# TODO: need to add functionality to add move feature maps out of VRAM - https://medium.com/syncedreview/how-to-train-a-very-large-and-deep-model-on-one-gpu-7b7edfe2d072 https://arxiv.org/pdf/1602.08124.pdf https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9
# TODO: these look most promising: https://www.sicara.ai/blog/2019-28-10-deep-learning-memory-usage-and-pytorch-optimization-tricks https://pytorch.org/docs/stable/checkpoint.html
metadata_path = 'D:/HDD Data/CMAopenaccess/data.csv'
data_dir = 'D:/HDD Data/CMAopenaccess/images/'
# TODO: see if there's anything we can do to avoid passing device everywhere without making it global/unconfigurable
# TODO: try and move general functions out


class Trainer:

    def __init__(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss


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


def get_target(get_label, class_to_index):

    def _get(metadatum):
        label = get_label(metadatum)
        label = class_to_index[-1][label]
        return torch.tensor(label)
    return _get


# composes a closure to get the image and label for a metadatum given functions for getting each separately
def prepare_example(get_image, get_label):
    def _prepare(metadatum):
        return get_image(metadatum), get_label(metadatum)
    return _prepare


def define_layers(num_classes):
    return [
        sh.Input(3),  # TODO: make this always match loader?
        lambda i: nn.Conv2d(i, 16, 3),
        lambda i: nn.ReLU(),
        lambda i: nn.Conv2d(i, 32, 4),
        lambda i: nn.ReLU(),
        lambda i: nn.Conv2d(i, 64, 5),
        lambda i: nn.ReLU(),
        lambda i: nn.MaxPool2d(2),
        lambda i: nn.Conv2d(i, 64, 6),
        lambda i: nn.ReLU(),
        lambda i: nn.Conv2d(i, 64, 5),
        lambda i: nn.ReLU(),
        lambda i: nn.MaxPool2d(2),
        lambda i: nn.Conv2d(i, 128, 4),
        lambda i: nn.ReLU(),
        lambda i: nn.Conv2d(i, 64, 4),
        lambda i: nn.ReLU(),
        lambda i: nn.Conv2d(i, 64, 3),
        lambda i: nn.ReLU(),
        lambda i: nn.Flatten(),
        lambda i: nn.Linear(i, 1024),
        lambda i: nn.ReLU(),
        lambda i: nn.Linear(i, 1024),
        lambda i: nn.ReLU(),
        lambda i: nn.Linear(i, 512),
        lambda i: nn.ReLU(),
        lambda i: nn.Linear(i, num_classes),
        lambda i: nn.Softmax()
    ]


# TODO: generalize and clean up
# TODO: device? keep in mind both data and network will need to be on same device
def create_network(layers, loader):
    inferer = sh.ShapeInferer(loader)
    for l, layer in enumerate(layers[1:]):
        layers[l+1] = layer(inferer.infer(layer, layers[:l+1]))
    return nn.Sequential(*layers[1:])


def get_output_shape(layer, input_size):
    dummy_input = torch.zeros(input_size)
    dummy_output = layer(dummy_input)
    return dummy_output.size


# get whether this module and each submodule recursively is in train or evaluation mode
def get_train_mode_tree(module):
    def _get_mode(module, mode):
        mode.append([module.training])
        for submodule in module.children():
            mode[-1].append(_get_mode(submodule, mode))
        return mode
    return _get_mode(module, [])


# set the train or evaluation state of this module and each submodule recursively
def set_train_mode_tree(module, mode):
    module.train(mode[0])
    for s, submodule in enumerate(module.children()):
        set_train_mode_tree(submodule, mode[s])


# runs the network once without modifying the loader's state as a test/for profiling
def dry_run(net, loader, trainer, train_step_func, device=None):
    def _apply():
        prev_mode = get_train_mode_tree(net)
        inputs, gtruth = iter(loader).next()
        result = train_step_func(net, trainer, device=device)(inputs, gtruth)
        set_train_mode_tree(net, prev_mode)
        return result
    return _apply


def train_step(net, trainer, device=None):
    def _apply(inputs, gtruth):
        inputs, gtruth = inputs.to(device, non_blocking=True), gtruth.to(device, non_blocking=True)
        trainer.optimizer.zero_grad()  # reset the gradients to zero

        # run the inputs through the network and compute loss relative to gtruth
        outputs = net(inputs)
        loss = trainer.loss(outputs, gtruth)
        loss.backward()
        trainer.optimizer.step()
        return loss
    return _apply


def train(net, loader, trainer, callbacks=None, device=None, epochs=1):
    if callbacks is None:
        callbacks = []

    steps_per_epoch = len(loader)
    callbacks = [callback(steps_per_epoch) for callback in callbacks]
    take_step = train_step(net, trainer, device=device)

    for epoch in range(epochs):
        print('----BEGIN EPOCH ', epoch, '----')
        for step, (inputs, gtruth) in enumerate(loader):
            loss = take_step(inputs, gtruth)
            run_callbacks("on_step", callbacks, loss, step, epoch)
        run_callbacks("on_epoch_end", callbacks)
    print('TRAINING COMPLETE!')


def test(net, loader, metrics=None, device=None):
    if metrics is None:
        metrics = []

    print('TESTING')
    with torch.no_grad():
        for (inputs, gtruth) in loader:
            inputs, gtruth = inputs.to(device, non_blocking=True), gtruth.to(device, non_blocking=True)
            outputs = net(inputs)
            run_callbacks("on_item", metrics, inputs, outputs, gtruth)
    return try_reduce_list(run_callbacks("on_end", metrics))


# validation is just an alias for testing
validate = test


def get_from_metadata():
    get = rt.get_img_from_file_or_url(img_format='JPEG')

    def _apply(metadatum):
        filepath = data_dir + metadatum[0] + '.jpg'
        url = metadatum[-1]
        return get(filepath, url)
    return _apply


def get_label(metadatum):
    return metadatum[1]


def run():
    metadata, len_metadata, metadata_headers, class_to_index, index_to_class, num_classes = load_metadata(
        metadata_path,
        cols=(COL_ID, COL_TYPE, COL_IMG_WEB),
        class_cols=(COL_TYPE,)
    )
    len_metadata = 31149  # TODO: either the dataset is corrupted/in a different format after this point or the endpoint was down last I tried
    metadata = metadata[:len_metadata]

    # shuffle at beginning to get random sampling for train, test and validation datasets
    random.shuffle(metadata)

    print(class_to_index)
    print(index_to_class)

    # TODO: make this easier to read/abstracted out to do for train, validation, test all at once?
    # TODO: don't restrict it to just those though, or to requiring metadata
    data_split_points = (None, 512, 256, 0)

    train_metadata, validation_metadata, test_metadata = (
        metadata[n:m] for m, n in zip(data_split_points[:-1], data_split_points[1:])
    )

    dataset, validation_dataset, test_dataset = (
        dt.metadata_to_prepared_dataset(
            m,
            prepare_example(
                pipe(get_from_metadata(), it.random_fit_to((256, 256)), it.to_tensor()),
                get_target(get_label, class_to_index)
            )
        )
        for m in (train_metadata, validation_metadata, test_metadata)
    )

    print(metadata_headers)
    print(metadata[:10])
    print(len(dataset))

    loader, validation_loader, test_loader = (
        dt.dataset_to_loader(
            d,
            batch_size=16,
            shuffle=True,  # shuffle every epoch so learning and testing is order-independent
            num_workers=0,
            pin_memory=True
        ) for d in (dataset, validation_dataset, test_dataset))

    dataiter = iter(loader)
    demo_batch = dataiter.next()

    is_cuda = cuda.is_available()

    device = torch.device("cuda:0" if is_cuda else "cpu")
    print("using ", device)

    layers = define_layers(num_classes)
    net = create_network(layers, loader)
    net = net.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    trainer = Trainer(optimizer, loss_func)

    metrics = [mt.calc_category_accuracy()]

    if is_cuda:
        profile_cuda_memory_by_layer(net, dry_run(net, loader, trainer, train_step, device=device), device=device)
        optimize_cuda_for_fixed_input_size()

    accuracy = test(net, test_loader, metrics, device)
    print("pre-training accuracy: ", accuracy)

    callbacks = [
        cb.tensorboard_record_loss(),
        cb.calc_interval_avg_loss(print_interval=16),
        cb.validate(validate, net, validation_loader, metrics, device)
    ]

    train(net, loader, trainer, callbacks, device, 3)

    accuracy = test(net, test_loader, metrics, device)
    print("post-training accuracy: ", accuracy)


if __name__ == '__main__':
    COL_ID = 0
    COL_CREATION_DATE = 9
    COL_TYPE = 18
    COL_IMG_WEB = -4  # lower resolution makes for smaller dataset/faster training
    run()
