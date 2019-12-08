from typing import Any, Iterator
import csv
import random
from copy import deepcopy

import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.utils.data as torchdata
import torch.optim as optim

import transforms.image_transforms as it
import data.retrieval as rt
import callbacks as cb
from functional.core import pipe


# TODO: need to add functionality to add move feature maps out of VRAM - https://medium.com/syncedreview/how-to-train-a-very-large-and-deep-model-on-one-gpu-7b7edfe2d072 https://arxiv.org/pdf/1602.08124.pdf https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9
# TODO: these look most promising: https://www.sicara.ai/blog/2019-28-10-deep-learning-memory-usage-and-pytorch-optimization-tricks https://pytorch.org/docs/stable/checkpoint.html
metadata_path = 'D:/HDD Data/CMAopenaccess/data.csv'
data_dir = 'D:/HDD Data/CMAopenaccess/images/'
# TODO: see if there's anything we can do to avoid passing device everywhere without making it global/unconfigurable


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


# TODO: rename mode?
# get whether this module and each submodule recursively is in train or evaluation mode
def get_mode_tree(module):
    def _get_mode(module, mode):
        mode.append([module.training])
        for submodule in module.children():
            mode[-1].append(_get_mode(submodule, mode))
        return mode
    return _get_mode(module, [])


# set the train or evaluation state of this module and each submodule recursively
def set_mode_tree(module, mode):
    module.train(mode[0])
    for s, submodule in enumerate(module.children()):
        set_mode_tree(submodule, mode[s])


# runs the network once without modifying the loader's state as a test/for profiling
def dry_run(net, loader, trainer, device=None):
    def _apply():
        pass
        prev_mode = get_mode_tree(net)
        dryrun_loader = deepcopy(loader)
        inputs, gtruth = iter(loader).next()
        train_step(net, trainer, device=device)(inputs, gtruth)
        set_mode_tree(net, prev_mode)
    return _apply


# NOTE: only works on CUDA devices
def profile_cuda_memory_by_layer(net, run_func, device=None):
    profiler_hooks = []

    def _profile_layer(net, input, output):
        print(type(net).__name__, cuda.memory_allocated(device), cuda.memory_cached(device))

    def _add_profiler_hook(net):
        profiler_hooks.append(net.register_forward_hook(_profile_layer))

    print("CUDA MEMORY PROFILE")

    cuda.empty_cache()
    cuda.reset_max_memory_allocated(device)
    cuda.reset_max_memory_cached(device)
    print("CUDA device initial allocated memory: ", cuda.memory_allocated(device))
    print("CUDA device initial cached memory: ", cuda.memory_cached(device))

    print('Name Allocated Cached')
    net.apply(_add_profiler_hook)

    # train step
    run_func()

    print("CUDA device max allocated memory: ", cuda.max_memory_allocated(device))
    print("CUDA device max cached memory: ", cuda.max_memory_cached(device))

    for h in profiler_hooks:
        h.remove()

    cuda.empty_cache()
    cuda.reset_max_memory_allocated(device)
    cuda.reset_max_memory_cached(device)


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
            for callback in callbacks:
                callback["on_step"](loss, step, epoch)
    print('TRAINING COMPLETE!')


# TODO: abstract out to be able to inject metrics
def test(net, loader, device=None):
    print('TESTING')
    correct, total = 0, 0
    with torch.no_grad():
        for (inputs, gtruth) in loader:
            inputs, gtruth = inputs.to(device, non_blocking=True), gtruth.to(device, non_blocking=True)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += gtruth.size(0)
            correct += (predicted == gtruth).sum().item()
    return correct / total


# TODO: obviously train and test data should be kept separate, make functions for it
# TODO: also add validation set capabilities
# TODO: should be easy to make a func that accepts a dataset or loader and splits it in two to avoid duplicating
# TODO: preprocessing code
def run():
    metadata, len_metadata, metadata_headers, class_to_index, index_to_class, num_classes = load_metadata(
        metadata_path,
        cols=(COL_ID, COL_TYPE, COL_IMG_WEB),
        class_cols=(COL_TYPE,)
    )
    len_metadata = 31149  # TODO: either the dataset is corrupted/in a different format after this point or the endpoint was down last I tried
    metadata = metadata[:len_metadata]

    # shuffle at beginning to get random sampling for train and test datasets
    random.shuffle(metadata)

    print(class_to_index)
    print(index_to_class)

    dataset = PreparedDataset(
        metadata,
        len_metadata,
        prepare_example(
            pipe(rt.get_from_file_or_url(data_dir), it.random_fit_to((256, 256)), it.to_tensor()),
            get_target(rt.get_label, class_to_index)
        )
    )

    test_dataset = PreparedDataset(
        metadata[:256],
        256,
        prepare_example(
            pipe(rt.get_from_file_or_url(data_dir), it.random_fit_to((256, 256)), it.to_tensor()),
            get_target(rt.get_label, class_to_index)
        )
    )

    print(metadata_headers)
    print(metadata[:10])
    print(len(dataset))

    loader = torchdata.DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,  # shuffle every epoch so learning is order-independent
        num_workers=0,
        pin_memory=True,
    )

    test_loader = torchdata.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=True,  # shuffle every epoch so learning is order-independent
        num_workers=0,
        pin_memory=True,
    )

    dataiter = iter(loader)
    demo_batch = dataiter.next()

    is_cuda = cuda.is_available()

    device = torch.device("cuda:0" if is_cuda else "cpu")
    print("using ", device)

    net = define_network(num_classes)
    net = net.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    trainer = Trainer(optimizer, loss_func)

    if is_cuda:
        profile_cuda_memory_by_layer(net, dry_run(net, loader, trainer, device=device), device=device)

    accuracy = test(net, test_loader, device)
    print("pre-training accuracy: ", accuracy)

    callbacks = [
        cb.tensorboard_record_loss(),
        cb.calc_interval_avg_loss(print_interval=16)
    ]

    train(net, loader, trainer, callbacks, device, 3)

    accuracy = test(net, test_loader, device)
    print("post-training accuracy: ", accuracy)


if __name__ == '__main__':
    COL_ID = 0
    COL_CREATION_DATE = 9
    COL_TYPE = 18
    COL_IMG_WEB = -4  # lower resolution makes for smaller dataset/faster training
    run()
