from typing import Any, Iterator
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
import model.initialization as ninit
from model.execution import dry_run, train, validate, test, train_step, Trainer
from profiling import profile_cuda_memory_by_layer
from performance import optimize_cuda_for_fixed_input_size, checkpoint_sequential


metadata_path = 'D:/HDD Data/CMAopenaccess/data.csv'
data_dir = 'D:/HDD Data/CMAopenaccess/images/'
# TODO: see if there's anything we can do to avoid passing device everywhere without making it global/unconfigurable
# TODO: next step: make this a library


# allows for variable-sized inputs in a batch
def custom_collate(batch):
    inputs, targets = [e[0] for e in batch], [e[1] for e in batch]
    targets = torch.LongTensor(targets)
    return [inputs, targets]


def define_layers(num_classes):
    return [
        sh.Input(),
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
    metadata, len_metadata, metadata_headers, class_to_index, index_to_class, num_classes = rt.load_metadata(
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
            dt.prepare_example(
                pipe(get_from_metadata(), it.random_fit_to((256, 256)), it.to_tensor()),
                dt.get_target(get_label, class_to_index)
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
    net = ninit.from_iterable(sh.infer_shapes(layers, loader))
    net = net.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    trainer = Trainer(optimizer, loss_func)

    metrics = [mt.calc_category_accuracy()]

    # TODO: make this dynamically set no/the correct number of checkpoints based on avail memory
    #net = checkpoint_sequential(net, 3)

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
