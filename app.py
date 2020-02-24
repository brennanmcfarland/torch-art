from typing import Any, Iterator
import random
import functools
from copy import deepcopy

import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim

import arc23.shape_inference as sh
import arc23.data.retrieval as rt
import arc23.data.handling.dali as dt
import arc23.data.handling.handling as hd
import arc23.callbacks as cb
import arc23.metrics as mt
import arc23.model.initialization as ninit
from arc23.model.execution import dry_run, train, validate, test, train_step, Trainer
from arc23.profiling import profile_cuda_memory_by_layer
from arc23.performance import \
    optimize_cuda_for_fixed_input_size, checkpoint_sequential, adapt_checkpointing

metadata_path = '/media/guest/Main Storage/HDD Data/CMAopenaccess/data.csv'
data_dir = '/media/guest/Main Storage/HDD Data/CMAopenaccess/preprocessed_images/'
train_label_out_path = './train_labels.csv'
validation_label_out_path = './validation_labels.csv'
test_label_out_path = './test_labels.csv'


# TODO: see if there's anything we can do to avoid passing device everywhere without making it global/unconfigurable


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
        lambda i: nn.Linear(i, 4096),
        lambda i: nn.ReLU(),
        lambda i: nn.Linear(i, 2048),
        lambda i: nn.ReLU(),
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
    get = rt.get_img_from_file_or_url(img_format='PNG')

    def _apply(metadatum):
        filepath = data_dir + metadatum[0] + '.png'
        url = metadatum[-1]
        return get(filepath, url)

    return _apply


def get_label(metadatum):
    return metadatum[1]


# write metadata image-label mappings to a file so the pipeline can efficiently load them, and return a loader
def make_loader(label_out_dir, metadata, class_to_index):
    hd.write_transformed_metadata_to_file(
        metadata, label_out_dir, lambda m: m[0] + '.png ' + str(class_to_index[0][m[1]]) + '\n'
    )
    return dt.DALIIterableDataset(
        dt.dali_standard_image_classification_pipeline(data_dir, label_out_dir),
        metadata=metadata,
        batch_size=16
    )


def run():

    print('preparing metadata...')

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

    print(metadata_headers)

    print('initializing loaders...')

    loader, validation_loader, test_loader = (make_loader(ldir, m, class_to_index)
                                              for ldir, m in zip(
        (train_label_out_path, validation_label_out_path, test_label_out_path),
        (train_metadata, validation_metadata, test_metadata)
    ))

    loader.build()
    validation_loader.build()
    test_loader.build()

    dataiter = iter(loader)
    demo_batch = next(dataiter)

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

    net = adapt_checkpointing(
        checkpoint_sequential,
        lambda n: dry_run(n, loader, trainer, functools.partial(train_step, squeeze_gtruth=True), device=device)(),
        net
    )

    if is_cuda:
        profile_cuda_memory_by_layer(
            net,
            dry_run(net, loader, trainer, functools.partial(train_step, squeeze_gtruth=True), device=device),
            device=device
        )
        optimize_cuda_for_fixed_input_size()

    accuracy = test(net, test_loader, metrics, device)
    print("pre-training accuracy: ", accuracy)

    callbacks = [
        cb.tensorboard_record_loss(),
        cb.calc_interval_avg_loss(print_interval=16),
        cb.validate(validate, net, validation_loader, metrics, device)
    ]

    train(net, loader, trainer, callbacks, device, 100, squeeze_gtruth=True)

    accuracy = test(net, test_loader, metrics, device)
    print("post-training accuracy: ", accuracy)


if __name__ == '__main__':
    COL_ID = 0
    COL_CREATION_DATE = 9
    COL_TYPE = 18
    COL_IMG_WEB = -4  # lower resolution makes for smaller dataset/faster training
    run()
