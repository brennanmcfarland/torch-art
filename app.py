from typing import Any, Iterator
import random
import functools
from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
import matplotlib.pyplot as plt

import arc23.shape_inference as sh
import arc23.data.retrieval as rt
import arc23.data.handling.dali as dt
import arc23.data.handling.handling as hd
import arc23.callbacks as cb
import arc23.metrics as mt
import arc23.output as out
import arc23.model.initialization as ninit
from arc23.model.execution import dry_run, train, validate, test, train_step, Trainer
from arc23.profiling import profile_cuda_memory_by_layer
from arc23.performance import \
    optimize_cuda_for_fixed_input_size, checkpoint_sequential, adapt_checkpointing
from arc23.model import hooks as mh
from arc23.__utils import on_interval
from arc23.layers.residual import Residual


metadata_path = './preprocessed_data.csv'
data_dir = '/media/guest/Main Storage/HDD Data/CMAopenaccess/preprocessed_images'
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
        lambda i: nn.Conv2d(i, 32, 3),
        lambda i: nn.LeakyReLU(),
        lambda i: nn.BatchNorm2d(i),
        lambda i: nn.Conv2d(i, 64, 4),
        lambda i: nn.LeakyReLU(),
        lambda i: nn.BatchNorm2d(i),
        lambda i: Residual(
            nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(i, 64, 4),
                nn.LeakyReLU(),
                nn.BatchNorm2d(i),
                nn.Conv2d(i, 64, 4),
                nn.LeakyReLU(),
                nn.BatchNorm2d(i),
            )
        ),
        lambda i: nn.Conv2d(i, 128, 4),
        lambda i: nn.LeakyReLU(),
        lambda i: nn.BatchNorm2d(i),
        lambda i: Residual(
            nn.Sequential(
                nn.ReflectionPad2d(2),
                nn.Conv2d(i, 128, 5),
                nn.LeakyReLU(),
                nn.BatchNorm2d(i),
                nn.ReflectionPad2d(2),
                nn.Conv2d(i, 128, 5),
                nn.LeakyReLU(),
                nn.BatchNorm2d(i),
            )
        ),
        lambda i: nn.MaxPool2d(2),
        lambda i: nn.Conv2d(i, 128, 4),
        lambda i: nn.LeakyReLU(),
        lambda i: nn.BatchNorm2d(i),
        lambda i: Residual(
            nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(i, 128, 4),
                nn.LeakyReLU(),
                nn.BatchNorm2d(i),
                nn.Conv2d(i, 128, 4),
                nn.LeakyReLU(),
                nn.BatchNorm2d(i),
            )
        ),
        lambda i: nn.Conv2d(i, 128, 5),
        lambda i: nn.LeakyReLU(),
        lambda i: nn.BatchNorm2d(i),
        lambda i: nn.MaxPool2d(2),
        lambda i: nn.Conv2d(i, 256, 4),
        lambda i: nn.LeakyReLU(),
        lambda i: nn.BatchNorm2d(i),
        lambda i: Residual(
            nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(i, 256, 4),
                nn.LeakyReLU(),
                nn.BatchNorm2d(i),
                nn.Conv2d(i, 256, 4),
                nn.LeakyReLU(),
                nn.BatchNorm2d(i),
            )
        ),
        lambda i: nn.Conv2d(i, 512, 4),
        lambda i: nn.LeakyReLU(),
        lambda i: nn.BatchNorm2d(i),
        lambda i: Residual(
            nn.Sequential(
                nn.ReflectionPad2d(6),
                nn.Conv2d(i, 512, 4),
                nn.LeakyReLU(),
                nn.BatchNorm2d(i),
                nn.Conv2d(i, 512, 4),
                nn.LeakyReLU(),
                nn.BatchNorm2d(i),
                nn.Conv2d(i, 512, 4),
                nn.LeakyReLU(),
                nn.BatchNorm2d(i),
                nn.Conv2d(i, 512, 4),
                nn.LeakyReLU(),
                nn.BatchNorm2d(i),
            )
        ),
        lambda i: nn.Conv2d(i, 128, 4),
        lambda i: nn.LeakyReLU(),
        lambda i: nn.BatchNorm2d(i),
        lambda i: nn.Conv2d(i, 128, 3),
        lambda i: nn.LeakyReLU(),
        lambda i: nn.BatchNorm2d(i),
        lambda i: Residual(
            nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(i, 128, 3),
                nn.LeakyReLU(),
                nn.BatchNorm2d(i),
                nn.Conv2d(i, 128, 3),
                nn.LeakyReLU(),
                nn.BatchNorm2d(i),
                nn.Conv2d(i, 128, 3),
                nn.LeakyReLU(),
                nn.BatchNorm2d(i),
            )
        ),
        lambda i: nn.Flatten(),
        lambda i: nn.Linear(i, 1024),
        lambda i: nn.LeakyReLU(),
        lambda i: nn.BatchNorm1d(i),
        lambda i: Residual(
            nn.Sequential(
                nn.Linear(i, 1024),
                nn.LeakyReLU(),
                nn.BatchNorm1d(i),
                nn.Linear(i, 1024),
                nn.LeakyReLU(),
                nn.BatchNorm1d(i),
                nn.Linear(i, 1024),
                nn.LeakyReLU(),
                nn.BatchNorm1d(i),
            )
        ),
        lambda i: nn.Linear(i, 512),
        lambda i: nn.LeakyReLU(),
        lambda i: nn.BatchNorm1d(i),
        lambda i: nn.Linear(i, num_classes),
        lambda i: nn.BatchNorm1d(i),
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


def make_trainer(net):
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    return Trainer(optimizer, loss_func)


def run():

    print('preparing metadata...')

    metadata, len_metadata, metadata_headers, class_to_index, index_to_class, num_classes = rt.load_metadata(
        metadata_path,
        cols=(COL_IMG_WEB, COL_TYPE),
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
    # demo_batch = next(dataiter)
    # print(demo_batch['labels'][0].cpu().item())
    # demo_img = np.swapaxes(demo_batch['inputs'][0].cpu(), 0, -1)
    # plt.imshow(demo_img / 256.)
    # plt.show()

    is_cuda = cuda.is_available()

    device = torch.device("cuda:0" if is_cuda else "cpu")
    print("using ", device)

    layers = define_layers(num_classes)
    net = ninit.from_iterable(sh.infer_shapes(layers, loader))
    net = net.to(device)

    metrics = [mt.category_accuracy()]

    net = adapt_checkpointing(
        checkpoint_sequential,
        lambda n: dry_run(n, loader, make_trainer, functools.partial(train_step, squeeze_gtruth=True), device=device)(),
        net
    )

    if is_cuda:
        profile_cuda_memory_by_layer(
            net,
            dry_run(net, loader, make_trainer, functools.partial(train_step, squeeze_gtruth=True), device=device),
            device=device
        )
        optimize_cuda_for_fixed_input_size()

    # the trainer is not used above or it would be modified
    trainer = make_trainer(net)

    accuracy = test(net, test_loader, metrics, device, squeeze_gtruth=True)
    print("pre-training accuracy: ", accuracy)

    callbacks = {
        "on_step": [
            out.record_tensorboard_scalar(cb.loss(), out.tensorboard_writer()),
            lambda steps_per_epoch: on_interval(
                out.print_with_step(
                    cb.interval_avg_loss(interval=1)
                )(steps_per_epoch),
                16
            )
        ],
        "on_epoch_start": [
            out.print_tables(
                cb.layer_stats(
                    net,
                    dry_run(net, loader, trainer, functools.partial(train_step, squeeze_gtruth=True), device=device),
                    [
                        mh.weight_stats_hook((torch.mean,)),
                        mh.output_stats_hook((torch.var,)),
                        mh.grad_stats_hook((torch.var_mean,)),
                    ]
                ), titles=["WEIGHT STATS", "OUTPUT_STATS", "GRADIENT STATS"], headers=["Layer", "Value"]
            ),
        ],
        "on_epoch_end": [
            cb.validate(functools.partial(validate, squeeze_gtruth=True), net, validation_loader, metrics, device)
        ]
    }

    train(net, loader, trainer, callbacks, device, 5, squeeze_gtruth=True)

    accuracy = test(net, test_loader, metrics, device, squeeze_gtruth=True)
    print("post-training accuracy: ", accuracy)


if __name__ == '__main__':
    COL_TYPE = 1
    COL_IMG_WEB = 0  # lower resolution makes for smaller dataset/faster training
    run()
