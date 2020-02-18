from typing import Any, Iterator
import random
import functools
from copy import deepcopy

import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim

import arctic_flaming_monkey_typhoon.shape_inference as sh
import arctic_flaming_monkey_typhoon.transforms.image_transforms as it
import arctic_flaming_monkey_typhoon.data.retrieval as rt
import arctic_flaming_monkey_typhoon.data.handling as dt
import arctic_flaming_monkey_typhoon.callbacks as cb
import arctic_flaming_monkey_typhoon.metrics as mt
from arctic_flaming_monkey_typhoon.functional.core import pipe
import arctic_flaming_monkey_typhoon.model.initialization as ninit
from arctic_flaming_monkey_typhoon.model.execution import dry_run, train, validate, test, train_step, Trainer
from arctic_flaming_monkey_typhoon.profiling import profile_cuda_memory_by_layer
from arctic_flaming_monkey_typhoon.performance import \
    optimize_cuda_for_fixed_input_size, checkpoint_sequential, adapt_checkpointing


metadata_path = '/media/guest/Main Storage/HDD Data/CMAopenaccess/data.csv'
data_dir = '/media/guest/Main Storage/HDD Data/CMAopenaccess/preprocessed_images/'
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
        # dt.metadata_to_prepared_dataset(
        #     m,
        #     dt.prepare_example(
        #         pipe(get_from_metadata(), it.to_tensor()),
        #         dt.get_target(get_label, class_to_index)
        #     )
        # )
        dt.DALIDataset(metadata=m, data_dir=data_dir, class_to_index=class_to_index)
        for m in (train_metadata, validation_metadata, test_metadata)
    )

    dataset.build()
    validation_dataset.build()
    test_dataset.build()

    print(metadata_headers)
    print(metadata[:10])
    print(len(dataset))

    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    loader, validation_loader, test_loader = (
            dt.DALIDataset(metadata=m, data_dir=data_dir, class_to_index=class_to_index)
        for m in (train_metadata, validation_metadata, test_metadata))

    loader.build()
    validation_loader.build()
    test_loader.build()

    loader = DALIGenericIterator(loader, ['data', 'label'], len(train_metadata))
    validation_loader = DALIGenericIterator(validation_loader, ['data', 'label'], len(validation_metadata))
    test_loader = DALIGenericIterator(test_loader, ['data', 'label'], len(test_metadata))

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

    net = adapt_checkpointing(
        checkpoint_sequential,
        lambda n: dry_run(n, loader, trainer, train_step, device=device)(),
        net
    )

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
