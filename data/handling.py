import torch.utils.data as torchdata
import torch


class PreparedDataset(torchdata.Dataset):

    def __init__(self, metadata, metadata_len, prepare):
        self.metadata = metadata
        self.metadata_len = metadata_len
        self.prepare = prepare

    def __getitem__(self, item):
        return self.prepare(self.metadata[item])

    def __len__(self):
        return self.metadata_len


def metadata_to_prepared_dataset(metadata, *args, **kwargs):
    return PreparedDataset(
        metadata,
        len(metadata),
        *args,
        **kwargs
    )


def dataset_to_loader(dataset, *args, **kwargs):
    return torchdata.DataLoader(dataset, *args, **kwargs)


# given a function to get a label and a list of class dict columns (the last one being for the label), return a function
# that consumes a metadatum and returns a tensor of the label's class
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
