import torch.utils.data as torchdata


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
