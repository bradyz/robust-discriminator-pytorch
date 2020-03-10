from itertools import repeat

import torch

from torchvision import transforms
from torchvision.datasets import ImageFolder


def repeater(loader):
    for loader in repeat(loader):
        for data in loader:
            yield data


def dataloader(data, batch_size, num_workers):
    return repeater(
            torch.utils.data.DataLoader(
                data, batch_size=batch_size, num_workers=num_workers,
                shuffle=True, drop_last=True, pin_memory=True))


def get_data(dataset_dir, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    return Wrap(ImageFolder(dataset_dir, transform=transform), batch_size, 1000, num_workers)


class Wrap(object):
    def __init__(self, data, batch_size, samples, num_workers):
        self.data = dataloader(data, batch_size, num_workers)
        self.samples = samples

    def __iter__(self):
        for i in range(self.samples):
            yield next(self.data)

    def __len__(self):
        return self.samples
