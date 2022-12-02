import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from typing import Callable, Optional
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

import yaml

''' yaml parser
'''
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class MNISTDataset(Dataset):
    def __init__(
        self, 
        raw_dir,
        transform: Optional[Callable] = None
    ) -> None:
        super().__init__()
        self.raw_dir = Path(raw_dir)
        self.image_list = sorted(self.raw_dir.glob("*.png"))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx) -> torch.Tensor:
        image_path = self.image_list[idx]
        image = torchvision.io.read_image(
                str(image_path), torchvision.io.image.ImageReadMode.RGB
                )
        if self.transform is not None:
            image = self.transform(image)

        return image.float() / 127.5 - 1.0 

class GaussianNoiseDataset(Dataset):

    def __init__(self, shape, length, mean=0, std=1) -> None:
        super().__init__()
        self.shape = shape
        self.length = length
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> torch.Tensor:
        return self.mean + self.std * torch.randn(self.shape)

if __name__ == '__main__':
    mnist_dir = Path('../data/hw3/mnist')
    dataset = MNISTDataset(mnist_dir, transform=T.Resize((32, 32), InterpolationMode.NEAREST))
    print(len(dataset)) #60000
    print(*dataset[0].shape) #torch.Size([3, 28, 28])
    print(dataset[0].dtype) #torch.float32
    print(dataset[0].max(), dataset[0].min()) #tensor(1.) tensor(-1.)
