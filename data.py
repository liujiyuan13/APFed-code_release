import os
import numpy as np
import math
import torch
import torchvision
import mat73
from torch.utils.data import DataLoader

from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Optional, Tuple

import utils.caltech101


class MatData(VisionDataset):

    def __init__(self,
                 root: str,
                 name: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        name = name + '_cla.mat'
        content = mat73.loadmat(os.path.join(root, name))

        if train:
            self.X = content['X_tr']
            self.Y = content['Y_tr']
        else:
            self.X = content['X_te']
            self.Y = content['Y_te']

        for i, x in enumerate(self.X):
            self.X[i] = self.X[i][0].T.astype(np.float32)
        self.Y = self.Y.astype(dtype=np.int64) - 1

        del content

    def __getitem__(self, idx) -> Tuple[Any, Any]:

        # data = np.concatenate([self.X[0][idx,:], self.X[1][idx,:]], axis=0)

        x = [None]*len(self.X)
        for i in range(len(self.X)):
            x[i] = self.transform(self.X[i][idx,:]) if self.transform else self.X[i][idx,:]

        y = self.target_transform(self.Y[idx]) if self.target_transform else self.Y[idx]

        return x, y


    def __len__(self) -> int:
        return self.Y.shape[0]


class MultiCrop(torch.nn.Module):
    def __init__(self, n_views, overlap=False, horizontal=True, valid_views=None):
        super().__init__()
        self.n_views = n_views
        self.overlap = overlap
        self.horizontal = horizontal
        self.valid_views = valid_views

    def forward(self, img):
        _, height, width = img.shape

        if self.overlap:
            n, step = self.n_views + 1, 2
        else:
            n, step = self.n_views, 1

        imgs = []
        if self.horizontal:
            sep = math.floor(height / n)
            for v in range(self.n_views):
                imgs.append(img[:, v * sep:(v + step) * sep, :])
        else:
            sep = math.floor(width / n)
            for v in range(self.n_views):
                imgs.append(img[:, :, v * sep:(v+step) * sep])

        if self.valid_views is None:
            return imgs
        else:
            return [imgs[i] for i in self.valid_views]


def get_loader(dataset, batch_size, n_views=2, overlap=False, horizontal=True, train=True, valid_views=None):
    if dataset == 'mnist':
        data_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./datasets/', train=train, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           MultiCrop(n_views=n_views, overlap=overlap, horizontal=horizontal, valid_views=valid_views)
                                       ])),
            batch_size=batch_size, shuffle=train)
    elif dataset == 'fmnist':
        data_loader = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST('./datasets/', train=train, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  MultiCrop(n_views=n_views, overlap=overlap, horizontal=horizontal, valid_views=valid_views)
                                              ])),
            batch_size=batch_size, shuffle=train)
    elif dataset == 'cifar10':
        data_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10('./datasets/', train=train, download=True,
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.ToTensor(),
                                             MultiCrop(n_views=n_views, overlap=overlap, horizontal=horizontal, valid_views=valid_views)
                                         ])),
            batch_size=batch_size, shuffle=train)
    elif dataset == 'cifar100':
        data_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100('./datasets/', train=train, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              MultiCrop(n_views=n_views, overlap=overlap, horizontal=horizontal, valid_views=valid_views)
                                          ])),
            batch_size=batch_size, shuffle=train)
    elif dataset == 'stl10':
        split = 'train' if train else 'test'
        data_loader = torch.utils.data.DataLoader(
            torchvision.datasets.STL10('./datasets/', split=split, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              MultiCrop(n_views=n_views, overlap=overlap, horizontal=horizontal, valid_views=valid_views)
                                          ])),
            batch_size=batch_size, shuffle=train)
    elif dataset == 'caltech101':
        if train:
            data_loader = torch.utils.data.DataLoader(
                utils.caltech101.Caltech101('./datasets/', split='train', download=False,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.Resize(32),
                                                torchvision.transforms.CenterCrop(32),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                 std=[0.229, 0.224, 0.225]),
                                                MultiCrop(n_views=n_views, overlap=overlap, horizontal=horizontal,
                                                          valid_views=valid_views)
                                            ])),
                batch_size=batch_size, shuffle=train)
        else:
            data_loader = torch.utils.data.DataLoader(
                utils.caltech101.Caltech101('./datasets/', split='test', download=False,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.Resize(32),
                                                torchvision.transforms.CenterCrop(32),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                 std=[0.229, 0.224, 0.225]),
                                                MultiCrop(n_views=n_views, overlap=overlap, horizontal=horizontal,
                                                          valid_views=valid_views)
                                            ])),
                batch_size=batch_size, shuffle=train)
    elif dataset == 'flower102':
        if train:
            data_loader = torch.utils.data.DataLoader(
                utils.flowers102.Flowers102('./datasets/', split='train', download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.Resize(156),
                                                    torchvision.transforms.RandomRotation(30),
                                                    torchvision.transforms.RandomResizedCrop(144),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                                     [0.229, 0.224, 0.225]),
                                                    MultiCrop(n_views=n_views, overlap=overlap, horizontal=horizontal, valid_views=valid_views)
                                                ])),
                batch_size=batch_size, shuffle=train)
        else:
            data_loader = torch.utils.data.DataLoader(
                # torchvision.datasets.Flowers102('./datasets/', split='test', download=True,
                utils.flowers102.Flowers102('./datasets/', split='test', download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.Resize(156),
                                                    torchvision.transforms.CenterCrop(144),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                                     [0.229, 0.224, 0.225]),
                                                    MultiCrop(n_views=n_views, overlap=overlap, horizontal=horizontal, valid_views=valid_views)
                                                ])),
                batch_size=batch_size, shuffle=train)
    elif dataset == 'winnipeg':
        data_loader = torch.utils.data.DataLoader(
            MatData('./datasets/', 'Winnipeg1', train=train), batch_size = batch_size, shuffle=train
        )
    elif dataset == 'ccv':
        data_loader = torch.utils.data.DataLoader(
            MatData('./datasets/', 'CCV', train=train), batch_size=batch_size, shuffle=train
        )
    else:
        raise NotImplementedError

    return data_loader