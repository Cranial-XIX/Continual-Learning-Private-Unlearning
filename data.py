import numpy as np
import pandas as pd
import torch

from config import Config
from sklearn import preprocessing
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, Dataset, random_split

# ==================
# Dataset Transforms
# ==================

_MNIST_TRAIN_TRANSFORMS = _MNIST_TEST_TRANSFORMS = [
    transforms.Pad(2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
]

_FASHION_MNIST_TRAIN_TRANSFORMS = _FASHION_MNIST_TEST_TRANSFORMS = [
    transforms.Pad(2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
]

_E_MNIST_TRAIN_TRANSFORMS = _E_MNIST_TEST_TRANSFORMS = [
    transforms.Pad(2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
]

_CIFAR100_TRAIN_TRANSFORMS = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
]


_CIFAR100_TEST_TRANSFORMS = [
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
]

_CIFAR10_TRAIN_TRANSFORMS = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))
]


_CIFAR10_TEST_TRANSFORMS = [
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))
]


class SubDataset(Dataset):
    '''To sub-sample a dataset, taking only those samples with label in [sub_labels].

    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units.'''

    def __init__(self, original_dataset, sub_labels, is_train, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indices = []
        for index in range(len(self.dataset)):
            if self.dataset.target_transform is None:
                label = self.dataset.targets[index]
            else:
                label = self.dataset.target_transform(self.dataset.targets[index])
            if label in sub_labels:
                self.sub_indices.append(index)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sub_indices)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indices[index]]
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)
        return sample


def split_data(config):
    T = config.n_tasks
    CPT = config.class_per_task
    C = T * CPT
    permutation = np.arange(C)

    data = {
        'mnist': datasets.MNIST,
        'fashion': datasets.FashionMNIST,
        'emnist': datasets.EMNIST,
        'cifar100': datasets.CIFAR100,
        'cifar10': datasets.CIFAR10,
    }
    train_transform = {
        'mnist': _MNIST_TRAIN_TRANSFORMS,
        'fashion': _FASHION_MNIST_TRAIN_TRANSFORMS,
        'emnist': _E_MNIST_TRAIN_TRANSFORMS,
        'cifar100': _CIFAR100_TRAIN_TRANSFORMS,
        'cifar10': _CIFAR10_TRAIN_TRANSFORMS,
    }
    test_transform = {
        'mnist': _MNIST_TEST_TRANSFORMS,
        'fashion': _FASHION_MNIST_TEST_TRANSFORMS,
        'emnist': _E_MNIST_TEST_TRANSFORMS,
        'cifar100': _CIFAR100_TEST_TRANSFORMS,
        'cifar10': _CIFAR10_TEST_TRANSFORMS,
    }

    train = data[config.dataset]('./data',
                                 train=True,
                                 download=True,
                                 transform=transforms.Compose(train_transform[config.dataset]))
    test  = data[config.dataset]('./data',
                                 train=False,
                                 download=True,
                                 transform=transforms.Compose(test_transform[config.dataset]))

    # generate labels-per-task
    labels_per_task = [list(np.array(range(CPT)) + CPT * task_id) for task_id in range(T)]

    SD = SubDataset
    # split them up into sub-tasks
    train_datasets = []
    test_datasets = []
    for labels in labels_per_task:
        target_transform = None
        #target_transform = transforms.Lambda(lambda y, x=labels[0]: y-x)
        train_datasets.append(SD(train, labels, True, target_transform=target_transform))
        test_datasets.append(SD(test, labels, False, target_transform=target_transform))

    user_request_sequence = [
        (0, "R"),
        (1, "T"),
        (2, "T"),
        (3, "R"),
        (1, "R"),
        (2, "F"),
        (4, "T"),
        (4, "F"),
    ]
    return train_datasets, test_datasets, user_request_sequence


class TransformedMNISTDataset(Dataset):
    def __init__(self, data):
        self.X = data[1]
        self.Y = data[2]
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


def get_transformed_mnist_dataset(config):
    if config.dataset == "rot_mnist":
        load_path = "./data/mnist_rotations.pt"
    elif config.dataset == "perm_mnist":
        load_path = "./data/mnist_permutations.pt"
    else:
        raise Exception(f"[ERROR] unknown dataset {config.dataset}")

    d_tr, d_te = torch.load(load_path)
    d_tr = d_tr[:config.n_tasks]
    d_te = d_te[:config.n_tasks]

    for t, (tr, te) in enumerate(zip(d_tr, d_te)):
        tr[2] = tr[2] + t * config.class_per_task
        te[2] = te[2] + t * config.class_per_task

    n_inputs = d_tr[0][1].size(1)
    n_outputs = 0
    for i in range(len(d_tr)):
        n_outputs = max(n_outputs, d_tr[i][2].max().item())
        n_outputs = max(n_outputs, d_te[i][2].max().item())
        
    train_datasets = [TransformedMNISTDataset(dataset) for dataset in d_tr]
    test_datasets = [TransformedMNISTDataset(dataset) for dataset in d_te]

    user_request_sequence = [
        (0, "R"),
        (1, "T"),
        (2, "T"),
        (3, "R"),
        (1, "R"),
        (2, "F"),
        (4, "T"),
        (4, "F"),
    ]
    return train_datasets, test_datasets, user_request_sequence


def get_cl_dataset(config):
    if config.dataset in ["rot_mnist", "perm_mnist"]:
        return get_transformed_mnist_dataset(config)
    else:
        return split_data(config)
