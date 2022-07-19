import copy
import numpy as np
import quadprog
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from torch.optim import Adam, SGD, RMSprop
from torch.utils.data import DataLoader
from .backbone import resnet18
from .base import Base


class Sequential(Base):
    def __init__(self, config):
        super(Sequential, self).__init__(config)

    def learn(self, task_id, dataset):
        loader = DataLoader(dataset,
                            batch_size=self.config.batch_size,
                            shuffle=True,
                            num_workers=2)
        self.opt = SGD(self.net.parameters(),
                       lr=self.config.lr,
                       momentum=self.config.momentum,
                       weight_decay=self.config.weight_decay)

        for i in range(self.config.n_epochs):
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.opt.zero_grad()
                pred = self.forward(x, task_id)
                loss = self.loss_fn(pred, y)
                loss.backward()
                self.opt.step()
