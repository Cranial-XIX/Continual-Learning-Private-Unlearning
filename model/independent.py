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


class Independent(Base):
    def __init__(self, config):
        super(Independent, self).__init__(config)
        self.nets = {} # learn a separate network per task

    def forward(self, x, task): 
        if (task in self.nets):
            pred = self.nets[task].forward(x).view(x.shape[0], -1)
        else:
            pred = self.net.forward(x).view(x.shape[0], -1)

        if task > 0:
            pred[:, :self.cpt*task].data.fill_(-10e10)
        if task < self.n_tasks-1:
            pred[:, self.cpt*(task+1):].data.fill_(-10e10)
        return pred

    def learn(self, task_id, dataset):
        self.nets[task_id] = copy.deepcopy(self.net)

        loader = DataLoader(dataset,
                            batch_size=self.config.batch_size,
                            shuffle=True,
                            num_workers=2)

        opt = SGD(self.nets[task_id].parameters(),
                  lr=self.config.lr,
                  momentum=self.config.momentum,
                  weight_decay=self.config.weight_decay)

        for epoch in range(self.config.n_epochs):
            for i, (x, y) in enumerate(loader):
                x = x.to(self.device)
                y = y.to(self.device)

                # current loss
                h = self.forward(x, task_id)
                loss = self.loss_fn(h, y)
                opt.zero_grad()
                loss.backward()
                opt.step()

    def forget(self, task_id):
        del self.nets[task_id]
