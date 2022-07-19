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
from .base import *


def smooth(logits, temp, dim):
    log = logits ** (1 / temp)
    return log / torch.sum(log, dim).unsqueeze(1)


def modified_kl_div(old, new):
    return -torch.mean(torch.sum(old * torch.log(new), 1))


class LwF(Base):
    def __init__(self, config):
        super(LwF, self).__init__(config)
        self.old_net = None
        self.soft = torch.nn.Softmax(dim=-1)

    def learn(self, task_id, dataset):
        self.old_net = copy.deepcopy(self.net)
        self.old_net.eval()
        for p in self.old_net.parameters():
            p.requires_grad = False

        loader = DataLoader(dataset,
                            batch_size=self.config.batch_size,
                            shuffle=True,
                            num_workers=2)
        self.opt = SGD(self.net.parameters(),
                       lr=self.config.lr,
                       momentum=self.config.momentum,
                       weight_decay=self.config.weight_decay)

        if task_id > 0:
            self.opt_cls = SGD(self.net.classifier.parameters(),
                           lr=self.config.lr,
                           momentum=self.config.momentum,
                           weight_decay=self.config.weight_decay)
            for epoch in range(self.config.n_epochs):
                for i, (x, y) in enumerate(loader):
                    x = x.to(self.device)
                    y = y.to(self.device)

                    loss = self.loss_fn(self.forward(x, task_id), y)
                    self.opt_cls.zero_grad()
                    loss.backward()
                    self.opt_cls.step()

        for epoch in range(self.config.n_epochs):
            for i, (x, y) in enumerate(loader):
                x = x.to(self.device)
                y = y.to(self.device)

                loss = self.loss_fn(self.forward(x, task_id), y)

                # current loss
                if task_id > 0:
                    n_prev_tasks = len(self.prev_tasks)
                    for t in self.prev_tasks:
                        outputs = self.forward(x, t)[...,t*self.cpt:(t+1)*self.cpt]
                        with torch.no_grad():
                            targets = self.old_net.forward(x)[..., t*self.cpt:(t+1)*self.cpt]
                        loss += self.config.lwf_alpha * modified_kl_div(
                            smooth(self.soft(targets), self.config.lwf_temp, 1),
                            smooth(self.soft(outputs), self.config.lwf_temp, 1)) / n_prev_tasks
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

        self.prev_tasks.append(task_id)

    def forget(self, task_id):
        self.prev_tasks.remove(task_id)
