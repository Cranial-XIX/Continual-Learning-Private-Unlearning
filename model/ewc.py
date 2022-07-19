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


class EWC(Base):
    def __init__(self, config):
        super(EWC, self).__init__(config)
        self.logsoft = nn.LogSoftmax(dim=1)
        self.fish = {}
        self.checkpoint = {}

    def penalty(self):
        ### ewc penalty
        if len(self.prev_tasks) == 0:
            return torch.tensor(0.0).to(self.device)
        current_param = self.net.get_params()
        penalty = 0.0
        for t in self.prev_tasks:
            penalty += (self.fish[t] * (current_param - self.checkpoint[t]).pow(2)).sum()
        return penalty

    def learn(self, task_id, dataset):
        loader = DataLoader(dataset,
                            batch_size=self.config.batch_size,
                            shuffle=True,
                            num_workers=2)
        self.opt = SGD(self.net.parameters(),
                       lr=self.config.lr,
                       momentum=self.config.momentum,
                       weight_decay=self.config.weight_decay)

        ewc_lmbd  = self.config.ewc_lmbd
        self.n_iters = self.config.n_epochs * len(loader)

        for epoch in range(self.config.n_epochs):
            for i, (x, y) in enumerate(loader):
                x = x.to(self.device)
                y = y.to(self.device)

                loss = self.loss_fn(self.forward(x, task_id), y)

                # current loss
                if task_id > 0:
                    loss += ewc_lmbd * self.penalty()
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                #if task_id > 0:
                #    self.opt.zero_grad()
                #    Lr.backward()
                #    import pdb; pdb.set_trace()
                #    self.opt.step()

        ### end of task
        fish = torch.zeros_like(self.net.get_params())

        for j, (x, y) in enumerate(loader):
            x = x.to(self.device)
            y = y.to(self.device)
            for ex, lab in zip(x, y):
                self.opt.zero_grad()
                output = self.forward(ex.unsqueeze(0), task_id)
                loss = - F.nll_loss(self.logsoft(output), lab.unsqueeze(0), reduction='none')
                exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
                loss = torch.mean(loss)
                loss.backward()
                fish += exp_cond_prob * self.net.get_grads() ** 2

        fish /= (len(loader)*self.config.batch_size)
        self.prev_tasks.append(task_id)
        self.fish[task_id] = fish
        self.checkpoint[task_id] = self.net.get_params().data.clone()

    def forget(self, task_id):
        assert task_id in self.prev_tasks, f"[ERROR] {task_id} not seen before"
        self.prev_tasks.remove(task_id)
        del self.fish[task_id]
        del self.checkpoint[task_id]
