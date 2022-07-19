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
from .derpp import Derpp, Replay


class CLPU_Derpp(Derpp):
    def __init__(self, config):
        super(CLPU_Derpp, self).__init__(config)
        self.side_nets = {}

    def forward(self, x, task): 
        if (task in self.side_nets):
            pred = self.side_nets[task].forward(x).view(x.shape[0], -1)
        else:
            pred = self.net.forward(x).view(x.shape[0], -1)

        if self.config.scenario != 'domain':
            if task > 0:
                pred[:, :self.cpt*task].data.fill_(-10e10)
            if task < self.n_tasks-1:
                pred[:, self.cpt*(task+1):].data.fill_(-10e10)
        return pred

    def learn(self, task_id, dataset):
        loader = DataLoader(dataset,
                            batch_size=self.config.batch_size,
                            shuffle=True,
                            num_workers=2)
        self.opt = SGD(self.net.parameters(),
                       lr=self.config.lr,
                       momentum=self.config.momentum,
                       weight_decay=self.config.weight_decay)

        exclude_list = [t for t in self.task_status.keys() if self.task_status[t] == 'T' ]

        self.n_iters = self.config.n_epochs * len(loader)
        for epoch in range(self.config.n_epochs):
            for i, (x, y) in enumerate(loader):
                x = x.to(self.device)
                y = y.to(self.device)

                # current loss
                h = self.forward(x, task_id)
                loss = self.loss_fn(h, y)

                R_prev_list = [t for t in self.prev_tasks if t not in exclude_list]
                n_prev_tasks = len(self.prev_tasks)
                for t in self.prev_tasks:
                    x_past, y_past, h_past = self.memory.sample_task(self.config.mem_batch_size//n_prev_tasks, t)
                    h_tmp = self.forward(x_past, t)
                    loss += self.alpha * self.der_loss(h_tmp, h_past, t) / n_prev_tasks
                    loss += self.beta * self.loss_fn(h_tmp, y_past) / n_prev_tasks

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                self.memory.add(x, y, h.detach(), task_id)

        self.prev_tasks.append(task_id)

    def temporarily_learn(self, task_id, dataset):
        # initialize a side network
        assert task_id not in self.side_nets, f"[ERROR] should not see {task_id} in side nets"

        if self.config.use_pretrain:
            self.side_nets[task_id] = copy.deepcopy(self.net)
        else:
            if "cifar" in self.config.dataset:
                if self.config.scenario != 'domain':
                    self.side_nets[task_id] = resnet18(self.cpt*self.n_tasks).to(self.config.device)
                else:
                    self.side_nets[task_id] = resnet18(self.cpt).to(self.config.device)
            else:
                self.side_nets[task_id] = MLP(self.config).to(self.config.device)


        loader = DataLoader(dataset,
                            batch_size=self.config.batch_size,
                            shuffle=True,
                            num_workers=2)

        opt = SGD(self.side_nets[task_id].parameters(),
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
                self.memory.add(x, y, h.detach(), task_id)

    def finally_learn(self, task_id):
        # use knowledge distillation to merge two networks
        self.opt = SGD(self.net.parameters(),
                       lr=self.config.lr,
                       momentum=self.config.momentum,
                       weight_decay=self.config.weight_decay)

        task_net = self.side_nets[task_id]
        task_net.eval()
        del self.side_nets[task_id]

        exclude_list = [t for t in self.task_status.keys() if self.task_status[t] == 'T' ]
        R_prev_list = [t for t in self.prev_tasks if t not in exclude_list]
        n_prev_tasks = len(R_prev_list)

        #for it in range(self.config.memorize_iters):
        for it in range(self.n_iters):
            loss = 0.0
            x_t, y_t, h_t = self.memory.sample_task(self.config.mem_batch_size, task_id)
            for tt in R_prev_list:
                x_past, y_past, h_past = self.memory.sample_task(self.config.mem_batch_size//n_prev_tasks, tt)
                h_tmp = self.forward(x_past, tt)
                loss += self.alpha * self.der_loss(h_tmp, h_past, tt) / n_prev_tasks
                loss += self.beta * self.loss_fn(h_tmp, y_past) / n_prev_tasks

            h_tmp_t = self.forward(x_t, task_id)
            #h_tmp_t_target = task_net.forward(x_t).detach()
            loss += self.beta * self.loss_fn(h_tmp_t, y_t) + self.alpha * self.der_loss(h_tmp_t, h_t, task_id)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        self.prev_tasks.append(task_id)

    def forget(self, task_id):
        self.memory.remove(task_id)
        del self.side_nets[task_id]
