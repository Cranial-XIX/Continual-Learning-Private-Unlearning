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
from .backbone import resnet18, MLP
from .base import Base
from .er import Replay, ER


class CLU_ER(ER):
    def __init__(self, config):
        super(CLU_ER, self).__init__(config)
        self.side_nets = {}

    def forward(self, x, task): 
        if (task in self.side_nets):
            pred = self.side_nets[task].forward(x).view(x.shape[0], -1)
        else:
            pred = self.net.forward(x).view(x.shape[0], -1)

        if task > 0:
            pred[:, :self.cpt*task].data.fill_(-10e10)
        if task < self.n_tasks-1:
            pred[:, self.cpt*(task+1):].data.fill_(-10e10)
        return pred

    def distill_loss(self, scores, target_scores, task_id, T=2.):
        cpt = self.cpt
        log_scores_norm = F.log_softmax(scores / T, dim=1)[:, task_id*cpt:(task_id+1)*cpt]
        targets_norm = F.softmax(target_scores / T, dim=1)[:, task_id*cpt:(task_id+1)*cpt]
        # calculate distillation loss (see e.g., Li and Hoiem, 2017)
        loss = -(targets_norm * log_scores_norm).sum(1).mean() * T**2
        return loss

    def learn(self, task_id, dataset):
        loader = DataLoader(dataset,
                            batch_size=self.config.batch_size,
                            shuffle=False,
                            num_workers=2)

        self.opt = SGD(self.net.parameters(),
                       lr=self.config.lr,
                       momentum=self.config.momentum,
                       weight_decay=self.config.weight_decay)

        #exclude_list = [t if self.task_status[t] == 'T' for t in self.task_status.keys()]
        exclude_list = [t for t in self.task_status.keys() if self.task_status[t] == 'T' ]

        for epoch in range(self.config.n_epochs):
            for i, (x, y) in enumerate(loader):
                x = x.to(self.device)
                y = y.to(self.device)

                #x_past, y_past = self.memory.sample(self.config.mem_batch_size, exclude_list)
                #if x_past is None:
                #    x_ = x; y_ = y;
                #else:
                #    x_ = torch.cat((x, x_past))
                #    y_ = torch.cat((y, y_past))

                # current loss
                #loss = self.loss_fn(self.forward(x_), y_)
                #self.opt.zero_grad()
                #loss.backward()
                #self.opt.step()
                #self.memory.add(x, y, task_id)

                loss = self.loss_fn(self.forward(x, task_id), y)

                R_prev_list = [t for t in self.prev_tasks if t not in exclude_list]
                n_prev_tasks = len(R_prev_list)
                for t in R_prev_list:
                    x_past, y_past = self.memory.sample_task(self.config.mem_batch_size//n_prev_tasks, t)
                    loss += self.loss_fn(self.forward(x_past, t), y_past) / n_prev_tasks

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                self.memory.add(x, y, task_id)

        self.prev_tasks.append(task_id)

    def temporarily_learn(self, task_id, dataset):
        # initialize a side network
        assert task_id not in self.side_nets, f"[ERROR] should not see {task_id} in side nets"

        if self.config.use_pretrain:
            self.side_nets[task_id] = copy.deepcopy(self.net)
        else:
            if "cifar" in self.config.dataset:
                self.side_nets[task_id] = resnet18(self.cpt*self.n_tasks).to(self.config.device)
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
                loss = self.loss_fn(self.forward(x, task_id), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                self.memory.add(x, y, task_id)

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

        for it in range(self.config.memorize_iters):
            loss = 0.0
            x_t, y_t = self.memory.sample_task(self.config.mem_batch_size, task_id)
            for tt in R_prev_list:
                x_past, y_past = self.memory.sample_task(self.config.mem_batch_size//n_prev_tasks, tt)
                loss += self.loss_fn(self.forward(x_past, tt), y_past) / n_prev_tasks
            loss += self.distill_loss(self.forward(x_t, task_id), task_net.forward(x_t).detach(), task_id)
            loss += self.loss_fn(self.forward(x_t, task_id), y_t)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        self.prev_tasks.append(task_id)

    def forget(self, task_id):
        self.memory.remove(task_id)
        del self.side_nets[task_id]
