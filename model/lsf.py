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
from .lwf import smooth, modified_kl_div


class LSF(Base):
    def __init__(self, config):
        super(LSF, self).__init__(config)
        self.dim_input = config.dim_input
        self.mnemonic_code = torch.randn(
            config.n_tasks*config.class_per_task,
            *config.dim_input
        ).to(self.device) # all mnemonic codes

        self.mnemonic_target = torch.arange(
            config.n_tasks*config.class_per_task
        ).to(self.device)
        if self.config.scenario == 'domain':
            self.mnemonic_target = self.mnemonic_target % config.class_per_task

        self.logsoft = nn.LogSoftmax(dim=1)
        self.fish = {}
        self.checkpoint = {}

        self.old_net = None
        self.soft = torch.nn.Softmax(dim=-1)
        self.prev_dataset = None

    def penalty(self):
        ### ewc penalty
        if len(self.prev_tasks) == 0:
            return torch.tensor(0.0).to(self.device)
        current_param = self.net.get_params()
        penalty = 0.0
        for t in self.prev_tasks:
            penalty += (self.fish[t] * (current_param - self.checkpoint[t]).pow(2)).sum()
        return penalty

    def learn(self, task_id, dataset, is_forget=False):
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

        lsf_gamma = self.config.lsf_gamma
        ewc_lmbd  = self.config.ewc_lmbd

        self.n_iters = self.config.n_epochs * len(loader)

        if not is_forget:
            self.prev_dataset = (task_id, dataset)

        for epoch in range(self.config.n_epochs):
            for i, (x, y) in enumerate(loader):
                x = x.to(self.device)
                y = y.to(self.device)

                target_shape = [x.shape[0]] + [1] * (len(x.shape) - 1)
                lsf_lmbd = torch.rand(*target_shape).to(x.device)
                y_idx = y
                hat_x = lsf_lmbd * x + (1-lsf_lmbd) * self.mnemonic_code[y_idx]

                x_ = torch.cat([x, hat_x], 0)
                y_ = torch.cat([y, y], 0)
                loss = self.loss_fn(self.forward(x_, task_id), y_)

                if task_id > 0:
                    loss += ewc_lmbd * self.penalty()

                    n_prev_tasks = len(self.prev_tasks)
                    for t in self.prev_tasks:
                        loss += lsf_gamma * self.loss_fn(
                                self.forward(
                                    self.mnemonic_code[t*self.cpt:(t+1)*self.cpt].view(
                                        -1, *self.dim_input), 
                                    t),
                                self.mnemonic_target[t*self.cpt:(t+1)*self.cpt]
                        ) / n_prev_tasks

                        # lwf
                        outputs = self.forward(x, t)[...,t*self.cpt:(t+1)*self.cpt]
                        with torch.no_grad():
                            targets = self.old_net.forward(x)[..., t*self.cpt:(t+1)*self.cpt]
                        loss += self.config.lwf_alpha * modified_kl_div(
                            smooth(self.soft(targets), 2, 1),
                            smooth(self.soft(outputs), 2, 1)) / n_prev_tasks

                # current loss
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

        ### end of task
        if not is_forget:
            fish = torch.zeros_like(self.net.get_params())

            for j, (x, y) in enumerate(loader):
                x = x.to(self.device)
                y = y.to(self.device)
                for ex, lab in zip(x, y):
                    self.opt.zero_grad()
                    output = self.net(ex.unsqueeze(0))
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
        cpt = self.cpt
        self.prev_tasks.remove(task_id)
        del self.fish[task_id]
        del self.checkpoint[task_id]
        self.learn(self.prev_dataset[0], self.prev_dataset[1], is_forget=True)
        #self.opt = SGD(self.net.parameters(),
        #               lr=self.config.lr,
        #               momentum=self.config.momentum,
        #               weight_decay=self.config.weight_decay)

        #lsf_gamma = self.config.lsf_gamma
        #ewc_lmbd  = self.config.ewc_lmbd

        #for it in range(self.n_iters):
        #    loss = ewc_lmbd * self.penalty()
        #    n_prev_tasks = len(self.prev_tasks)
        #    for t in self.prev_tasks:
        #        loss += lsf_gamma * self.loss_fn(
        #                self.forward(
        #                    self.mnemonic_code[t*self.cpt:(t+1)*self.cpt].view(
        #                        -1, *self.dim_input), t),
        #                self.mnemonic_target[t*self.cpt:(t+1)*self.cpt]
        #        ) / n_prev_tasks

        #        # lwf
        #        outputs = self.forward(x, t)[...,t*self.cpt:(t+1)*self.cpt]
        #        with torch.no_grad():
        #            targets = self.old_net.forward(x)[..., t*self.cpt:(t+1)*self.cpt]
        #        loss += self.config.lwf_alpha * modified_kl_div(
        #            smooth(self.soft(targets), 2, 1),
        #            smooth(self.soft(outputs), 2, 1)) / n_prev_tasks

        #    # current loss
        #    self.opt.zero_grad()
        #    loss.backward()
        #    self.opt.step()
