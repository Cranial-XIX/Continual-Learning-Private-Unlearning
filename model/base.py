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
from .backbone import resnet18, MLP


class Base(nn.Module):
    def __init__(self, config):
        super(Base, self).__init__()
        self.config = config
        self.device = config.device
        self.n_tasks = config.n_tasks
        self.cpt = config.class_per_task
        if "cifar" in config.dataset:
            if config.scenario == 'domain':
                self.net = resnet18(config.class_per_task)
            else:
                self.net = resnet18(config.class_per_task*config.n_tasks)
        else:
            self.net = MLP(config)

        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn_reduction_none = nn.CrossEntropyLoss(reduction='none')
        self.opt = None
        self.task_status = {}
        self.prev_tasks = []
        self.n_iters = 1

    def forward(self, x, task):
        x = self.net(x)
        x = x.view(x.shape[0], -1)

        if self.config.scenario != 'domain':
            if task > 0:
                x[:, :self.cpt*task].data.fill_(-10e10)
            if task < self.n_tasks-1:
                x[:, self.cpt*(task+1):].data.fill_(-10e10)
        return x

    def learn(self, task_id, dataset):
        pass

    def temporarily_learn(self, task_id, dataset):
        return self.learn(task_id, dataset) # default to same as learn

    def finally_learn(self, task_id):
        return # default: do nothing when we finally learn a task

    def forget(self, task_id):
        return # default: do nothing when we want to forget a task

    def continual_learn_and_unlearn(self, task_id, dataset, learn_type):
        if learn_type in ["R", "T"]:
            if task_id not in self.task_status: # first time learn the task
                self.task_status[task_id] = learn_type
                if learn_type == "R":
                    self.learn(task_id, dataset)
                else:
                    self.temporarily_learn(task_id, dataset)
            else: # second time consolidate
                assert learn_type == "R", "[ERROR] second time learn task should be memorize"
                assert self.task_status[task_id] == "T", "[ERROR] the task should have been temporarily learned"
                self.finally_learn(task_id)
                self.task_status[task_id] = "R"
        else: # learn type is "F" forget
            assert learn_type == "F", f"[ERROR] unknown learning type {learn_type}"
            assert task_id in self.task_status, f"[ERROR] {task_id} is not learned"
            assert self.task_status[task_id] == "T", f"[ERROR] {task_id} was remembered, cannot unlearn"
            self.forget(task_id)
            self.task_status[task_id] = "F"
