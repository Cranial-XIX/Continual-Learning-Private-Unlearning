import argparse
import numpy as np


class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.device = 'cpu'
        self.verbose = True
        self.init_args()

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def init_args(self):
        self.parser.add_argument('--method', default='er', help='[er, clu, retrain]')
        self.parser.add_argument('--seed', default=1, type=int, help='seed')
        self.parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'mnist', 'fashion', 'rot_mnist', 'perm_mnist', 'law'])
        self.parser.add_argument('--batch_size', default=32, type=int, help='batch size')
        self.parser.add_argument('--n_epochs', default=10, type=int, help='number of iterations')

        self.parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
        self.parser.add_argument('--momentum', default=0.0, type=float, help='momentum')
        self.parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight_decay')

        # EWC
        self.parser.add_argument('--ewc_lmbd', default=100., type=float, help='ewc lambda')

        # LSF
        self.parser.add_argument('--lsf_gamma', default=10.0, type=float, help='lsf gamma')

        # LWF
        self.parser.add_argument('--lwf_alpha', default=1.0, type=float, help='lsf gamma')
        self.parser.add_argument('--lwf_temp', default=2.0, type=float, help='lsf gamma')

        # ER
        self.parser.add_argument('--forget_iters', default=1000, type=int, help='number of forgetting iterations')
        self.parser.add_argument('--memorize_iters', default=1000, type=int, help='number of forgetting iterations')
        self.parser.add_argument('--mem_budget', default=200, type=int, help='memory budgeet')
        self.parser.add_argument('--mem_batch_size', default=32, type=int, help='memory batch size')

        # CLU_ER & CLPU_Derpp
        self.parser.add_argument('--use_pretrain', default=False, action='store_true', help='whether to initialize from previous model')

        # Derpp & CLPU_Derpp
        self.parser.add_argument('--alpha', default=0.5, type=float, help='memory batch size')
        self.parser.add_argument('--beta', default=1.0, type=float, help='memory batch size')

        args = self.parser.parse_args()
        dict_ = vars(args)

        for k, v in dict_.items():
            setattr(self, k, v)
        self.scenario = 'class'

        if self.dataset == 'cifar100':
            self.dim_input = (3, 32, 32)
            self.class_per_task = 20
            self.n_tasks = 5
        elif self.dataset == 'cifar10':
            self.dim_input = (3, 32, 32)
            self.class_per_task = 2
            self.n_tasks = 5
        elif self.dataset == 'mnist':
            self.dim_input = (1, 32, 32)
            self.class_per_task = 2
            self.n_tasks = 5
        elif self.dataset == 'fashion':
            self.dim_input = (1, 32, 32)
            self.class_per_task = 2
            self.n_tasks = 5
        elif self.dataset == 'rot_mnist':
            self.dim_input = (784,)
            self.class_per_task = 10
            self.n_tasks = 5
        elif self.dataset == 'perm_mnist':
            self.dim_input = (784,)
            self.class_per_task = 10
            self.n_tasks = 5
        elif self.dataset == 'law':
            self.scenario = 'domain'
            self.dim_input = (11,)
            self.class_per_task = 2
            self.n_tasks = 5

        if self.verbose:
            print("="*80)
            print("[INFO] -- Experiment Configs --")
            print("       1. data & task")
            print("          dataset:      %s" % self.dataset)
            print("          n_tasks:      %d" % self.n_tasks)
            print("          # class/task: %d" % self.class_per_task)
            print("       2. training")
            print("          lr:           %5.4f" % self.lr)
            print("       3. model")
            print("          method:       %s" % self.method)
            print("="*80)
