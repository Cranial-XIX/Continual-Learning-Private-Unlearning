import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from data import *
from model import *
from torch.utils.data import DataLoader, ConcatDataset


def check_path(path):
    if not os.path.exists(path):
        print("[INFO] making folder %s" % path)
        os.makedirs(path)


def evaluate(testsets, config, model, Dr, Df):
    model.eval()
    n_tasks = config.n_tasks
    L = torch.zeros(n_tasks)
    A = torch.zeros(n_tasks)
    logits = []

    max_task = -1
    cpt = config.class_per_task
    with torch.no_grad():

        for task, dataset in enumerate(testsets):
            max_task = max(max_task, task+1)

            bch = config.batch_size
            loader = DataLoader(dataset, batch_size=bch, shuffle=False)

            l = a = n = 0.0
            logit_ = torch.zeros(len(dataset), cpt)

            for i, (x, y) in enumerate(loader):
                y_tensor = y.to(config.device)
                x_tensor = x.to(config.device)
                y_ = model(x_tensor, task)

                l += F.cross_entropy(y_, y_tensor, reduction='sum').item()
                a += y_.argmax(-1).eq(y_tensor).float().sum().item()
                logit_[i*bch:i*bch+y_tensor.shape[0]].copy_(
                        y_[...,cpt*task:cpt*(task+1)].cpu())
                n += y_tensor.shape[0]

            L[task] = l / n
            A[task] = a / n
            logits.append(logit_)

    model.train()
    print("[INFO] loss: ", L[:max_task])
    print("[INFO] acc.: ", A[:max_task])
    return {
        'loss': L,
        'accuracy': A,
        'logits': logits,
    }


def get_continual_learning_unlearning_dataset(config):
    train_datasets, test_datasets, user_request_sequence = get_cl_dataset(config) # list(n_tasks * dataset), list(n_tasks * datasets)
    #user_request_sequence = [
    #    (0, "R"),
    #    (1, "T"),
    #    (2, "T"),
    #    (3, "R"),
    #    (1, "R"),
    #    (2, "F"),

    #    #(0, "R"),
    #    #(1, "T"),
    #    #(1, "F"),

    #    #(0, "R"),
    #    #(1, "T"),
    #    #(1, "R"),
    #]

    def clear_all_forget_request(li):
        remove_list = []
        for request_id, (task_id, learn_type, dr_list) in enumerate(li):
            if learn_type == "F":
                remove_list.append(request_id)
                for j in range(request_id):
                    if li[j][0] == task_id and li[j][1] == "T":
                        remove_list.append(j)
                        break
        new_list = []
        Dr_list = []
        for request_id, (task_id, learn_type, dr_list) in enumerate(li):
            if request_id not in remove_list:
                if (learn_type in ["R", "T"]) and (task_id not in Dr_list):
                    Dr_list.append(task_id)
                new_list.append((task_id, learn_type, list(Dr_list)))
        return new_list

    user_request_sequence_with_Dr = []
    Dr_list = []
    for task_id, learn_type in user_request_sequence:
        if (learn_type in ["R", "T"]) and (task_id not in Dr_list):
            Dr_list.append(task_id)
        elif learn_type == "F":
            Dr_list.remove(task_id)
        user_request_sequence_with_Dr.append((task_id, learn_type, list(Dr_list)))

    forget_learn_list = []
    for request_id, (task_id, learn_type, dr_list) in enumerate(user_request_sequence_with_Dr):
        if learn_type == "F":
            list_upto = list(user_request_sequence_with_Dr[:request_id+1])
            forget_learn_list.append(clear_all_forget_request(list_upto))
    print(user_request_sequence_with_Dr)
    print(forget_learn_list)
    #forget_learn_list = [list(user_request_sequence_with_Dr)]
    return train_datasets, test_datasets, user_request_sequence_with_Dr, forget_learn_list


def clu_train(config, model, train_datasets, test_datasets, user_request_sequence_with_Dr):

    n_tasks = config.n_tasks
    Df = []

    loss     = torch.zeros(len(user_request_sequence_with_Dr), n_tasks)
    accuracy = torch.zeros(len(user_request_sequence_with_Dr), n_tasks)
    times    = torch.zeros(len(user_request_sequence_with_Dr))
    Df_mask  = torch.zeros(len(user_request_sequence_with_Dr), n_tasks)
    Dr_mask  = torch.zeros(len(user_request_sequence_with_Dr), n_tasks)
    logits   = [torch.zeros(len(user_request_sequence_with_Dr), 
                            len(ds), config.class_per_task) for ds in test_datasets]

    for request_id, (task_id, learn_type, Dr) in enumerate(user_request_sequence_with_Dr):
        if config.verbose:
            print('='*80)
            learn_type_str = {
                "R": "Learning",
                "T": "Temporarily learning",
                "F": "Forgetting",
            }[learn_type]
            print(f'[INFO] {learn_type_str} Task {task_id} ...')

        if learn_type == "F": # forget
            Df.append(task_id)

        # learn
        t0 = time.time()
        model.continual_learn_and_unlearn(task_id, train_datasets[task_id], learn_type)
        t1 = time.time()

        # evaluate
        for df in Df:
            Df_mask[request_id][df] = 1.
        for dr in Dr:
            Dr_mask[request_id][dr] = 1.
        stat = evaluate(test_datasets, config, model, Dr, Df)
        loss[request_id] = stat['loss']
        accuracy[request_id] = stat['accuracy']
        times[request_id] = t1-t0
        for t in range(n_tasks):
            #if stat['logits'][t] is not None:
            logits[t][request_id] = stat['logits'][t]

    return {
        'loss': loss,
        'accuracy': accuracy,
        'times': times,
        'Df_mask': Df_mask,
        'Dr_mask': Dr_mask,
        'logits': logits
    }


def run(config):
    train_datasets, test_datasets, user_request_sequence_with_Dr, forget_learn_list = get_continual_learning_unlearning_dataset(config)
    print("[INFO] finish processing data")

    fn_map = {
        "independent": Independent,
        "sequential": Sequential,
        "ewc": EWC,
        "er" : ER,
        "lsf": LSF,
        "lwf": LwF,
        "clu_er": CLU_ER,
        "derpp": Derpp,
        "clpu_derpp": CLPU_Derpp,
    }
    model = fn_map[config.method](config).to(config.device)
    sd = model.state_dict()
    check_path('./results')

    if config.method == "ewc":
        exp_name = f"{config.dataset}_{config.method}_lr{config.lr}_ewclmbd{config.ewc_lmbd}_seed{config.seed}"
    elif config.method == "lwf":
        exp_name = f"{config.dataset}_{config.method}_lr{config.lr}_lwfalpha{config.lwf_alpha}_lwftemp{config.lwf_temp}_seed{config.seed}"
    elif config.method == "lsf":
        exp_name = f"{config.dataset}_new{config.method}_lr{config.lr}_lsfgamma{config.lsf_gamma}_ewclmbd{config.ewc_lmbd}_seed{config.seed}"
    elif config.method == "er":
        exp_name = f"{config.dataset}_{config.method}_lr{config.lr}_mem{config.mem_budget}_seed{config.seed}"
    elif config.method == "clu_er":
        exp_name = f"{config.dataset}_{config.method}_lr{config.lr}_mem{config.mem_budget}_pretrain{config.use_pretrain}_seed{config.seed}"
    elif config.method == "derpp":
        exp_name = f"{config.dataset}_{config.method}_lr{config.lr}_mem{config.mem_budget}_alpha{config.alpha}_beta{config.beta}_seed{config.seed}"
    elif config.method == "clpu_derpp":
        exp_name = f"{config.dataset}_{config.method}_lr{config.lr}_mem{config.mem_budget}_alpha{config.alpha}_beta{config.beta}_pretrain{config.use_pretrain}_seed{config.seed}"
    else:
        exp_name = f"{config.dataset}_{config.method}_lr{config.lr}_seed{config.seed}"

    stats = []
    #for urs in [user_request_sequence_with_Dr] + forget_learn_list:
    for urs in [user_request_sequence_with_Dr] + [forget_learn_list[0]]:
        print("[INFO] processing user's requests:")
        print(urs)
        model = fn_map[config.method](config).to(config.device)
        model.load_state_dict(sd)
        stat = clu_train(config, model, train_datasets, test_datasets, urs)
        stats.append(stat)

    result = {
        'stats': stats,
        'user_requests': user_request_sequence_with_Dr,
        'forget_learn_list': forget_learn_list,
    }
    torch.save(result, f"./results/{exp_name}.log")


if __name__ == "__main__":
    config = Config()

    # control seed
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

    run(config)
