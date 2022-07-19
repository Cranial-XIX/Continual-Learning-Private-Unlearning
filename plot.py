import glob
import matplotlib.pyplot as plt
import torch
from config import Config
import torch.nn.functional as F
import numpy as np
from scipy import stats as st


methods = {
    "perm-mnist": [
        "perm_mnist_sequential_lr0.01",
        "perm_mnist_independent_lr0.01",
        "perm_mnist_ewc_lr0.01_ewclmbd100.0",
        "perm_mnist_er_lr0.01_mem200",
        "perm_mnist_derpp_lr0.01_mem200_alpha0.5_beta1.0",
        "perm_mnist_clpu_derpp_lr0.01_mem200_alpha0.5_beta1.0_pretrainFalse",
        "perm_mnist_clpu_derpp_lr0.01_mem200_alpha0.5_beta1.0_pretrainTrue",
    ],

    "rot-mnist": [
        "rot_mnist_sequential_lr0.01",
        "rot_mnist_independent_lr0.01",
        "rot_mnist_ewc_lr0.01_ewclmbd100.0",
        "rot_mnist_er_lr0.01_mem200",
        "rot_mnist_derpp_lr0.01_mem200_alpha0.5_beta1.0",
        "rot_mnist_clpu_derpp_lr0.01_mem200_alpha0.5_beta1.0_pretrainFalse",
        "rot_mnist_clpu_derpp_lr0.01_mem200_alpha0.5_beta1.0_pretrainTrue",
    ],

    "split-cifar10": [
        "cifar10_sequential_lr0.01",
        "cifar10_independent_lr0.01",
        "cifar10_ewc_lr0.01_ewclmbd500.0",
        "cifar10_er_lr0.01_mem200",
        "cifar10_derpp_lr0.01_mem200_alpha0.5_beta0.5",
        "cifar10_clpu_derpp_lr0.01_mem200_alpha0.5_beta0.5_pretrainFalse",
        "cifar10_clpu_derpp_lr0.01_mem200_alpha0.5_beta0.5_pretrainTrue",
    ],

    "split-cifar100": [
        "cifar100_sequential_lr0.01",
        "cifar100_independent_lr0.01",
        "cifar100_ewc_lr0.01_ewclmbd1000.0",
        "cifar100_er_lr0.01_mem200",
        "cifar100_derpp_lr0.01_mem200_alpha0.5_beta1.0",
        "cifar100_clpu_derpp_lr0.01_mem200_alpha0.5_beta1.0_pretrainFalse",
        "cifar100_clpu_derpp_lr0.01_mem200_alpha0.5_beta1.0_pretrainTrue",
    ],
}

def get_log(name):

    seeds = [1,2,3,4,5]

    AA, FF = [], []
    FA = []

    IL = {}
    AL = {}
    mask = {}

    for seed in seeds:
        IL[seed] = {}
        AL[seed] = {}
        mask[seed] = {}

        res = torch.load(f"./results/{name}_seed{seed}.log")

        stats = res["stats"][0]
        user_requests = res["user_requests"]

        acc = stats["accuracy"] # (n_requests, n_tasks)
        Dr  = stats["Dr_mask"]  # (n_requests, n_tasks)
        Df  = stats["Df_mask"]  # (n_requests, n_tasks)
        logits = stats["logits"] # List (n_requests, n_data, class_per_task) * n_tasks

        a = (acc * Dr).sum(-1).sum(0) / Dr.sum()
        #print("="*80)
        #print(user_requests)
        #print(acc)
        #print(res["stats"][1]["accuracy"])

        task_set = set()
        first_time_acc = torch.zeros(acc.shape[-1])
        forget_idx = 0
        kl = 0.0

        forget_mask = torch.zeros_like(acc)

        for request_id, (task_id, learn_type, dr) in enumerate(user_requests):
            if (learn_type in ["R", "T"]) and (task_id not in task_set):
                first_time_acc[task_id] = acc[request_id, task_id]
                task_set.add(task_id)
            elif learn_type == "F":
                #forget_idx = min(1, forget_idx + 1)
                forget_idx = forget_idx + 1
                stats_f = res["stats"][1]#forget_idx]
                logits_ = stats_f["logits"]
                logits_ = [l[-1] for l in logits_] # List(n_data, class_per_task) * n_tasks

                IL[seed][forget_idx] = [l[request_id] for l in logits]
                AL[seed][forget_idx] = logits_
                mask[seed][forget_idx] = Df[request_id]

                #L[forget_idx]["in"].append(l[request_id] for l in logits])
                #L[forget_idx]["across"].append(logits_)
                #L[forget_idx]["mask"].append(Df[request_id])

                #for my_l, forget_l, mask in zip([l[request_id] for l in logits], logits_, Df[request_id]):
                #    if mask.item() > 0:
                #        try:
                #            kl += (F.softmax(my_l, -1) * (F.log_softmax(my_l, -1) - F.log_softmax(forget_l, -1))).sum(-1).mean()
                #        except:
                #            import pdb; pdb.set_trace()
                forget_mask[request_id][task_id] = 1.0

        #kl = kl / (forget_idx+1e-4)
        f = ((first_time_acc.view(1,-1) - acc) * Dr).sum(-1).sum(0) / Dr.sum()

        #fm = torch.Tensor(forget_mask).view(-1, 1)
        #fa = (acc * Df * fm).sum(-1).sum(0) / (Df * fm).sum()
        fa = (acc * forget_mask).sum() / forget_mask.sum()

        AA.append(a)
        FF.append(f)
        FA.append(fa)

    AA = torch.stack(AA)
    FF = torch.stack(FF)
    FA = torch.stack(FA)

    # calculate the KL divergence

    def js(a, b, m):
        #kl_ = 0.0
        js = 0.0
        cnt = 0
        for k in a.keys():
            cnt += 1
            for my_l, forget_l, mm in zip(a[k], b[k], m[k]):
                if mm.item() > 0:
                    p = F.softmax(my_l, -1)
                    q = F.softmax(forget_l, -1)
                    m = (p+q)/2
                    js += 0.5 * (p * (p.log() - m.log())).sum(-1).mean() + 0.5 * (q * (q.log() - m.log())).sum(-1).mean()
                    #kl_ += (F.softmax(my_l, -1) * (F.log_softmax(my_l, -1) - F.log_softmax(forget_l, -1))).sum(-1).mean()
        return js / cnt

    IGKL = []
    AGKL = []

    # in group
    for i in seeds:
        for j in seeds:
            if i == j:
                continue
            i1 = AL[i]
            i2 = AL[j]
            m  = mask[i]
            IGKL.append(js(i1, i2, m))

    # across group
    for i in seeds:
        for j in seeds:
            i1 = IL[i]
            i2 = AL[j]
            m = mask[i]
            AGKL.append(js(i1, i2, m))

    IGKL = torch.stack(IGKL)
    AGKL = torch.stack(AGKL)
    return AA, FF, FA, IGKL, AGKL

def get_method(x):
    if "sequential" in x:
        return "Seq"
    elif "lwf" in x:
        return "LwF"
    elif "ewc" in x and "lsf" not in x:
        return "EWC"
    elif "lsf" in x:
        return "LSF"
    elif "er" in x  and "derpp" not in x:
        return "ER"
    elif "derpp" in x and "clpu" not in x:
        return "DER++"
    elif "clpu" in x and "pretrainTrue" in x:
        return "CLPU-DER++"
    else:
        return "CLPU-DER++ (w/o pretraining)"

for data, method in methods.items():
    print("="*30 + f" {data} "+ "="*30)
    for m in method:
        try:
            AA, FF, FA, IGKL, AGKL = get_log(m)
        except:
            print('failed ', m)
            continue
        mmax = IGKL.numpy().max()
        rate = (AGKL < mmax).sum().item() / AGKL.shape[0]
        mm = get_method(m)
        print(f"{mm:50s} & {AA.mean()*100:4.2f} \\fs {{ {AA.std()*100:4.2f} }} & {FF.mean()*100:4.2f} \\fs {{ {FF.std()*100:4.2f} }} & {IGKL.mean():4.2f} \\fs {{ {IGKL.std():4.2f} }}  & {AGKL.mean():4.2f} \\fs {{ {AGKL.std():4.2f} }} & {abs(AGKL.mean()-IGKL.mean())/IGKL.mean():4.2f} & {rate:4.2f} \\\\") 
