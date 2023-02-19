from comet_ml import Experiment
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision as tv
import torchvision.transforms as tr
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import argparse
from tqdm import tqdm
import time
import json
import pandas as pd
import numpy as np
import os

from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
import itertools
import string
from pathlib import Path

import matplotlib.pyplot as plt



def compute_precisions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    src_lengths: Optional[torch.Tensor] = None,
    minsep: int = 6,
    maxsep: Optional[int] = None,
    name: Optional[str] = None,
    count: Optional[str] = None,
    slen: Optional[int] = None,
    override_length: Optional[int] = None,  # for casp
):
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)
    if targets.dim() == 2:
        targets = targets.unsqueeze(0)
    override_length = (targets[0, 0] >= 0).sum()

    # Check sizes
    if predictions.size() != targets.size():
        raise ValueError(
            f"Size mismatch. Received predictions of size {predictions.size()}, "
            f"targets of size {targets.size()}"
        )
    device = predictions.device
    
    # Elements for plot
    x, y = np.nonzero(targets.squeeze().cpu().numpy()) #extract ones
    c = np.full_like(x.astype(str), 'tab:gray')
    a = np.full_like(x.astype(float), 0.5)
    
    batch_size, seqlen, _ = predictions.size()
    seqlen_range = torch.arange(seqlen, device=device)

    sep = seqlen_range.unsqueeze(0) - seqlen_range.unsqueeze(1)
    sep = sep.unsqueeze(0)
    valid_mask = sep >= minsep
    valid_mask = valid_mask & (targets >= 0)  # negative targets are invalid

    if maxsep is not None:
        valid_mask &= sep < maxsep

    if src_lengths is not None:
        valid = seqlen_range.unsqueeze(0) < src_lengths.unsqueeze(1)
        valid_mask &= valid.unsqueeze(1) & valid.unsqueeze(2)
    else:
        tmp = seqlen if int(slen) == 256 else int(slen)
        src_lengths = torch.full([batch_size], tmp, device=device, dtype=torch.long)

    predictions = predictions.masked_fill(~valid_mask, float("-inf"))

    x_ind, y_ind = np.triu_indices(seqlen, minsep)
    predictions_upper = predictions[:, x_ind, y_ind]
    targets_upper = targets[:, x_ind, y_ind]

    topk = seqlen if int(slen) == 256 else int(slen)
    indices = predictions_upper.argsort(dim=-1, descending=True)[:, :topk]
     
    # Elements for plot
    l = indices[0, :int(topk/5)].cpu()
    al = np.append(a, np.ones(l.size(0)))
    a = np.append(a, np.ones(indices.size(1)))
    xl = np.append(x, x_ind[l])
    x = np.append(x, x_ind[indices.cpu()])
    yl = np.append(y, y_ind[l])
    y = np.append(y, y_ind[indices.cpu()])
    cl = np.append(c, targets_upper[0, l].cpu().numpy().astype(str))
    c = np.append(c, targets_upper[0, indices.cpu()].cpu().numpy().astype(str))
    c[(c == '1.0') | (c == '1')] = 'tab:blue'
    c[(c == '0.0') | (c == '0')] = 'tab:red'
    cl[(cl == '1.0') | (cl == '1')] = 'tab:blue'
    cl[(cl == '0.0') | (cl == '0')] = 'tab:red'
    
    f1 = plt.figure()
    f2 = plt.figure()
    
    ax1 = f1.add_subplot()
    ax1.scatter(x, y, s=5, c=c, alpha=a)
    ax1.grid(True, which='both')
    ax1.set_box_aspect(1)
    f1.savefig('img/' + str(count) + '_' + str(name) + '_L.png')
    
    ax2 = f2.add_subplot()
    ax2.scatter(xl, yl, s=5, c=cl, alpha=al)
    ax2.grid(True, which='both')
    ax2.set_box_aspect(1)
    f2.savefig('img/' + str(count) + '_' + str(name) + '_L5.png')
    
    plt.show()
    
    topk_targets = targets_upper[torch.arange(batch_size).unsqueeze(1), indices]
    if topk_targets.size(1) < topk:
        topk_targets = F.pad(topk_targets, [0, topk - topk_targets.size(1)])

    cumulative_dist = topk_targets.type_as(predictions).cumsum(-1)

    gather_lengths = src_lengths.unsqueeze(1)

    gather_indices = (
        torch.arange(0.1, 1.1, 0.1, device=device).unsqueeze(0) * gather_lengths
    ).type(torch.long) - 1

    binned_cumulative_dist = cumulative_dist.gather(1, gather_indices)
    binned_precisions = binned_cumulative_dist / (gather_indices + 1).type_as(
        binned_cumulative_dist
    )

    pl5 = binned_precisions[:, 1].mean()
    pl2 = binned_precisions[:, 4].mean()
    pl = binned_precisions[:, 9].mean()

    return {"L": pl, "L/2": pl2, "L/5": pl5}


def precision(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    slen: Optional[int] = None,
    count: Optional[str] = None,
) -> Dict[str, float]:
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    contact_ranges = [
        #("local", 3, 6),
        #("short", 6, 12),
        ("MLR", 12, None),
        ("LR", 24, None)
    ]
    metrics = {}
    targets = targets.to(predictions.device)
    for name, minsep, maxsep in contact_ranges:
        rangemetrics = compute_precisions(
            predictions,
            targets,
            minsep=minsep,
            maxsep=maxsep,
            name=name, #name of contact
            count=count, #name of protein
            slen=slen #sequence lenght
        )
        for key, val in rangemetrics.items():
            metrics[f"{name}_{key}"] = val.item()
    return metrics

