#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

# After threads are joined, this util function calculate L2 norm for each update
def calculate_l2_norm(w_locals):
    norms = []
    for model in w_locals:
        # Flatten model parameters into a single tensor
        flat_params = torch.cat([param.flatten().float().cpu() for param in model.values()])
        # Calculate the L2 norm of the flattened model update
        l2_norm = torch.norm(flat_params, p=2).item()
        norms.append(l2_norm)
    return norms