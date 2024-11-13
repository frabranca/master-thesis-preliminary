import torch
import torch.nn as nn

import sinabs.activation
import sinabs.layers as sl

model = nn.Sequential(
    nn.Linear(16, 64),
    sl.LIF(
        tau_mem=10.0,
        surrogate_grad_fn=sinabs.activation.SingleExponential()
    ),
    nn.Linear(64, 4),
    sl.LIF(
        tau_mem=10.0,
        surrogate_grad_fn=sinabs.activation.SingleExponential()
    ),
)