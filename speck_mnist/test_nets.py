from tonic.datasets.nmnist import NMNIST
from torch import nn
from tonic.transforms import ToFrame
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from tqdm.notebook import tqdm
from torch.nn import CrossEntropyLoss
from sinabs.from_torch import from_model
import os
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import Subset
import samna
from sinabs.backend.dynapcnn import DynapcnnNetwork

# define a CNN model
cnn = nn.Sequential(
    # [2, 34, 34] -> [8, 17, 17]
    nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(3, 3), padding=(1, 1), bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),
    # [8, 17, 17] -> [16, 8, 8]
    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=(1, 1), bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),
    # [16 * 8 * 8] -> [16, 4, 4]
    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2),  bias=False),
    nn.ReLU(),
    # [16 * 4 * 4] -> [10]
    nn.Flatten(),
    nn.Linear(16 * 4 * 4, 10, bias=False),
    nn.ReLU(),
)

batch_size = 4
num_workers = 4
device = 'cpu'

cnn.load_state_dict(torch.load('my_cnn_model.pth'))

snn_convert = from_model(model=cnn, input_shape=(2, 34, 34), batch_size=batch_size).spiking_model

# define a transform that accumulate the events into a raster-like tensor
root_dir = './NMNIST'
n_time_steps = 100
to_raster = ToFrame(sensor_size=NMNIST.sensor_size, n_time_bins=n_time_steps)
snn_test_dataset = NMNIST(save_to=root_dir, train=False, transform=to_raster)
snn_test_dataloader = DataLoader(snn_test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=False)

snn_convert = snn_convert.to(device)

# sequence of 100 steps to form 2 channel image of 34x34 pixels

snn_test_dataset = NMNIST(save_to=root_dir, train=False)
# for time-saving, we only select a subset for on-chip infernce， here we select 1/100 for an example run
subset_indices = list(range(0, len(snn_test_dataset), 100))
snn_test_dataset = Subset(snn_test_dataset, subset_indices)

inferece_p_bar = next(iter(tqdm(snn_test_dataset)))

data, label = inferece_p_bar
print(data)
