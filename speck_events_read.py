import torch
import torch.nn as nn
from sinabs.layers import IAF
from sinabs.backend.dynapcnn import DynapcnnNetwork
import matplotlib.pyplot as plt
from tonic.transforms import ToFrame
from sinabs.backend.dynapcnn.chip_factory import ChipFactory
from utils import config_modify_callback
from tonic.datasets.nmnist import NMNIST
from tonic.transforms import ToFrame
from sinabs.backend.dynapcnn.dynapcnn_visualizer import DynapcnnVisualizer
from sinabs.layers import IAFSqueeze
import numpy as np

batch_size = 4
num_workers = 4
device = "cpu"
chip = ChipFactory('speck2edevkit')

snn = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), bias=False),
    IAFSqueeze(min_v_mem=-1, spike_threshold=1, batch_size=1),
)

snn_quant = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), bias=False),
    IAFSqueeze(min_v_mem=-127, spike_threshold=127, batch_size=1),
)

input_shape = (1,1,1)

weight_value = 0.1
vmem_value = 0.5

weight_value_quant = 127
vmem_value_quant = 635

snn[0].weight.data = torch.ones_like(snn[0].weight.data) * weight_value
snn_quant[0].weight.data = torch.ones_like(snn_quant[0].weight.data) * weight_value_quant


_ = snn(torch.zeros(1,*input_shape))
snn[1].v_mem.data = torch.ones_like(snn[1].v_mem.data) * vmem_value
_ = snn_quant(torch.zeros(1,*input_shape))
snn_quant[1].v_mem.data = torch.ones_like(snn_quant[1].v_mem.data) * vmem_value_quant

speck = DynapcnnNetwork(snn=snn, input_shape=input_shape, discretize=True, dvs_input=True)
# devkit_name = "speck2edevkit:0"
# speck.to(device=devkit_name, chip_layers_ordering='auto', monitor_layers=[0], config_modifier=config_modify_callback)

print('sinabs quant net parameters ----')
print('weight data: ', snn_quant[0].weight.data.item())
print('vmem data: ', snn_quant[1].v_mem.data.item())

print('samna net parameters ----')
print('weight data: ', speck.sequence[1].conv_layer.weight.data.item())
print('vmem data: ', speck.sequence[1].spk_layer.v_mem.data.item())

# Get all attributes of the object
conv_layer = dir(speck.sequence[1].conv_layer)
spk_layer = dir(speck.sequence[1].spk_layer)

n_time_steps = 100
np.random.seed(0)
input_sinabs = torch.zeros((n_time_steps,1,1,1))
for i in range(n_time_steps):
    if np.random.rand() < 0.4:
        input_sinabs[i][0][0][0] = 1.0

output_sinabs = snn(input_sinabs)
output_sinabs_quant = snn_quant(input_sinabs)
output_sinabs = chip.raster_to_events(output_sinabs, layer=0)
output_sinabs_quant = chip.raster_to_events(output_sinabs_quant, layer=0)

input_samna = chip.raster_to_events(input_sinabs, layer=0)
# output_samna = speck(input_samna)

input_samna_time = [ev.timestamp for ev in input_samna]
# timestamps_samna  = [ev.timestamp for ev in output_samna]
timestamps_sinabs = [ev.timestamp for ev in output_sinabs]
timestamps_sinabs_quant = [ev.timestamp for ev in output_sinabs_quant]

# time_delay = sum(abs(np.array(timestamps_samna) - np.array(timestamps_sinabs))) / len(timestamps_sinabs)
# print(time_delay)

fig, ax = plt.subplots()
ax.eventplot(input_samna_time, label='Input Spikes', color='orange', lineoffsets=3)
ax.eventplot(timestamps_sinabs, label='Sinabs Spikes', color='b', lineoffsets=2)
ax.eventplot(timestamps_sinabs_quant, label='Sinabs Quant Spikes', color='r', lineoffsets=1)
# ax.eventplot(timestamps_samna, label='Samna Spikes', color='r', lineoffsets=1)

ax.set_yticks([1, 2, 3])
ax.set_yticklabels(['Samna Spikes', 'Sinabs Spikes', 'Input Spikes'], fontsize=15)
ax.set_xlabel('Time Steps [$\mu$s]', fontsize=12)
plt.show()