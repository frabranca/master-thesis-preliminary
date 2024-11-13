from sinabs.activation import MembraneSubtract, MembraneReset
import sinabs.layers as sl
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt

''' This code shows the membrane activity for LIF and IAF neuron models '''

lif = sl.LIF(record_states=True, spike_threshold=0.1, tau_mem=0.6)
iaf = sl.IAF(record_states=True, spike_threshold=5.)

n_steps = 100
time = np.arange(0,n_steps,1)
input_ = torch.rand((1, n_steps, 1))
mask = torch.rand((1, n_steps, 1)) < 0.3
input_ = input_ * mask

output_iaf = iaf(input_)
output_lif = lif(input_)

print(torch.where(output_lif)[0])

vmem_iaf = iaf.recordings['v_mem'].detach().numpy().reshape(n_steps,)
vmem_lif = lif.recordings['v_mem'].detach().numpy().reshape(n_steps,)

# plt.figure(figsize=(7,7))
# plt.subplot(311)
# plt.ylabel('Input Spikes')
# plt.eventplot(torch.where(input_)[1])
# plt.grid()
# plt.xlim(0,n_steps)
# plt.subplot(312)
# plt.ylabel('Potential IAF')
# plt.plot(time, vmem_iaf)
# plt.grid()
# plt.xlim(0,n_steps)
# plt.subplot(313)
# plt.ylabel('Output Spikes')
# plt.xlabel('Time Steps')
# plt.eventplot(torch.where(output_iaf)[1])
# plt.grid()
# plt.xlim(0,n_steps)

plt.figure(figsize=(7,7))
plt.subplot(311)
plt.ylabel('Input Spikes')
plt.eventplot(torch.where(input_)[1])
plt.grid()
plt.xlim(0,n_steps)
plt.subplot(312)
plt.ylabel('Potential LIF')
plt.plot(time, vmem_lif)
plt.grid()
plt.xlim(0,n_steps)
plt.subplot(313)
plt.ylabel('Output Spikes')
plt.xlabel('Time Steps', fontfamily='serif')
plt.eventplot(torch.where(output_lif)[1])
plt.grid()
plt.xlim(0,n_steps)

plt.show()
