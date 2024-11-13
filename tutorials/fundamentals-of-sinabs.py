import torch
import sinabs.layers as sl
import matplotlib.pyplot as plt


# Define a neuron in 'SINABS'
neuron = sl.IAF(spike_threshold=4.)
t_sim = 100
# Set a constant bias current to excite the neuron
input_current = (torch.rand((1, t_sim, 1)) < 0.5).float()

# Membrane potential trace
vmem = [0]

# Output spike raster
spike_train = []
for t in range(t_sim):
    with torch.no_grad():
        out = neuron(input_current[:, t : t + 1])
        # Check if there is a spike
        if (out != 0).any():
            spike_train.append(t)
        # Record membrane potential
        vmem.append(neuron.v_mem[0, 0])


# Plot membrane potential
plt.figure()
plt.plot(range(t_sim + 1), vmem)
plt.title("IAF membrame dynamics")
plt.xlabel("$t$")
plt.ylabel("$V_{mem}$")
plt.show()