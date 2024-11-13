import numpy as np
import matplotlib.pyplot as plt

t_sim = 100  # ms
tau = 20  # ms
v_th = 10.0  # Membrane spiking threshold
v_reset = 0  # Resting potential
vmem = [0]  # Initial membrane potential
i_syn = 11  # unit free, assuming constant synaptic input
spike_train = []
for t in range(t_sim):
    # Check threshold
    if vmem[t] >= v_th:
        vmem[t] = v_reset
        spike_train.append(t)
    # Membrane dynamics
    dv = (1 / tau) * (-vmem[t] + i_syn)
    # Save data
    vmem.append(vmem[t] + dv)

print(
    f"Neuron spikes {len(spike_train)} times at the following simulation time steps :{spike_train}"
)

# Plot membrane potential

plt.figure()
plt.plot(range(t_sim + 1), vmem)
plt.title("LIF membrame dynamics")
plt.xlabel("$t$")
_ = plt.ylabel("$V_{mem}$")

plt.show()