import time
import samna
from torch import nn
from sinabs.layers import IAF
from connect_to_device import make_dynapcnn_network, generate_events
import matplotlib.pyplot as plt
import numpy as np

snn = nn.Sequential(
    nn.Linear(1, 1, bias=False),
    IAF(spike_threshold=0.1, record_states=True),
)

input_shape = (1,1,1)

# make dynapcnn network
dynapcnn = make_dynapcnn_network(snn, input_shape)
print(dynapcnn.samna_config.cnn_layers[0])
threshold = dynapcnn.samna_config.cnn_layers[0].threshold_high

# define time steps
dt = 0.1
t = 10.
steps = int(t/dt)

# input events
input_events, input_events_sinabs = generate_events(0.5, steps, input_shape)
# print(input_events_sinabs)
output_sinabs = snn(input_events_sinabs)

# Check if neuron states decrease along with time pass by

input_spikes = []
v_mem = []
time_steps = []
output_spikes = []

for step in range(0, steps):
    # write input 
    dynapcnn.samna_input_buffer.write([input_events[step]])
    time.sleep(dt)

    if type(input_events[step]) == samna.speck2e.event.Spike:
        input_spikes.append(1)
    else:
        input_spikes.append(0)

    # get outputs
    output_events = dynapcnn.samna_output_buffer.get_events()
    for out_ev in output_events:
        if type(out_ev)==samna.speck2e.event.NeuronValue:
            v_mem.append(out_ev.neuron_state)
            time_steps.append(dt*step)
            output_spikes.append(0)
            
        else:
            print(out_ev)
            # spike is received
            v_mem.append(threshold)
            time_steps.append(dt*step)
            output_spikes.append(1)

# cut last elements
# add zero elements to shift
input_spikes = [0,0,0,0] + input_spikes

plt.figure(figsize=(7,7))

plt.subplot(311)
plt.plot(np.arange(0,10.4,0.1), input_spikes)
plt.xlim(0,10)
plt.ylabel('Input Spikes')
plt.grid()

plt.subplot(312)
plt.plot(time_steps, v_mem)
plt.ylabel('Membrane Potential')
plt.xlim(0,10)
plt.grid()

plt.subplot(313)
plt.plot(time_steps, output_spikes)
plt.ylabel('Output Spikes')
plt.xlabel('Time Steps')
plt.xlim(0,10)
plt.grid()

plt.show()