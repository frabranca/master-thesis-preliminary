import torch
import samna
import samnagui
import time
import random
import copy
import matplotlib.pyplot as plt

from torch import nn
from sinabs.backend.dynapcnn import DynapcnnNetwork
from multiprocessing import Process
from sinabs.from_torch import from_model
from sinabs.layers.pool2d import SumPool2d
from typing import Union
from matplotlib.ticker import MaxNLocator
from utils import remapping_output_index, create_fake_input_events


input_shape = (1, 16, 16)

cnn = nn.Sequential(SumPool2d(kernel_size=(1, 1)),
                    nn.Conv2d(in_channels=1,
                              out_channels=2,
                              kernel_size=(16, 16),
                              stride=(1, 1),
                              padding=(0, 0),
                              bias=False),
                    nn.ReLU())

# set handcraft weights for the CNN
weight_ones = torch.ones(1, 8, 16, dtype=torch.float32)
weight_zeros = torch.zeros(1, 8, 16, dtype=torch.float32)

channel_1_weight = torch.cat([weight_ones, weight_zeros], dim=1).unsqueeze(0)
channel_2_weight = torch.cat([weight_zeros, weight_ones], dim=1).unsqueeze(0)
handcraft_weight = torch.cat([channel_1_weight, channel_2_weight], dim=0)

output_cnn_lyr_id = 1
cnn[output_cnn_lyr_id].weight.data = handcraft_weight

cnn[output_cnn_lyr_id] = remapping_output_index(cnn[output_cnn_lyr_id])

# cnn to snn
snn = from_model(cnn, input_shape=input_shape, batch_size=1).spiking_model
# snn to DynapcnnNetwork
dynapcnn_net = DynapcnnNetwork(snn=snn, input_shape=input_shape, dvs_input=False)

readout_threshold = 1

# init devkit config
devkit_cfg = dynapcnn_net.make_config(device="speck2edevkit:0")

# ========== modify devkit config ==========

"""cnn layers configuration"""
# send to output spike from cnn output layer to readout layer as its input
cnn_output_layer = dynapcnn_net.chip_layers_ordering[-1]
# the readout layer id is fixed for speck2e devkit which is 12
readout_layer = 12
print(f'link output layer: {cnn_output_layer} to readout layer: {readout_layer}')
devkit_cfg.cnn_layers[cnn_output_layer].monitor_enable = True
devkit_cfg.cnn_layers[cnn_output_layer].destinations[0].enable = True
devkit_cfg.cnn_layers[cnn_output_layer].destinations[0].layer = readout_layer

"""readout layer configuration"""
devkit_cfg.readout.enable = True
devkit_cfg.readout.readout_configuration_sel = 0b10
devkit_cfg.readout.output_mode_sel = 0b01
devkit_cfg.readout.low_pass_filter_disable = True
devkit_cfg.readout.threshold = readout_threshold


"""dvs layer configuration"""
# link the dvs layer to the 1st layer of the cnn layers
devkit_cfg.dvs_layer.destinations[0].enable = True
devkit_cfg.dvs_layer.destinations[0].layer = dynapcnn_net.chip_layers_ordering[0]
# merge the polarity of input events
devkit_cfg.dvs_layer.merge = True
# drop the raw input events from the dvs sensor, since we write events to devkit manually
devkit_cfg.dvs_layer.pass_sensor_events = False
# enable monitoring the output from dvs pre-preprocessing layer
devkit_cfg.dvs_layer.monitor_enable = True

# open devkit
device_names = [each.device_type_name for each in samna.device.get_all_devices()]
print(f"Open device: {device_names[0]}")
devkit = samna.device.open_device(device_names[0])

 # init the graph
samna_graph = samna.graph.EventFilterGraph()


# init necessary nodes in samna graph
# node for writing fake inputs into devkit
input_buffer_node = samna.BasicSourceNode_speck2e_event_speck2e_input_event()
# node for reading ReadoutValue 
readout_value_buffer_node = samna.BasicSinkNode_speck2e_event_output_event()
# node for reading ReadoutPinValue
pin_value_buffer_node = samna.BasicSinkNode_speck2e_event_output_event()
# node for reading Spike(i.e. the output from last CNN layer)
spike_buffer_node = samna.BasicSinkNode_speck2e_event_output_event()


# build input branch for graph
samna_graph.sequential([input_buffer_node, devkit.get_model_sink_node()])


# build output branches for graph
# branch #1: for the dvs input visualization
_, _, streamer = samna_graph.sequential(
    [devkit.get_model_source_node(), "Speck2eDvsToVizConverter", "VizEventStreamer"])
# branch #2: for obtaining the ReadoutValue
_, type_filter_node_readout, _ = samna_graph.sequential(
    [devkit.get_model_source_node(), "Speck2eOutputEventTypeFilter", readout_value_buffer_node])
# branch #3: for obtaining the ReadoutPinValue
_, type_filter_node_pin, _ = samna_graph.sequential(
    [devkit.get_model_source_node(), "Speck2eOutputEventTypeFilter", pin_value_buffer_node])
# branch #4: for obtaining the output Spike from cnn output layer
_, type_filter_node_spike, _ = samna_graph.sequential(
    [devkit.get_model_source_node(), "Speck2eOutputEventTypeFilter", spike_buffer_node])


# set the streamer nodes of the graph
# tcp communication port for dvs input data visualization
streamer_endpoint = 'tcp://0.0.0.0:40000'  
streamer.set_streamer_endpoint(streamer_endpoint)
# add desired type for filter node
type_filter_node_readout.set_desired_type("speck2e::event::ReadoutValue")
type_filter_node_pin.set_desired_type("speck2e::event::ReadoutPinValue")
type_filter_node_spike.set_desired_type("speck2e::event::Spike")


# start samna graph before using the devkit
samna_graph.start()

# init samna node for tcp transmission
samna_node = samna.init_samna()
sender_endpoint = samna_node.get_sender_endpoint()
receiver_endpoint = samna_node.get_receiver_endpoint()
visualizer_id = 3
time.sleep(1)  # wait tcp connection build up, this is necessary to open remote node.

# define a function that run the GUI visualizer in the sub-process
def run_visualizer(receiver_endpoint, sender_endpoint, visualizer_id):

    samnagui.runVisualizer(0.6, 0.6, receiver_endpoint, sender_endpoint, visualizer_id)

    return

# create the subprocess
gui_process = Process(target=run_visualizer, args=(receiver_endpoint, sender_endpoint, visualizer_id))
gui_process.start()
print("GUI process started, you should see a window pop up!")

# wait for open visualizer and connect to it.
timeout = 10
begin = time.time()
name = "visualizer" + str(visualizer_id)
while time.time() - begin < timeout:

    try:

        time.sleep(0.05)
        samna.open_remote_node(visualizer_id, name)

    except:

        continue

    else:

        visualizer = getattr(samna, name)
        print(f"successful connect the GUI visualizer!")
        break

# set up the visualizer and GUI layout

# set visualizer's receiver endpoint to streamer's sender endpoint for tcp communication
visualizer.receiver.set_receiver_endpoint(streamer_endpoint)
# connect the receiver output to splitter inside the visualizer
visualizer.receiver.add_destination(visualizer.splitter.get_input_channel())

# add plots to gui
activity_plot_id = visualizer.plots.add_activity_plot(128, 128, "DVS Layer")
plot = visualizer.plot_0
plot.set_layout(0, 0, 0.5, 0.89)

visualizer.splitter.add_destination("dvs_event", visualizer.plots.get_plot_input(activity_plot_id))
visualizer.plots.report()

print("now you should see a change on the GUI window!")

dk_io = devkit.get_io_module()
slow_clk_freq = 20 # Hz
dk_io.set_slow_clk_rate(slow_clk_freq)
dk_io.set_slow_clk(True)

# create fake input events
input_time_length = 3 # seconds
data_rate = 5000
input_events = create_fake_input_events(time_sec=3, data_rate=data_rate)

print(f"number of fake input spikes: {len(input_events)}")

# estimated slow-clock cycle for processing the input spikes
clock_cycles_esitmated = slow_clk_freq * input_time_length

# to read the ReadoutPinValue, we need to modify the devkit's readout layer's config a little bit
devkit_cfg.readout.monitor_enable = False
devkit_cfg.readout.readout_pin_monitor_enable = True


# then apply the config to devkit
devkit.get_model().apply_configuration(devkit_cfg)
time.sleep(0.1)


# write the fake input into the devkit

# enable & reset the stop-watch of devkit, this is mainly for the timestamp processing for the input&output events.
stop_watch = devkit.get_stop_watch()
stop_watch.set_enable_value(True)
stop_watch.reset()
time.sleep(0.01)

# clear output buffer
pin_value_buffer_node.get_events()
spike_buffer_node.get_events()

# write through the input buffer node
input_time_length = (input_events[-1].timestamp - input_events[0].timestamp) / 1e6
input_buffer_node.write(input_events)
# sleep till all input events is sent and processed
time.sleep(input_time_length + 0.02)

# read the ReadoutPinValue from related buffer node
pin_value_events = pin_value_buffer_node.get_events()

print("You should see the input events through the GUI window!")

# the number of the ReadoutPinValue should be very close to (or the same as) the estimated clock cycle.
print(f"The estimated clock cycle is {clock_cycles_esitmated}")
print(f"Number of ReadoutPinValue events: {len(pin_value_events)}")

# get the timestamp of the output event
pin_value_timestamp = [each.timestamp for each in pin_value_events]
# shift timestep starting from 0
start_t = pin_value_timestamp[0]
pin_value_timestamp = [each - start_t for each in pin_value_timestamp]

# get the index of the output neuron with maximum output
neuron_id = [each.index for each in pin_value_events]

# plot the output neuron index vs. time
fig, ax = plt.subplots()
ax.scatter(pin_value_timestamp, neuron_id)
ax.set(xlim=(0, 3e6),ylim=(0, 2.5))
ax.set_xlabel("time( micro sec)")
ax.set_ylabel("neuron index")
ax.set_title("ReadoutPinValue")
ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # make y-axis only show integer
plt.show()


gui_process.terminate()
gui_process.join()

samna_graph.stop()
samna.device.close_device(devkit)