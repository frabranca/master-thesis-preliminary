import torch
from tqdm.auto import tqdm
import numpy as np
import torch.nn as nn
from torchvision import datasets
from sinabs.from_torch import from_model
from sinabs.backend.dynapcnn import io
from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.chip_factory import ChipFactory
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.nn.functional as F
import sinabs.layers as sl
import random
import samna

def make_layers(num_layers=9):
    layers = []
    for i in range(num_layers):
        layer = samna.speck2e.configuration.CnnLayerConfig()
        layers.append(layer)
    
    # LAYER 0 ---------------------------------------------------------------------------

    layers[0].dimensions.padding = samna.Vec2_unsigned_char(3//2,3//2) #(3//2)
    layers[0].dimensions.stride = samna.Vec2_unsigned_char(2,2)
    layers[0].dimensions.kernel_size = 3

    layers[0].dimensions.input_shape.feature_count = 2
    layers[0].dimensions.input_shape.size = samna.Vec2_unsigned_char(128,128)
    layers[0].dimensions.output_shape.feature_count = 4
    layers[0].dimensions.output_shape.size = samna.Vec2_unsigned_char(64,64)

    weights_dimensions = (4, 2, 3, 3)
    weight_zeros = np.zeros(weights_dimensions).astype(int).tolist()
    layers[0].weights = weight_zeros

    biases_dimensions = 4
    biases_zeros = np.zeros(biases_dimensions).astype(int).tolist()
    layers[0].biases = biases_zeros

    neurons_dimensions = (4,64,64)
    neurons_zeros = np.zeros(neurons_dimensions).astype(int).tolist()
    layers[0].neurons_initial_value = neurons_zeros

    # LAYER 1 ---------------------------------------------------------------------------

    layers[1].dimensions.padding = samna.Vec2_unsigned_char(3//2,3//2) #(3//2)
    layers[1].dimensions.stride = samna.Vec2_unsigned_char(1,1)
    layers[1].dimensions.kernel_size = 3

    layers[1].dimensions.input_shape.feature_count = 8
    layers[1].dimensions.input_shape.size = samna.Vec2_unsigned_char(64,64)
    layers[1].dimensions.output_shape.feature_count = 8
    layers[1].dimensions.output_shape.size = samna.Vec2_unsigned_char(64,64)

    weights_dimensions = (8, 8, 3, 3)
    weight_zeros = np.zeros(weights_dimensions).astype(int).tolist()
    layers[1].weights = weight_zeros

    biases_dimensions = 8
    biases_zeros = np.zeros(biases_dimensions).astype(int).tolist()
    layers[1].biases = biases_zeros

    neurons_dimensions = (8,64,64)
    neurons_zeros = np.zeros(neurons_dimensions).astype(int).tolist()
    layers[1].neurons_initial_value = neurons_zeros

    # LAYER 2 ---------------------------------------------------------------------------

    layers[2].dimensions.padding = samna.Vec2_unsigned_char(3//2,3//2) #(3//2)
    layers[2].dimensions.stride = samna.Vec2_unsigned_char(1,1)
    layers[2].dimensions.kernel_size = 3

    layers[2].dimensions.input_shape.feature_count = 8
    layers[2].dimensions.input_shape.size = samna.Vec2_unsigned_char(64,64)
    layers[2].dimensions.output_shape.feature_count = 4
    layers[2].dimensions.output_shape.size = samna.Vec2_unsigned_char(64,64)

    weights_dimensions = (4, 8, 3, 3)
    weight_zeros = np.zeros(weights_dimensions).astype(int).tolist()
    layers[2].weights = weight_zeros

    biases_dimensions = 4
    biases_zeros = np.zeros(biases_dimensions).astype(int).tolist()
    layers[2].biases = biases_zeros

    neurons_dimensions = (4,64,64)
    neurons_zeros = np.zeros(neurons_dimensions).astype(int).tolist()
    layers[2].neurons_initial_value = neurons_zeros

    # # LAYER 3 ---------------------------------------------------------------------------

    layers[3].dimensions.padding = samna.Vec2_unsigned_char(3//2,3//2) #(3//2)
    layers[3].dimensions.stride = samna.Vec2_unsigned_char(2,2)
    layers[3].dimensions.kernel_size = 3

    layers[3].dimensions.input_shape.feature_count = 8
    layers[3].dimensions.input_shape.size = samna.Vec2_unsigned_char(64,64)
    layers[3].dimensions.output_shape.feature_count = 16
    layers[3].dimensions.output_shape.size = samna.Vec2_unsigned_char(32,32)

    weights_dimensions = (16, 8, 3, 3)
    weight_zeros = np.zeros(weights_dimensions).astype(int).tolist()
    layers[3].weights = weight_zeros

    biases_dimensions = 16
    biases_zeros = np.zeros(biases_dimensions).astype(int).tolist()
    layers[3].biases = biases_zeros

    neurons_dimensions = (16,32,32)
    neurons_zeros = np.zeros(neurons_dimensions).astype(int).tolist()
    layers[3].neurons_initial_value = neurons_zeros

    # DESTINATIONS -----------------------------------------------------------------------
    layers[0].destinations[0].layer = 1
    layers[0].destinations[0].enable = True

    layers[1].destinations[0].layer = 3
    layers[1].destinations[0].enable = True
    layers[1].destinations[1].layer = 2
    layers[1].destinations[1].enable = True

    layers[2].destinations[0].layer = 1
    layers[2].destinations[0].enable = True
    layers[2].destinations[0].feature_shift = 4
    
    layers[3].destinations[0].layer = 12
    layers[3].destinations[0].enable = True

    # LEAKS
    layers[0].leak_enable = True
    layers[0].leak_internal_slow_clk_enable = True
    layers[1].leak_enable = True
    layers[1].leak_internal_slow_clk_enable = True
    layers[2].leak_enable = True
    layers[2].leak_internal_slow_clk_enable = True
    layers[3].leak_enable = True
    layers[3].leak_internal_slow_clk_enable = True

    # layers[3].destinations[0].layer = 4
    # layers[3].destinations[0].enable = True
    # layers[4].destinations[0].layer = 12
    # layers[4].destinations[0].enable = True
    # torch.cat([I, W*O], dim=1)

    for i in range(num_layers):
        print('layer', i)
        if layers[i].destinations[0].enable==1:
            print(layers[i].destinations[0].layer)
        if layers[i].destinations[1].enable==1:
            print(layers[i].destinations[1].layer)

    return layers

layers = make_layers()

# print(layers[1].destinations[1].feature_shift)

# print(layer.destinations)

config = samna.speck2e.configuration.SpeckConfiguration()
config.cnn_layers = layers
print(np.array(layers[0].weights).shape)
# print(config.cnn_layers)

device_names = [each.device_type_name for each in samna.device.get_all_devices()]
devkit = samna.device.open_device(device_names[0])


devkit.get_model().apply_configuration(config)
