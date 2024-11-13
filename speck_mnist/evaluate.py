from tonic.datasets.nmnist import NMNIST
import torch
from torch.optim import SGD
from tqdm.notebook import tqdm
import samna
from torch.utils.data import Subset
from sinabs.backend.dynapcnn import DynapcnnNetwork
from collections import Counter
from torch import nn
from sinabs.activation.surrogate_gradient_fn import PeriodicExponential
import matplotlib.pyplot as plt
import numpy as np

def eval_snn(snn, snn_test_dataloader, device, batch_size, n_time_steps):
    correct_predictions = []
    with torch.no_grad():
        test_p_bar = tqdm(snn_test_dataloader)
        for data, label in test_p_bar:
            # reshape the input from [Batch, Time, Channel, Height, Width] into [Batch*Time, Channel, Height, Width]
            data = data.reshape(-1, 2, 34, 34).to(dtype=torch.float, device=device)
            label = label.to(dtype=torch.long, device=device)

            # forward
            output = snn(data)
            # reshape the output from [Batch*Time,num_classes] into [Batch, Time, num_classes]
            output = output.reshape(batch_size, n_time_steps, -1)
            # accumulate all time-steps output for final prediction
            output = output.sum(dim=1)
            # calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            # compute the total correct predictions
            correct_predictions.append(pred.eq(label.view_as(pred)))
            # set progressing bar
            test_p_bar.set_description(f"-BPTT Testing Model...")

        correct_predictions = torch.cat(correct_predictions)
        accuracy = correct_predictions.sum().item()/(len(correct_predictions))*100
    return accuracy

def eval_speck(snn):
    cpu_snn = snn.to(device="cpu")
    root_dir = './../NMNIST'
    dynapcnn = DynapcnnNetwork(snn=snn, input_shape=(2, 34, 34), discretize=True, dvs_input=False)
    devkit_name = "speck2edevkit:0"

    # use the `to` method of DynapcnnNetwork to deploy the SNN to the devkit
    dynapcnn.to(device=devkit_name, chip_layers_ordering="auto")

    speck_test_dataset = NMNIST(save_to=root_dir, train=False)
    # for time-saving, we only select a subset for on-chip infernce， here we select 1/100 for an example run
    subset_indices = list(range(0, len(speck_test_dataset), 100))
    speck_test_dataset = Subset(speck_test_dataset, subset_indices)

    test_p_bar = tqdm(speck_test_dataset)
    test_samples = 0
    correct_samples = 0

    for events, label in test_p_bar:
        # create samna Spike events stream
        samna_event_stream = []
        for ev in events:
            spk = samna.speck2e.event.Spike()
            spk.x = ev['x']
            spk.y = ev['y']
            spk.timestamp = ev['t'] - events['t'][0]
            spk.feature = ev['p']
            # Spikes will be sent to layer/core #0, since the SNN is deployed on core: [0, 1, 2, 3]
            spk.layer = 0
            samna_event_stream.append(spk)

        # inference on chip
        # output_events is also a list of Spike, but each Spike.layer is 3, since layer#3 is the output layer
        output_events = dynapcnn(samna_event_stream)
        
        # use the most frequent output neruon index as the final prediction
        neuron_index = [each.feature for each in output_events]
        if len(neuron_index) != 0:
            frequent_counter = Counter(neuron_index)
            prediction = frequent_counter.most_common(1)[0][0]
        else:
            prediction = -1
        test_p_bar.set_description(f"label: {label}, prediction: {prediction}， output spikes num: {len(output_events)}") 

        if prediction == label:
            correct_samples += 1

        test_samples += 1
    
    accuracy = correct_samples/test_samples
    return accuracy