from torch import nn
from tonic.datasets.nmnist import NMNIST
import os
from tonic.transforms import ToFrame
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from tqdm.notebook import tqdm
from torch.nn import CrossEntropyLoss
import samna
from torch.utils.data import Subset
from sinabs.backend.dynapcnn import DynapcnnNetwork
from collections import Counter
from utils import calculate_synops
import sinabs.layers as sl
from torch import nn
from sinabs.activation.surrogate_gradient_fn import PeriodicExponential
from utils import calculate_synops
from evaluate import eval_snn, eval_speck

epochs = 1
lr = 1e-3
batch_size = 4
num_workers = 4
device = "cuda:0"
shuffle = True
target_synops = 5e6

# SNN
snn_bptt = nn.Sequential(
    # [2, 34, 34] -> [8, 34, 34] -> [8, 17, 17]
    nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(3, 3), padding=(1, 1), bias=False),
    sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
    nn.AvgPool2d(2, 2),
    # [8, 17, 17] -> [16, 17, 17] -> [16, 8, 8]
    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=(1, 1), bias=False),
    sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
    nn.AvgPool2d(2, 2),
    # [16 * 8 * 8] -> [16, 4, 4]
    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2),  bias=False),
    sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
    # [16 * 4 * 4] -> [10]
    nn.Flatten(),
    nn.Linear(16 * 4 * 4, 10, bias=False),
    sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
)

# init the model weights
for layer in snn_bptt.modules():
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(layer.weight.data)

root_dir = './../NMNIST'
n_time_steps = 100
to_raster = ToFrame(sensor_size=NMNIST.sensor_size, n_time_bins=n_time_steps)

snn_train_dataset = NMNIST(save_to=root_dir, train=True, transform=to_raster)
snn_test_dataset = NMNIST(save_to=root_dir, train=False, transform=to_raster)
snn_train_dataloader = DataLoader(snn_train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=True)
snn_test_dataloader = DataLoader(snn_test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=False)

snn_bptt = snn_bptt.to(device=device)

# TRAINING 
optimizer = SGD(params=snn_bptt.parameters(), lr=lr)
criterion = CrossEntropyLoss()

trained_model_file = 'my_model_snn2.pth'

if os.path.isfile(trained_model_file):
    trained_model = torch.load(trained_model_file, map_location=torch.device(device))
    for i, layer in enumerate(snn_bptt):
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            key = str(i) + '.weight'
            layer.weight.data = trained_model[key]
        if isinstance(layer, sl.IAFSqueeze):
            key = str(i) + '.v_mem'
            layer.v_mem = trained_model[key]

else:
    # TRAINING LOOP
    for e in range(epochs):
        train_p_bar = tqdm(snn_train_dataloader)
        for data, label in train_p_bar:
            # reshape the input from [Batch, Time, Channel, Height, Width] into [Batch*Time, Channel, Height, Width]
            data = data.reshape(-1, 2, 34, 34).to(dtype=torch.float, device=device)
            label = label.to(dtype=torch.long, device=device)
            # forward
            optimizer.zero_grad()
            output = snn_bptt(data)

            # reshape the output from [Batch*Time,num_classes] into [Batch, Time, num_classes]
            output = output.reshape(batch_size, n_time_steps, -1)
            # accumulate all time-steps output for final prediction
            output = output.sum(dim=1)

            # calculate loss
            loss = criterion(output, label)
            # backward
            loss.backward()
            optimizer.step()
            
            # detach the neuron states and activations from current computation graph(necessary)
            for layer in snn_bptt.modules():
                if isinstance(layer, sl.StatefulLayer):
                    for name, buffer in layer.named_buffers():
                        buffer.detach_()
            
            # set progressing bar
            train_p_bar.set_description(f"Epoch {e} - BPTT Training Loss: {round(loss.item(), 4)}")

    torch.save(snn_bptt.state_dict(), trained_model_file)

evaluate_snn = False
if evaluate_snn:
    accuracy = eval_snn(snn_bptt, snn_test_dataloader, device, batch_size, n_time_steps)
    print(accuracy)

evaluate_speck = False
if evaluate_speck:
    accuracy = eval_speck(snn_bptt)   
    print(accuracy)    

evaluate_single = False
if evaluate_single:
    images, labels = next(iter(snn_test_dataloader))
    images = images.reshape(-1, 2, 34, 34).to(dtype=torch.float, device=device)
    output, synops_per_sec = calculate_synops(snn_bptt, images, verbose=False)
    output = output.reshape(batch_size, n_time_steps, -1)
    output = output.sum(dim=1)
    pred = output.argmax(dim=1, keepdim=True)
    
    print('Prediction: ', pred.T)
    print('Labels: ', labels)
    print('Synops/s (millions): ', synops_per_sec/1e6)     

evaluate_speck_vs_snn = True
if evaluate_speck_vs_snn:
    snn_test_dataset = NMNIST(save_to=root_dir, train=False)
    # for time-saving, we only select a subset for on-chip infernce， here we select 1/100 for an example run
    subset_indices = list(range(0, len(snn_test_dataset), 100))
    snn_test_dataset = Subset(snn_test_dataset, subset_indices)
    samples = tqdm(snn_test_dataset)

    # evalutate speck
    snn_bptt.to(device="cpu")
    dynapcnn = DynapcnnNetwork(snn=snn_bptt, input_shape=(2, 34, 34), discretize=True, dvs_input=False)
    devkit_name = "speck2edevkit:0"
    dynapcnn.to(device=devkit_name, chip_layers_ordering="auto")

    for layer in dynapcnn.samna_config.cnn_layers:
        print(layer.output_decimator_enable)
    import pdb; pdb.set_trace()
    same_predictions = []
    correct_sinabs = []
    correct_speck = []

    for events, label in samples:
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

        output_events = dynapcnn(samna_event_stream)
        print(output_events)

        neuron_index = [each.feature for each in output_events]
        if len(neuron_index) != 0:
            frequent_counter = Counter(neuron_index)
            prediction_speck = frequent_counter.most_common(1)[0][0]

        # evaluate SNN
        to_raster = ToFrame(sensor_size=NMNIST.sensor_size, n_time_bins=n_time_steps)
        images = torch.tensor(to_raster(events))
        
        # print(images_single)

        images = images.reshape(-1, 2, 34, 34).to(dtype=torch.float) #, device=device)
        
        output = snn_bptt(images)
        output = output.sum(dim=0)

        prediction_sinabs = output.argmax(dim=0, keepdim=True).item()

        if prediction_sinabs == prediction_speck:
            same_predictions.append(1)
        else:
            print(prediction_sinabs, prediction_speck, label)

        if prediction_sinabs == label:
            correct_sinabs.append(1)
        if prediction_speck == label:
            correct_speck.append(1)

    print('same predictions: ', len(same_predictions)/100)
    print('correct sinabs: ', len(correct_sinabs)/100)
    print('correct speck: ', len(correct_speck)/100)

