from torch import nn
from tonic.datasets.nmnist import NMNIST
import os
from tonic.transforms import ToFrame
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from tqdm.notebook import tqdm
from torch.nn import CrossEntropyLoss
from sinabs.from_torch import from_model


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

# init the model weights
for layer in cnn.modules():
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(layer.weight.data)

# define a transform that accumulate the events into a single frame image
to_frame = ToFrame(sensor_size=NMNIST.sensor_size, n_time_bins=1)

root_dir = './../NMNIST'
cnn_train_dataset = NMNIST(save_to=root_dir, train=True, transform=to_frame)
cnn_test_dataset = NMNIST(save_to=root_dir, train=False, transform=to_frame)

# check the transformed data
sample_data, label = cnn_train_dataset[0]
print(f"The transformed array is in shape [Time-Step, Channel, Height, Width] --> {sample_data.shape}")


epochs = 3
lr = 1e-3
batch_size = 4
num_workers = 4
device = "cpu"
shuffle = True

cnn = cnn.to(device=device)

cnn_train_dataloader = DataLoader(cnn_train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=shuffle)
cnn_test_dataloader = DataLoader(cnn_test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=shuffle)

optimizer = SGD(params=cnn.parameters(), lr=lr)
criterion = CrossEntropyLoss()

if os.path.isfile('my_model.pth'):
    cnn.load_state_dict(torch.load('my_model.pth')) #, map_location=torch.device('cpu'))
    print("Loaded pre-trained model.")
else:
    for e in range(epochs):

        # train
        train_p_bar = tqdm(cnn_train_dataloader)
        for data, label in train_p_bar:
            # remove the time-step axis since we are training CNN
            # move the data to accelerator
            data = data.squeeze(dim=1).to(dtype=torch.float, device=device)
            label = label.to(dtype=torch.long, device=device)
            # forward
            optimizer.zero_grad()
            output = cnn(data)
            print(output)
            loss = criterion(output, label)
            # backward
            loss.backward()
            optimizer.step()
            # set progressing bar
            train_p_bar.set_description(f"Epoch {e} - Training Loss: {round(loss.item(), 4)}")
    
    torch.save(cnn.state_dict(), 'my_model.pth')

# define a transform that accumulate the events into a raster-like tensor
n_time_steps = 100
to_raster = ToFrame(sensor_size=NMNIST.sensor_size, n_time_bins=n_time_steps)
snn_test_dataset = NMNIST(save_to=root_dir, train=False, transform=to_raster)
snn_test_dataloader = DataLoader(snn_test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=False)

snn_convert = from_model(model=cnn, input_shape=(2, 34, 34), batch_size=batch_size).spiking_model
snn_convert = snn_convert.to(device)
print(snn_convert)

correct_predictions = []
with torch.no_grad():
    test_p_bar = tqdm(snn_test_dataloader)
    print('chilling ...')
    for data, label in test_p_bar:
        # reshape the input from [Batch, Time, Channel, Height, Width] into [Batch*Time, Channel, Height, Width]
        data = data.reshape(-1, 2, 34, 34).to(dtype=torch.float, device=device)
        label = label.to(dtype=torch.long, device=device)
        # forward
        output = snn_convert(data)
        # reshape the output from [Batch*Time,num_classes] into [Batch, Time, num_classes]
        output = output.reshape(batch_size, n_time_steps, -1)
        # accumulate all time-steps output for final prediction
        output = output.sum(dim=1)
        # calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        # compute the total correct predictions
        correct_predictions.append(pred.eq(label.view_as(pred)))
        # set progressing bar
        test_p_bar.set_description(f"Testing SNN Model...")

    correct_predictions = torch.cat(correct_predictions)
    print(f"accuracy of converted SNN: {correct_predictions.sum().item()/(len(correct_predictions))*100}%")

