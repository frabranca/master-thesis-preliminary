from tonic.datasets.nmnist import NMNIST
import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
from tonic.transforms import ToFrame
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Roboto'


root_dir = './../NMNIST'
n_time_steps = 100
batch_size = 4
num_workers = 4
to_raster = ToFrame(sensor_size=NMNIST.sensor_size, n_time_bins=n_time_steps)

snn_test_dataset = NMNIST(save_to=root_dir, train=False, transform=to_raster)
snn_test_dataloader = DataLoader(snn_test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=False)

samples = {i: None for i in range(10)}
# Iterate through the DataLoader to collect one example for each digit
for images, labels in snn_test_dataloader:
    for image, label in zip(images, labels):
        digit = label.item()
        if samples[digit] is None:
            samples[digit] = image[:, 0:1, :] 

        # Check if we have collected one example for each digit
        if all(ex is not None for ex in samples.values()):
            break

    if all(ex is not None for ex in samples.values()):
        break

plt.figure(figsize=(10,5))
plt.subplot(131)
plt.imshow(samples[0].sum(dim=0).view(34,34))
plt.subplot(132)
plt.imshow(samples[1].sum(dim=0).view(34,34))
plt.subplot(133)
plt.imshow(samples[2].sum(dim=0).view(34,34))
plt.show()

make_video = False
if make_video:
    # Now, examples_by_digit contains one example for each digit (0-9)
    frames = []
    frames = torch.cat([v for v in samples.values()], dim=0)    # for i in range(0,10):

    # Define the output video file and codec settings
    output_file = 'output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec for MP4 format
    fps = 30  # Frames per second
    frame_size = (34, 34)  # Frame dimensions (width, height)

    # Create a VideoWriter object to write the video
    out = cv2.VideoWriter(output_file, fourcc, fps, frame_size, isColor=False)  # isColor=False for grayscale

    # Convert each frame tensor to a NumPy array and write it to the video
    for frame_tensor in frames:
        frame_np = frame_tensor.squeeze(0).cpu().numpy()  # Convert and remove the first dimension
        frame_np = np.uint8(frame_np * 255)  # Convert to 8-bit for visualization
        out.write(frame_np)  # Write the frame to the video

    # Release the VideoWriter object
    out.release()
