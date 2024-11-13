import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# Define the data type for the structured array

dataset_name = 'slider_hdr_far'
dataset_folder = 'datasets/' + dataset_name

npz_file_name = dataset_folder + '/frames_5ms.npz'
txt_file_name = dataset_folder + '/events.txt'

width = 240
height = 180
time_window = 5 #ms

if os.path.exists(npz_file_name):
    frames = np.load(npz_file_name, allow_pickle=True)['arr_0']
else:
    events = pd.read_csv(txt_file_name, sep=' ', header=None, names=['t', 'x', 'y', 'p'])  
    events['t'] = round(events['t'] * 1000/time_window).astype(int) * time_window

    timestamps = np.unique(events['t'])

    frames = []

    for t in timestamps:
        frame = np.zeros((width, height))
        events_time_window = events[events['t']==t]
        for i, ev in events_time_window.iterrows():
            x = events['x'][i]
            y = events['y'][i]
            p = events['p'][i]
            if p==0:
                frame[x][y] = -1
            else:
                frame[x][y] = p
                
        frame = sp.sparse.csr_matrix(frame)
        frames.append(frame)

    frames = np.array(frames)
    np.savez(npz_file_name, frames)

output_video = dataset_folder + '/events_5ms.mp4'
cmap = {1: (0, 255, 0), 0: (0, 0, 0), -1: (255, 0, 0)}

# Create a VideoWriter object to save the frames as a video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
out = cv2.VideoWriter(output_video, fourcc, 120.0, (width, height))  # Adjust the frame rate as needed


# Loop through the frames and save them to the video
i = 0
for frame in frames:
    # Create an RGB image with the specified colors
    colored_frame = np.zeros((width, height, 3), dtype=np.uint8)

    for value, color in cmap.items():
        colored_frame[frame.toarray() == value] = color
    
    
    colored_frame = np.swapaxes(colored_frame, 0, 1)
    out.write(colored_frame)

# Release the VideoWriter
out.release()
