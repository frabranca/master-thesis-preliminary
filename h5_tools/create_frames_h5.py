import h5py
import numpy as np
import scipy as sp
import cv2
import os

width = 180
height = 180
time_window = 5
os.makedirs('h5_sequences', exist_ok=True)

for s in range(1):
    file = h5py.File('/home/francesco/speck/poster_rotation.h5', 'r')
    print(s)
    events = file['events']
    xs = events['xs'][:]
    ys = events['ys'][:]
    ps = events['ps'][:]
    ts = events['ts'][:] - events['ts'][0]

    ts = np.round(ts * 1000/time_window).astype(int) * time_window

    timestamps = np.unique(ts)
    frames = []

    for t in timestamps:
        frame = np.zeros((width, height))
        args = np.argwhere(ts==t)

        x_tw = xs[args]
        y_tw = ys[args]
        p_tw = ps[args]

        for i in range(len(x_tw)):
            x = x_tw[i][0]
            y = y_tw[i][0]
            p = p_tw[i][0]

            if p==0:
                frame[x][y] = -1
            else:
                frame[x][y] = p
                
        frame = sp.sparse.csr_matrix(frame)
        frames.append(frame)

    output_video = 'h5_sequences/rotation_seq_' + str(s) + '.mp4'
    cmap = {1: (0, 255, 0), 0: (0, 0, 0), -1: (0, 0, 255)}
    
    # Create a VideoWriter object to save the frames as a video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    out = cv2.VideoWriter(output_video, fourcc, 120.0, (width, height))  # Adjust the frame rate as needed

    # Loop through the frames and save them to the video
    for frame in frames:
        # Create an RGB image with the specified colors
        colored_frame = np.zeros((width, height, 3), dtype=np.uint8)

        for value, color in cmap.items():
            colored_frame[frame.toarray() == value] = color

        colored_frame = np.swapaxes(colored_frame, 0, 1)
        out.write(colored_frame)

    # Release the VideoWriter
    out.release()

