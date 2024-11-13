import cv2
import numpy as np

def edit_events_video(frames, file_name):
    cmap = {1: (0, 255, 0), 0: (0, 0, 0), -1: (0, 0, 255)}
    
    width = np.shape(frames[0])[0]
    height = np.shape(frames[0])[1]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_name, fourcc, 120.0, (width, height))  # Adjust the frame rate as needed

    # Loop through the frames and save them to the video
    for frame in frames:
        # Create an RGB image with the specified colors
        colored_frame = np.zeros((width, height, 3), dtype=np.uint8)

        for value, color in cmap.items():
            colored_frame[frame == value] = color

        colored_frame = np.swapaxes(colored_frame, 0, 1)
        out.write(colored_frame)

    # Release the VideoWriter
    out.release()

