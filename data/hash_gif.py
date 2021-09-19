'''This includes an example of calculating the average-hash for a GIF file in mp4 format.

Required packages:
```pip install opencv-python imagehash```

Example usage:
```python3 hash_gif.py test-gif.mp4```
'''

import argparse
import imagehash
import logging
import cv2
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("filepath", help="filepath of the GIF to run hash on.")
args = parser.parse_args()


def read_gif(file_path):
    cap = cv2.VideoCapture(file_path)
    video = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video.append(frame)
    return np.array(video)


def hash_gif(file_path):
    try:
        gif = read_gif(file_path)
        n_frame = gif.shape[0]
        first_frame = Image.fromarray(gif[0])
        mid_frame = Image.fromarray(gif[n_frame//2])
        last_frame = Image.fromarray(gif[n_frame-1])
        return str(imagehash.average_hash(first_frame)) \
                + str(imagehash.average_hash(mid_frame)) \
                + str(imagehash.average_hash(last_frame))
    except Exception as e:
        logging.exception(e)
        return None

if __name__ == "__main__":
    hash_val = hash_gif(args.filepath)
    print(f"hash value for {args.filepath} is {hash_val}")

