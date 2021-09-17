import os
import imageio
import pandas as pd
import torch
import ast
from PIL import Image
from preprocessing import transform

def load_dataset(from_path):
    dataset = pd.read_csv(from_path)
    dataset["tags"] = dataset["tags"].apply(ast.literal_eval)

    dataset_info = {}
    dataset_info['id_to_label'] = list(
            set(dataset['tags'].to_list()))
    dataset_info['label_to_id'] = dict(
            zip(dataset_info['id_to_label'], range(0, len(dataset_info['id_to_label']))))
    dataset_info['n_labels'] = len(dataset_info['id_to_label'])
    return dataset, dataset_info

def gif_id_to_filepath(gif_id, ext='.mp4') -> str:
    GIFS_SOURCE = os.environ.get("GIF_PATH")

    def _gif_id_to_structured_path(gif_id):
        return os.path.join(gif_id[0], gif_id[1], gif_id[2], gif_id[3:])
    return os.path.join(GIFS_SOURCE, _gif_id_to_structured_path(gif_id)+ext)


def select_4_frames(frames: list):
    n_frames = len(frames)
    idx = [i*(n_frames//4) for i in range(4)]
    return [frames[i] for i in idx]

def load_gif(gif_id, frame_reduce_fn=select_4_frames):
    # 1. Check if the GIF is already preloaded
    _preload_path = gif_id_to_filepath(gif_id, ext='.pt')
    _preload_exists = os.path.exists(_preload_path)
    if _preload_exists:
        return torch.load(_preload_path)

    # 2. Load the GIF if is not preloaded
    def read_gif_id(gif_id, frame_reduce_fn=None) -> list:
        path = gif_id_to_filepath(gif_id)
        vid = imageio.get_reader(path)
        frames = [i for i in vid]  # list of image frames
        return frame_reduce_fn(frames) if frame_reduce_fn else frames
    raw_frames = read_gif_id(gif_id, frame_reduce_fn=frame_reduce_fn)
    assert len(raw_frames) == 4
    # Process Frames
    frames = [Image.fromarray(frame) for frame in raw_frames]
    frames = [transform(frame) for frame in frames]
    gif = torch.stack(frames)
    torch.save(gif, _preload_path)
    return gif
