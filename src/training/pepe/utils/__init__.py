import os
import imageio
from functools import partial
import pandas as pd
import pickle
import itertools
import torch
from PIL import Image
from preprocessing import transform
# from utils import loss_scaler


def load_dataset(from_path):
    dataset = pd.read_pickle(from_path)
    with open(from_path+'.meta', 'rb') as f:
        dataset_info = pickle.load(f)
    return dataset, dataset_info


def save_dataset(dataset, to_path, multiclass=False):
    dataset_info = {}
    if not multiclass:
        dataset_info['id_to_label'] = list(
            set(itertools.chain.from_iterable(dataset['all_tags'].to_list())))
        dataset_info['label_to_id'] = dict(
            zip(dataset_info['id_to_label'], range(0, len(dataset_info['id_to_label']))))
    else:
        dataset_info['id_to_label'] = list(
            set(dataset['all_tags'].to_list()))
        dataset_info['label_to_id'] = dict(
            zip(dataset_info['id_to_label'], range(0, len(dataset_info['id_to_label']))))
    dataset_info['n_labels'] = len(dataset_info['id_to_label'])
    dataset.to_pickle(to_path)
    with open(to_path+'.meta', 'wb') as f:
        pickle.dump(dataset_info, f)


def gif_id_to_filepath(gif_id, ext='.mp4') -> str:
    GIFS_SOURCE = '/home/xingyaow/gif-reply/data/processed/dataset/gifs'

    def _gif_id_to_structured_path(gif_id):
        return os.path.join(gif_id[0], gif_id[1], gif_id[2], gif_id[3:])
    return os.path.join(GIFS_SOURCE, _gif_id_to_structured_path(gif_id)+ext)


def select_4_frames(frames: list):
    n_frames = len(frames)
    idx = [i*(n_frames//4) for i in range(4)]
    return [frames[i] for i in idx]

# ==== load GIF in preprocessed form by gif_id ====


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


def _get_attr_from_opt(opt, attribute_name, default_value):
    return getattr(opt, attribute_name) if hasattr(
        opt, attribute_name) else default_value
