import itertools
import os
from functools import partial

import imageio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pandarallel import pandarallel
from PIL import Image
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split
from torch.utils import data
from torchvision import transforms


pandarallel.initialize()


def load_dataset(reply_dataset_path, metadata_path):
    metadata = pd.read_csv(metadata_path)
    gif_id_to_tags = dict(metadata[['gif_id', 'tags']].to_numpy())

    dataset = pd.read_csv(reply_dataset_path)
    dataset['tags'] = dataset['child_gif_id'].apply(
        lambda x: gif_id_to_tags.get(x),
    ).apply(ast.literal_eval)

    dataset_info = {}
    dataset_info['id_to_label'] = sorted(
        list(
            set(itertools.chain.from_iterable(dataset['tags'].to_list())),
        ),
    )
    dataset_info['label_to_id'] = dict(
        zip(
            dataset_info['id_to_label'], range(
            0, len(dataset_info['id_to_label']),
            ),
        ),
    )
    dataset_info['n_labels'] = len(dataset_info['id_to_label'])
    return dataset, dataset_info


def tags_to_vector(tags, label_to_id, n_labels) -> np.array:
    """Process given tags into a 1D vector of length `n_labels`,
    each tag in `tags` would be 1 in the vector.
    Used for multilabel classification

    Args:
        tags (list): list of string tags
        label_to_id (dict): a dict that map tag string to tag id (>= 0, < n_labels)
        n_labels ([type]): number of unique labels (classes)

    Returns:
        np.array: created hot vector for given set of tags.
    """
    assert type(tags) == list, 'Expected a list of tags'
    ids = []
    for tag in tags:
        ids.append(label_to_id[tag])
    ret = np.zeros(n_labels)
    ret[ids] = 1
    return ret


def gif_id_to_filepath(gif_id) -> str:
    GIFS_SOURCE = os.environ.get('GIF_PATH')

    def _gif_id_to_structured_path(gif_id):
        return os.path.join(gif_id[0], gif_id[1], gif_id[2], gif_id[3:])
    return os.path.join(GIFS_SOURCE, _gif_id_to_structured_path(gif_id)+'.mp4')


def select_4_frames(frames: list):
    n_frames = len(frames)
    idx = [i*(n_frames//4) for i in range(4)]
    return [frames[i] for i in idx]


def read_gif_id(gif_id, frame_reduce_fn=None, n_frame=None) -> list:
    path = gif_id_to_filepath(gif_id)
    vid = imageio.get_reader(path, 'ffmpeg')
    if not n_frame:
        frames = [i for i in vid]  # list of image frames
        return frame_reduce_fn(frames) if frame_reduce_fn else frames
    else:
        ret = []
        for i, frame in enumerate(vid):
            if i == n_frame:
                break
            ret.append(frame)
        return ret


class GifReplyMediaDataset(data.Dataset):
    def __init__(
        self,
        dataset_path,
        metadata_path,
        train=True,
        test=False,
        dev_size=0.05,
        with_frame_seq=False,
        image_size=None,
        multiclass=False,
        random_state=42,
        reuse_data=None,
        **kwargs,
    ):
        """
        Initialize GifReply dataset and load relavant information.
        Args:
            dataset_path ([type]): filepath to reply dataset.
            train (bool, optional): when true, __getitem__ return elements in train set;
                otherwise returns elements in dev set. Defaults to True.
            test (bool, optional): Whether in test mode, when true, __getitem__ only return
                the model_input without label; otherwise return (X, y_true) pair. Defaults to False.
            dev_size (float, optional): porpotion of data to set to dev set. Defaults to 0.05.
            image_size (int, optional): images in dataset will be automatically cropped to shape of (image_size, image_size)
            with_frame_seq (bool, optional): whether training using sequence of frames [4, frame_size], or
                concated frame [2*frame_size].
            multiclass (bool, optional): when true, the model will runs in multiclass mode,
                y_true returned by __getitem__ will be an integer value denotes the class of the sample;
                otherwise it will returns an hot_vector encoding multiple label in y_true. Defaults to False.
            reuse_data (data.Dataset): another instance of GifReplyDataset to prevent load and preprocess same dataset twice
        """
        # Store arguments
        self.train = train
        self.test = test
        self.multiclass = multiclass
        self.with_frame_seq = with_frame_seq

        if not reuse_data:
            # Load dataset
            self.data, dataset_info = load_dataset(dataset_path, metadata_path)
            self.num_classes = dataset_info['n_labels']
            self.label_to_id = dataset_info['label_to_id']
            self.id_to_label = dataset_info['id_to_label']

            # Preprocess ground truth - prepare overall class statistics
            if self.multiclass:
                self.data = self.data.assign(
                    y_true=self.data['tags'].apply(
                        lambda x: self.label_to_id.get(x),
                    ),
                )
            else:
                tags_to_vector_func = partial(
                    tags_to_vector, label_to_id=self.label_to_id, n_labels=self.num_classes,
                )
                self.data = self.data.assign(
                    y_true=self.data['tags'].parallel_apply(
                        tags_to_vector_func,
                    ),
                )

            # Train-Dev split - with stratify for class imbalance
            if 'set' in self.data.columns:
                self._train_df = self.data[self.data['set'] == 'train']
                self._dev_df = self.data[self.data['set'] == 'dev']
                print(
                    f'dataset is already splited, using the provided split: train {len(self._train_df)}, dev {len(self._dev_df)}',
                )
                # Ignoring the test set during training
            elif self.multiclass:
                self._train_df, self._dev_df = train_test_split(
                    self.data,
                    test_size=dev_size,
                    random_state=random_state,
                    stratify=self.data['y_true'].to_list(),
                )
            else:
                X = self.data.to_numpy()
                y = np.array(self.data['y_true'].to_list())
                X_train, y_train, X_dev, y_dev = iterative_train_test_split(
                    X, y, test_size=dev_size,
                )
                self._train_df = pd.DataFrame(
                    data=X_train, columns=self.data.columns,
                )
                self._dev_df = pd.DataFrame(
                    data=X_dev, columns=self.data.columns,
                )

        else:
            self.num_classes = reuse_data.num_classes
            self.label_to_id = reuse_data.label_to_id
            self.id_to_label = reuse_data.id_to_label
            self._train_df = reuse_data._train_df
            self._dev_df = reuse_data._dev_df

        # Assign split data and pre-calculate class n_samples_per_class
        if train:
            self.data = self._train_df
        else:
            self.data = self._dev_df
        self.data.reset_index(drop=True, inplace=True)

        # Build Transform

        def pad_same_length(img: torch.Tensor):
            h, w = img.size()[-2:]
            longest = max(h, w)
            h_diff, w_diff = longest - h, longest - w
            pad_h = h_diff // 2
            pad_w = w_diff // 2
            return nn.functional.pad(img, [pad_w, w_diff - pad_w, pad_h, h_diff - pad_h])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(pad_same_length),
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __getitem__(self, index):
        row = self.data.loc[index]
        gif_id = row['child_gif_id']

        # Select 4 frames evenly
        frames = read_gif_id(gif_id, frame_reduce_fn=select_4_frames)
        frames = [Image.fromarray(frame) for frame in frames]
        frames = [self.transform(frame) for frame in frames]
        assert len(frames) == 4

        if self.with_frame_seq:
            X = torch.stack(frames)
        else:
            # Concatenate 4 frames into one
            # concatenate on width direction
            w1 = torch.cat((frames[0], frames[1]), axis=2)
            w2 = torch.cat((frames[2], frames[3]), axis=2)
            # concatenate on height direction
            X = torch.cat((w1, w2), axis=1)

        # a scaler when multiclass, a vector when multilabel
        y_true = row['y_true']

        if self.multiclass:
            y_true = torch.Tensor(y_true).long()
        else:
            y_true = torch.Tensor(y_true).float()

        if self.test:
            return X
        else:
            return X, y_true

    def __len__(self):
        """
        Returns:
            int: number of elements in current dataset (train/dev).
        """
        return len(self.data)
