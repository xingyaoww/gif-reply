import ast
import itertools
from functools import partial

import numpy as np
import pandas as pd
import torch
from pandarallel import pandarallel
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split
from transformers import AutoTokenizer

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


class GifReplyDataset(torch.utils.data.Dataset):

    # Init tokenizer
    tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')

    def __init__(
        self,
        dataset_path,
        metadata_path,
        train=True,
        test=False,
        dev_size=0.05,
        multiclass=False,
        max_seq_length=128,
        random_state=42,
        reuse_data=None,
        **kwargs,
    ):
        """
        Initialize GifReply dataset and load relavant information.
        Args:
            dataset_path ([type]): filepath to reply dataset file.
            metadata_path ([type]): filepath to metadata file.
            train (bool, optional): when true, __getitem__ return elements in train set;
                otherwise returns elements in dev set. Defaults to True.
            test (bool, optional): Whether in test mode, when true, __getitem__ only return
                the model_input without label; otherwise return (X, y_true) pair. Defaults to False.
            dev_size (float, optional): porpotion of data to set to dev set. Defaults to 0.05.
            multiclass (bool, optional): when true, the model will runs in multiclass mode,
                y_true returned by __getitem__ will be an integer value denotes the class of the sample;
                otherwise it will returns an hot_vector encoding multiple label in y_true. Defaults to False.
            max_seq_length (int, optional): default be 128 for BERTweet model, refer to "2.Bertweet Optimization" section in paper.
            reuse_data (data.Dataset): another instance of GifReplyDataset to prevent load and preprocess same dataset twice
        """
        # Store arguments
        self.train = train
        self.test = test
        self.multiclass = multiclass
        self.max_seq_length = max_seq_length

        if not reuse_data:
            # Load dataset
            self.data, dataset_info = load_dataset(dataset_path, metadata_path)
            self.num_classes = dataset_info['n_labels']
            self.label_to_id = dataset_info['label_to_id']
            self.id_to_label = dataset_info['id_to_label']

            tags_to_vector_func = partial(
                tags_to_vector, label_to_id=self.label_to_id, n_labels=self.num_classes,
            )
            # remove empty tags
            _before = len(self.data)
            self.data = self.data[
                self.data['tags'].apply(
                    lambda x: len(x) > 0,
                )
            ]
            _after = len(self.data)
            print(
                f'Removed {_after - _before} entries with empty tags from dataset. Before {_before}, After {_after}',
            )

            self.data = self.data.assign(
                y_true=self.data['tags'].parallel_apply(tags_to_vector_func),
            )
            self.n_samples_per_class_overall = self.multilabel_class_stats(
                self.data,
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
            self.n_samples_per_class_overall = reuse_data.n_samples_per_class_overall

        # Assign split data and pre-calculate class n_samples_per_class
        if train:
            self.data = self._train_df
        else:
            self.data = self._dev_df
        self.data.reset_index(drop=True, inplace=True)

    def __getitem__(self, index):
        row = self.data.loc[index]
        X = row['parent_text']
        X = self.tokenizer.encode(
            X,
            max_length=self.max_seq_length,
            truncation=True,
        )
        X = torch.Tensor(X).long()
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
