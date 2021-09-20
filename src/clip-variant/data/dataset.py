import pandas as pd
import torch
from pandarallel import pandarallel
from transformers import AutoTokenizer
from utils import load_gif
pandarallel.initialize()


class GifReplyDataset(torch.utils.data.Dataset):

    # Init tokenizer
    tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')

    def __init__(
        self, dataset_path,
        train=True,
        test=False,
        reuse_data=None,
        max_seq_length=128,  # default for BERTweet
    ):

        # Store arguments
        self.train = train
        self.test = test
        self.max_seq_length = max_seq_length

        if not reuse_data:
            # Load dataset
            self.data = pd.read_csv(dataset_path)
            assert 'set' in self.data.columns
            self._train_df = self.data[self.data['set'] == 'train']
            self._dev_df = self.data[self.data['set'] == 'dev']
            print(
                f'dataset is already splited, using the provided split: train {len(self._train_df)}, dev {len(self._dev_df)}',
            )
        else:
            self._train_df = reuse_data._train_df
            self._dev_df = reuse_data._dev_df

        # Assign split data and pre-calculate class n_samples_per_class
        if train:
            self.data = self._train_df
        else:
            self.data = self._dev_df
        self.data.reset_index(drop=True, inplace=True)

    def __len__(self):
        """
        Returns:
            int: number of elements in current dataset (train/dev).
        """
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.loc[index]

        # Text Data
        _tweet = row['parent_text']
        # truncate query length
        _tweet_ids = self.tokenizer.encode(
            _tweet,
            max_length=self.max_seq_length,
            truncation=True,
        )
        tweet_ids = torch.Tensor(_tweet_ids).long()

        # GIF Data
        gif_id = row['child_gif_id']
        gif = load_gif(gif_id)

        return tweet_ids, gif, gif_id  # no ground truth in CLIP


class GIFFeatureInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, gif_ids: list):
        self.gif_ids = gif_ids

    def __len__(self):
        return len(self.gif_ids)

    def __getitem__(self, index):
        gif_id = self.gif_ids[index]
        X = load_gif(gif_id)
        return X, gif_id
