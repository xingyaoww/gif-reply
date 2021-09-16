import os
import numpy as np
import pandas as pd
import pickle
import torch
import itertools
from PIL import Image
from torch.utils import data
from torch import nn
from sklearn.model_selection import train_test_split
from functools import partial
from transformers import AutoTokenizer
from collections import Counter
from pandarallel import pandarallel
from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split
from utils import load_dataset, save_dataset, load_gif
from pytorch_transformers import BertTokenizer, BertConfig
pandarallel.initialize()


class GifReplyOSCARDataset(data.Dataset):

    def __init__(self, dataset_path,
                 gif_feature_path,
                 train=True,
                 test=False,
                 dev_size=0.05,
                 random_state=42,
                 reuse_data=None,
                 oscar_pretrained_model_dir=None,
                 ):
        # Init tokenizer
        tweet_tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
        gif_tokenizer = BertTokenizer.from_pretrained(oscar_pretrained_model_dir)
        gif_tokenizer.vocab.pop('[unused0]')
        gif_tokenizer.vocab.pop('[unused1]')
        gif_tokenizer.vocab['[INTER_FRAME_SEP]'] = 1
        gif_tokenizer.vocab['[INNER_FRAME_SEP]'] = 2

        # Store arguments
        self.train = train
        self.test = test
        self.bertweet_max_seq_len = 128  # default for bertweet
        self.max_seq_len = 256 # max_seq_length for OSCAR
        self.max_img_seq_len = 40  # at most 10 roi per frame, total 4 frames

        if not reuse_data:
            # Load dataset
            self.data, dataset_info = load_dataset(dataset_path)
            if 'set' in self.data.columns:
                self._train_df = self.data[self.data['set'] == 'train']
                self._dev_df = self.data[self.data['set'] == 'dev']
                print(
                    f"dataset is already splited, using the provided split: train {len(self._train_df)}, dev {len(self._dev_df)}")
                # Ignoring the test set during training
            else:
                self._train_df, self._dev_df = train_test_split(
                    self.data,
                    test_size=dev_size,
                    random_state=random_state,
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

        self.gif_features = pd.read_pickle(
            gif_feature_path).set_index("child_gif_id")

    def __len__(self):
        """
        Returns:
            int: number of elements in current dataset (train/dev).
        """
        return len(self.data)

    def get_gif_features(self, gif_id):
        row = self.gif_features.loc[gif_id]
        # text_a: caption
        text_a = row["ocr_results"].replace("[INNER_FRAME_SEP]", "")
        img_feat = torch.Tensor(np.vstack(row["roi_feature"]))
        # text_b: labels/tags
        text_b = row["roi_labels"]
        # remove sep for tokens
        # text_a = text_a.replace("[INTER_FRAME_SEP]", "")
        # text_b = text_b.replace("[INTER_FRAME_SEP]", "")
        return text_a, img_feat, text_b

    def tensorize_gif_example(self, gif_id,
                              cls_token_segment_id=0, pad_token_segment_id=0,
                              sequence_a_segment_id=0, sequence_b_segment_id=1):
        """Code modified from tensorize_example in run_retrieval.py from OSCAR repo."""

        text_a, img_feat, text_b = self.get_gif_features(gif_id)
        reserve_space_for_text_b = 10

        tokens_a = self.gif_tokenizer.tokenize(text_a)
        if len(tokens_a) > self.max_seq_len - 2 - reserve_space_for_text_b:
            tokens_a = tokens_a[:(self.max_seq_len - 2 -
                                  reserve_space_for_text_b)]

        tokens = [self.gif_tokenizer.cls_token] + \
            tokens_a + [self.gif_tokenizer.sep_token]

        segment_ids = [cls_token_segment_id] + \
            [sequence_a_segment_id] * (len(tokens_a) + 1)
        seq_a_len = len(tokens)
        if text_b and self.max_seq_len > seq_a_len:  # check to avoid -1 index for tokens_b
            tokens_b = self.gif_tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]
            tokens += tokens_b + [self.gif_tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        seq_len = len(tokens)
        seq_padding_len = self.max_seq_len - seq_len
        tokens += [self.gif_tokenizer.pad_token] * seq_padding_len
        segment_ids += [pad_token_segment_id] * seq_padding_len
        input_ids = self.gif_tokenizer.convert_tokens_to_ids(tokens)

        # image features
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0: self.max_img_seq_len, :]
            img_len = img_feat.shape[0]
            img_padding_len = 0
        else:
            img_padding_len = self.max_img_seq_len - img_len
            padding_matrix = torch.zeros((img_padding_len, img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        # generate attention_mask
        att_mask_type = "CLR"  # self.args.att_mask_type
        #  parser.add_argument("--att_mask_type", default='CLR', type=str,
        #                 help="attention mask type, support ['CL', 'CR', 'LR', 'CLR']"
        #                 "C: caption, L: labels, R: image regions; CLR is full attention by default."
        #                 "CL means attention between caption and labels."
        #                 "please pay attention to the order CLR, which is the default concat order.")
        if att_mask_type == "CLR":
            attention_mask = [1] * seq_len + [0] * seq_padding_len + \
                             [1] * img_len + [0] * img_padding_len
        else:
            # use 2D mask to represent the attention
            max_len = self.max_seq_len + self.max_img_seq_len
            attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
            # full attention of C-C, L-L, R-R
            c_start, c_end = 0, seq_a_len
            l_start, l_end = seq_a_len, seq_len
            r_start, r_end = self.max_seq_len, self.max_seq_len + img_len
            attention_mask[c_start: c_end, c_start: c_end] = 1
            attention_mask[l_start: l_end, l_start: l_end] = 1
            attention_mask[r_start: r_end, r_start: r_end] = 1
            if att_mask_type == 'CL':
                attention_mask[c_start: c_end, l_start: l_end] = 1
                attention_mask[l_start: l_end, c_start: c_end] = 1
            elif att_mask_type == 'CR':
                attention_mask[c_start: c_end, r_start: r_end] = 1
                attention_mask[r_start: r_end, c_start: c_end] = 1
            elif att_mask_type == 'LR':
                attention_mask[l_start: l_end, r_start: r_end] = 1
                attention_mask[r_start: r_end, l_start: l_end] = 1
            else:
                raise ValueError(
                    "Unsupported attention mask type {}".format(att_mask_type))

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        return (input_ids, attention_mask, segment_ids, img_feat)

    def __getitem__(self, index):
        row = self.data.loc[index]

        # Text Data
        _tweet = row['parent_text']
        # truncate query length
        _tweet_ids = self.tweet_tokenizer.encode(_tweet,
                                                 max_length=self.bertweet_max_seq_len,
                                                 truncation=True
                                                 )
        tweet_ids = torch.Tensor(_tweet_ids).long()

        # GIF Data
        gif_id = row['child_gif_id']
        gif_token_ids, attention_mask, segment_ids, img_feat\
            = self.tensorize_gif_example(gif_id)
        gif_inputs = (gif_token_ids, attention_mask, segment_ids, img_feat)
        return tweet_ids, gif_inputs, gif_id


class GIFFeatureInferenceDataset(GifReplyOSCARDataset):
    def __init__(self, gif_ids, train_dataset):
        self.gif_ids = gif_ids
        self.dataset = train_dataset

    def __len__(self):
        return len(self.gif_ids)

    def __getitem__(self, index):
        gif_id = self.gif_ids[index]
        gif_token_ids, attention_mask, segment_ids, img_feat\
            = self.dataset.tensorize_gif_example(gif_id)
        gif_inputs = (gif_token_ids, attention_mask, segment_ids, img_feat)
        return gif_inputs, gif_id
