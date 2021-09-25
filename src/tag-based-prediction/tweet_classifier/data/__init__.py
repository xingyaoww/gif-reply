import torch
from torch.nn.utils.rnn import pad_sequence

from .dataset import (
    GifReplyDataset,
)


def pad_batch_sequence(X):
    """ collate_fn of DataLoader.
    Pad list of ((input_ids, label), ...) to the same length (max length in this batch).

     - with automatic batching:
        for indices in batch_sampler:
            yield collate_fn([dataset[i] for i in indices])
    """

    input_ids, labels = list(zip(*X))
    # input_ids -> list of torch.Tensor with not necessary the same shape
    # labels -> list of torch.Tensor with same shape
    input_ids = pad_sequence(input_ids, batch_first=True)
    labels = torch.stack(labels)
    return input_ids, labels


COLLATE_FUNC_MAP = {
    'BERTweetModel': pad_batch_sequence,
}


def get_collate_fn(model_name):
    if model_name not in COLLATE_FUNC_MAP:
        return None
    else:
        return COLLATE_FUNC_MAP[model_name]
