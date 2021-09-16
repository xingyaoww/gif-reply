import torch
from .dataset import (
    GifReplyDataset,
    GIFFeatureInferenceDataset
)
from torch.nn.utils.rnn import pad_sequence

CLIP_CONTEXT_LENGTH = 77


def pad_batch_sequence(X):
    """ collate_fn of DataLoader.
    Pad list of ((input_ids, gif, gif_id), ...) to the same length (max length in this batch).

     - with automatic batching:
        for indices in batch_sampler:
            yield collate_fn([dataset[i] for i in indices])
    """
    input_ids, gifs, gif_ids = list(zip(*X))
    _batchsize = len(input_ids)
    input_ids = pad_sequence(input_ids, batch_first=True)
    gifs = torch.stack(gifs)
    y_true = torch.tensor(range(_batchsize), dtype=torch.long)
    return (input_ids, gifs, gif_ids), y_true


COLLATE_FUNC_MAP = {
    'CLIPModel': pad_batch_sequence
}


def get_collate_fn(model_name):
    if model_name not in COLLATE_FUNC_MAP:
        return None
    else:
        return COLLATE_FUNC_MAP[model_name]
