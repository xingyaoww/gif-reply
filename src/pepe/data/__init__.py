import torch
from torch.nn.utils.rnn import pad_sequence

from .dataset import GIFFeatureInferenceDataset
from .dataset import GifReplyOSCARDataset


def pad_OSCAR_sequence(X):
    """ collate_fn of DataLoader.
    Pad list of ((input_ids, gif_inputs), ...) to the same length (max length in this batch).
    where gif_inputs gif_inputs = (gif_token_ids, attention_mask, segment_ids, img_feat)
    # NOTE: all content of gif_inputs are padded to self.max_seq_length in Dataset

     - with automatic batching:
        for indices in batch_sampler:
            yield collate_fn([dataset[i] for i in indices])
    """
    input_ids, gif_inputs, gif_ids = list(zip(*X))
    _batchsize = len(input_ids)
    input_ids = pad_sequence(input_ids, batch_first=True)

    gif_token_ids, attention_mask, segment_ids, img_feat \
        = list(zip(*gif_inputs))

    gif_token_ids = torch.stack(gif_token_ids)
    attention_mask = torch.stack(attention_mask)
    segment_ids = torch.stack(segment_ids)
    img_feat = torch.stack(img_feat)
    gif_ids = list(gif_ids)

    gif_inputs = (gif_token_ids, attention_mask, segment_ids, img_feat)

    y_true = torch.tensor(range(_batchsize), dtype=torch.long)
    return (input_ids, gif_inputs, gif_ids), y_true


COLLATE_FUNC_MAP = {
    'OscarCLIPModel': pad_OSCAR_sequence,
}


def get_collate_fn(model_name):
    if model_name not in COLLATE_FUNC_MAP:
        return None
    else:
        return COLLATE_FUNC_MAP[model_name]
