import torch
from torch import nn
from transformers import AutoModel
from .modeling_bert import BertImgModel
from pytorch_transformers import BertConfig

class TweetEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        self._hidden_size = 768
        self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
        self.linear_transform = nn.Linear(self._hidden_size, output_dim)
        nn.init.xavier_normal_(self.linear_transform.weight)

    def forward(self, tweet_input_ids):
        # Models outputs are now tuples
        features = self.bertweet(tweet_input_ids)[1]
        return self.linear_transform(features)



class OscarGIFEncoder(nn.Module):
    def __init__(self, oscar_pretrained_model_dir, image_feature_size=512, n_frames=4):
        super().__init__()
        self._n_frames = n_frames
        self.image_feature_size = image_feature_size

        self.config = BertConfig.from_pretrained(oscar_pretrained_model_dir)
        # Use OSCAR as underlying image encoder
        self.bert = BertImgModel.from_pretrained(
            oscar_pretrained_model_dir, config=self.config)

        self.linear = nn.Linear(
            self.config.hidden_size, self.image_feature_size)

    def init_code_embedding(self, em):
        self.bert.code_embeddings.weight.data = em.clone()

    def forward(self, gif_inputs):
        input_ids, attention_mask, token_type_ids, img_feats = gif_inputs
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, img_feats=img_feats)
        pooled_output = outputs[1]
        logits = self.linear(pooled_output)
        return logits


class OscarCLIPModel(nn.Module):
    '''
    Paper: https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf
    Github: https://github.com/openai/CLIP

    Args:
        tau (float): scalar coefficient before applying softmax
    '''

    def __init__(self, oscar_pretrained_model_dir, n_frames=4):
        super().__init__()
        self.gif_encoder = OscarGIFEncoder(
            oscar_pretrained_model_dir,
            image_feature_size=512, 
            n_frames=n_frames
        )
        self.tweet_encoder = TweetEncoder(output_dim=512)
        # initialize learnable temperature to 0.07 as mentioned in the paper
        # "as a log-parameterized multiplicative scalar"
        self.log_of_tau = nn.Parameter(torch.tensor(0.07), requires_grad=True)
        self.log_of_tau_max = nn.Parameter(torch.log(
            torch.tensor(100.0)), requires_grad=False)

    def extract_gif_feature(self, gif_inputs):
        return self.gif_encoder(gif_inputs)

    def extract_tweet_feature(self, tweet_input_ids):
        return self.tweet_encoder(tweet_input_ids)

    def calculate_score(self, tweet_features, gif_features, include_softmax=False):
        """Calculate probabilty from gif_features and tweet_features.

        Args:
            tweet_features (torch.Tensor): shape of [num tweets, 512 (feature size)]
            gif_features (torch.Tensor): [num GIFs, 512 (feature size)]

        Returns:
            torch.Tensor: probablity of each gif for each given tweet
                          shape of [num tweet, num GIFs]
        """
        # L2 Normalization
        gif_features_norm = gif_features.norm(dim=-1, keepdim=True)
        tweet_features_norm = tweet_features.norm(dim=-1, keepdim=True)
        gif_features = gif_features / gif_features_norm
        tweet_features = tweet_features / tweet_features_norm
        # Similarity - shape of batchsize x batchsize similarity
        similarity = torch.matmul(tweet_features, gif_features.T)

        # Calculate tau via exponentiation, clip tau if necessary
        tau = torch.min(self.log_of_tau, self.log_of_tau_max).exp()
        # softmax to normalize probs for each tweet to match potential gif
        score = tau * similarity
        if include_softmax:
            return score.softmax(dim=-1)
        else:
            return score

    def forward(self, tweet_input_ids, gif_inputs, include_softmax=False):
        """Forward pass of model

        Args:
            input_ids : list of preprocessed (normalized) input ids.
        Returns:
            list: list of Tensors [regression feature, classification logits]
        """
        gif_features = self.extract_gif_feature(gif_inputs)
        tweet_features = self.extract_tweet_feature(tweet_input_ids)
        score = self.calculate_score(
            gif_features, tweet_features, include_softmax)
        # NOTE: the y_true would just be an identity matrix with shape [batchsize, batchsize]
        return score
