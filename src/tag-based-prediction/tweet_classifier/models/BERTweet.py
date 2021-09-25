import torch
from transformers import AutoModel


class BERTweetModel(torch.nn.Module):
    '''
    Paper: https://www.aclweb.org/anthology/2020.emnlp-demos.2.pdf
    Github: https://github.com/VinAIResearch/BERTweet
    self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
    self.tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    '''

    def __init__(self, num_classes, **kwargs):
        super().__init__()
        self._num_classes = num_classes
        self._hidden_size = 768

        self.bertweet = AutoModel.from_pretrained('vinai/bertweet-base')
        self.classifier = torch.nn.Linear(self._hidden_size, num_classes)
        torch.nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids):
        """Forward pass of model

        Args:
            input_ids : list of preprocessed (normalized) input ids.
        Returns:
            [type]: [description]
        """
        features = self.bertweet(input_ids)[1]  # Models outputs are now tuples
        y_pred = self.classifier(features)
        return y_pred
