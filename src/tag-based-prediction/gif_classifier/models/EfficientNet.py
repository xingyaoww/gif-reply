import torch
from efficientnet_pytorch import EfficientNet
from torch import nn


class EfficientNetGifSeqModel(nn.Module):
    '''
    Github: https://github.com/lukemelas/EfficientNet-PyTorch
    '''

    def __init__(self, num_classes, use_seq_processor=True):
        super().__init__()
        self._num_classes = num_classes
        self._hidden_size = 1000  # output of EfficientNet
        self._feature_size = 256
        self._n_frames = 4
        self.use_seq_processor = use_seq_processor

        self.model = EfficientNet.from_pretrained(
            'efficientnet-b0',
        )

        if self.use_seq_processor:
            self.seq_processor = nn.TransformerEncoderLayer(
                d_model=self._hidden_size, nhead=8,
            )
        self.feature_encoder = nn.Linear(
            self._n_frames * self._hidden_size,
            self._feature_size,
        )
        self.classifier = nn.Linear(self._feature_size, num_classes)
        nn.init.xavier_normal_(self.classifier.weight)

    def extract_feature(self, inputs):
        """Forward pass of model

        Args:
            inputs : batched tensor of preprocessed image frames (4 frames).
                        tensor should have size of [N, self._n_frames, C, H, W] (N: batchsize)
        Returns:
            y_pred: tensor of shape [N, self._num_classes]
        """
        batchsize, n_frames, C, H, W = inputs.size()
        inputs = inputs.view(batchsize*n_frames, C, H, W)
        outputs = self.model(inputs)
        outputs = outputs.view(batchsize, n_frames, self._hidden_size)

        if self.use_seq_processor:
            outputs = torch.transpose(outputs, 0, 1)
            # assert outputs.size() == (n_frames, batchsize, self._hidden_size)
            outputs = self.seq_processor(outputs)
            # assert outputs.size() == (n_frames, batchsize, self._hidden_size)
            outputs = torch.transpose(outputs, 0, 1)

        outputs = outputs.flatten(start_dim=1)
        # assert outputs.size() == (batchsize, self._hidden_size * n_frames)
        outputs = self.feature_encoder(outputs)
        # assert outputs.size() == (batchsize, self._feature_size)
        return outputs

    def forward(self, inputs):
        """Forward pass of model

        Args:
            inputs : batched tensor of preprocessed image frames (4 frames).
                        tensor should have size of [N, self._n_frames, C, H, W] (N: batchsize)
        Returns:
            y_pred: tensor of shape [N, self._num_classes]
        """
        features = self.extract_feature(
            inputs,
        )  # with shape [N, self._n_frames * self._hidden_size]
        y_pred = self.classifier(features)
        return y_pred
