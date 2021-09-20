import numpy as np
import pandas as pd
import torch
from ignite.metrics import Accuracy
from ignite.metrics import Average
from ignite.metrics import MetricsLambda
from ignite.metrics import Precision
from ignite.metrics import Recall
from sklearn.neighbors import NearestNeighbors
from torch import nn


class BERTweetMetrics():
    def __init__(self, multiclass=True, weight=None, **kwargs):
        self.multiclass = multiclass
        self.weight = weight is not None

        # Initialize all metrics
        self.metrics = {}
        self.metrics['loss'] = Average()
        self.metrics['accuracy'] = Accuracy(is_multilabel=not multiclass)
        self.metrics['precision'] = Precision(
            average=False, is_multilabel=not multiclass,
        )
        self.metrics['recall'] = Recall(
            average=False, is_multilabel=not multiclass,
        )
        F1 = self.metrics['precision'] * self.metrics['recall'] * 2 / \
            (self.metrics['precision'] + self.metrics['recall'] + 1e-20)
        self.metrics['f1'] = MetricsLambda(lambda t: torch.mean(t).item(), F1)

        # Calculate weighted f1 if specified and when multiclass
        if self.multiclass and self.weight:
            F1 = F1 * (weight / weight.sum())
            self.metrics['weighted-f1'] = MetricsLambda(
                lambda t: torch.mean(t).item(), F1,
            )

    def reset(self):
        """Reset internal metrics.
        """
        for key in self.metrics:
            self.metrics[key].reset()

    def update(self, loss, y_pred, y_true):
        """Update internal metrics

        Args:
            loss ([type]):
            y_pred ([type]):
            y_true ([type]):
        """
        self.metrics['loss'].update(loss[0].item())
        if not self.multiclass:
            y_pred = torch.sigmoid(y_pred)
            y_pred = torch.round(y_pred)
            # Assign the y_pred & y_true to cpu for saving GPU memory
            # issue (ignite.Precision/Recall taking too much)
            y_pred = y_pred.cpu()
            y_true = y_true.cpu()
        for key in self.metrics:
            if key == 'loss':
                continue
            self.metrics[key].update((y_pred, y_true))

    def compute(self, **kwarg):
        """Compute and return the metrics.

        Returns:
            dict: a dict of metric results
        """
        result = {}
        result['loss'] = self.metrics['loss'].compute().item()
        result['accuracy'] = self.metrics['accuracy'].compute()
        result['precision'] = self.metrics['precision'].compute().mean().item()
        result['recall'] = self.metrics['recall'].compute().mean().item()
        result['f1'] = self.metrics['f1'].compute()
        if 'weighted-f1' in self.metrics:
            result['weighted-f1'] = self.metrics['weighted-f1'].compute()
        return result

    def log_tensorboard(self, writer, step, results=None, loss=None, train=True):
        """Compute and log the current metric to tensorboard.

        Args:
            writer ([type]): Writer for Tensorboard
            step ([type]): [description]
            loss ([float]): if not None, it will be the real time loss at `step`
            results ([dict]): if not None, it will be used as datasource to tensorboard,
                compute will not be called in this case.
            train (bool, optional): whether in training mode. Defaults to True.
        """
        results = self.compute() if results is None else results
        mode_str = 'train' if train else 'val'
        writer.add_scalar(
            'Loss/' + mode_str,
            results['loss'] if loss is None else loss[0].item(), step,
        )
        writer.add_scalar(
            'Accuracy/' + mode_str,
            results['accuracy'], step,
        )
        writer.add_scalar(
            'Precision/' + mode_str,
            results['precision'], step,
        )
        writer.add_scalar('Recall/' + mode_str, results['recall'], step)
        writer.add_scalar('F1/' + mode_str, results['f1'], step)
        if 'weighted-f1' in results:
            writer.add_scalar(
                'Weighted-F1/' + mode_str,
                results['weighted-f1'], step,
            )
        return results
