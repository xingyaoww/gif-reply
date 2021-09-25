"""Models"""
from .EfficientNet import EfficientNetGifSeqModel
from .metrics import (
    EfficientNetMetrics,
)

METRIC_MAP = {
    'EfficientNetGifSeqModel': EfficientNetMetrics,
}


def get_metric_class(model_name):
    """Return the metric class for given model_name
    """
    return METRIC_MAP[model_name]
