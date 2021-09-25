"""Models
"""
from .BERTweet import BERTweetModel
from .metrics import BERTweetMetrics

METRIC_MAP = {
    'BERTweetModel': BERTweetMetrics,
}


def get_metric_class(model_name):
    """Return the metric class for given model_name
    """
    return METRIC_MAP[model_name]
