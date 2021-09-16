"""Models
"""
from .CLIP import CLIPModel
from .OscarCLIP import OscarCLIPModel
from .metrics import CLIPMetrics

METRIC_MAP = {
    "CLIPModel": CLIPMetrics,
    "OscarCLIPModel": CLIPMetrics,
}


def get_metric_class(model_name):
    """Return the metric class for given model_name
    """
    return METRIC_MAP[model_name]
