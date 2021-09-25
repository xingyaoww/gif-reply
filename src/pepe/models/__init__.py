"""Models
"""
from .CLIP import CLIPModel
from .metrics import CLIPMetrics
from .OscarCLIP import OscarCLIPModel

METRIC_MAP = {
    'CLIPModel': CLIPMetrics,
    'OscarCLIPModel': CLIPMetrics,
}


def get_metric_class(model_name):
    """Return the metric class for given model_name
    """
    return METRIC_MAP[model_name]
