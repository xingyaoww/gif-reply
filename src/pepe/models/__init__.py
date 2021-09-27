"""Models
"""
from .CLIP import CLIPModel
from .metrics import CLIPMetrics
from .PEPE import PEPEModel

METRIC_MAP = {
    'CLIPModel': CLIPMetrics,
    'PEPEModel': CLIPMetrics,
}


def get_metric_class(model_name):
    """Return the metric class for given model_name
    """
    return METRIC_MAP[model_name]
