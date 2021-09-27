"""Models
"""
from .metrics import CLIPMetrics
from .PEPE import PEPEModel

METRIC_MAP = {
    'PEPEModel': CLIPMetrics,
}


def get_metric_class(model_name):
    """Return the metric class for given model_name
    """
    return METRIC_MAP[model_name]
