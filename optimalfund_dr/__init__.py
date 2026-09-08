"""Resumable diabetic retinopathy severity training with OT adaptation."""

from .config import Config, cfg
from .model import DRModel

__version__ = "0.1.0"
__all__ = ["Config", "DRModel", "cfg", "__version__"]
