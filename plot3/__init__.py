"""Isometric projection of 3D points for plotting"""

__version__ = "0.0.0a2"

__all__ = ["projection", "plotting", "shapes"]

from .projection import project
from .plotting import scatter, plot
from .shapes import *
