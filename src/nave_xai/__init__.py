# SPDX-License-Identifier: MIT
"""nave-xai: NAVE explanations of vision encoders.

NAVE (Neuro-Activated Vision Explanations) extracts concepts from a frozen vision encoder's
feature activations at several depths and shows how the encoder decomposes an image. This is
the official package of

    Ahcene Boubekki, Samuel G. Fadel, Sebastian Mair.
    Multi-Depth Concept Extraction for Post-Hoc Vision Encoder Explanation.
    arXiv:2411.19700

Please cite the paper if you use this package in your work.

    >>> from nave_xai import NAVEImage
    >>> explainer = NAVEImage()
    >>> exp = explainer.explain(image)
    >>> exp.explanation          # (H, W) concept of every pixel
"""
from . import models
from .memory import NaveMemoryWarning
from .image import ImageExplanation, NAVEImage
from .series import NAVESeries, SeriesExplanation
from .text import NAVEText, TextExplanation

__version__ = "0.1.0"

__all__ = ["NAVEImage", "ImageExplanation", "NAVESeries", "SeriesExplanation", "NAVEText", "TextExplanation",
           "models", "NaveMemoryWarning", "__version__"]
