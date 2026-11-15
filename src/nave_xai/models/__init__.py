# SPDX-License-Identifier: MIT
"""Built-in models and wrappers for your own."""
import os

from .backbone import (BUILTIN, IMAGENET_MEAN, IMAGENET_STD, Backbone, BuiltinBackbone, FunctionBackbone,
                       OnnxBackbone)
from .weights import cache_dir

_CACHE = {}


def available():
    """Built-in models, their blocks and the trainings with weights.

    >>> nave.models.available()["vits16"]
    {'blocks': 12, 'trainings': ('random', 'imagenet', 'dino'), 'default_blocks': [12]}
    """
    return {name: {"blocks": info["blocks"], "trainings": info["trainings"], "default_blocks": info["default"]}
            for name, info in BUILTIN.items()}


def TorchBackbone(*args, **kwargs):            # noqa: N802 - imported lazily so torch stays optional
    from .torch_backbone import TorchBackbone as _TorchBackbone
    return _TorchBackbone(*args, **kwargs)


def get_backbone(model, training="imagenet", seed=0, device="cpu", mean=None, std=None, progress=None):
    """Turn the `model=` argument into a Backbone.

    Accepts a built-in name, a Backbone, a torch.nn.Module, a path to an .onnx file,
    or a function returning feature maps.
    """
    if isinstance(model, Backbone):
        backbone = model
    elif isinstance(model, str) and model in BUILTIN:
        key = (model, training, seed if training == "random" else None)
        if key not in _CACHE:
            if len(_CACHE) >= 2:                        # keep at most two built-in models loaded
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[key] = BuiltinBackbone(model, training, seed, progress)
        backbone = _CACHE[key]
    elif isinstance(model, str) and model.endswith(".onnx") and os.path.exists(model):
        backbone = OnnxBackbone(model)
    elif type(model).__module__.startswith("torch") or _is_torch_module(model):
        backbone = TorchBackbone(model, device=device)
    elif callable(model):
        backbone = FunctionBackbone(model)
    else:
        raise ValueError(f"Unknown model {model!r}. Built-in models: {', '.join(BUILTIN)}.")
    if mean is not None:
        backbone.mean = mean
    if std is not None:
        backbone.std = std
    return backbone


def _is_torch_module(obj):
    try:
        import torch
    except ImportError:
        return False
    return isinstance(obj, torch.nn.Module)


__all__ = ["available", "get_backbone", "Backbone", "BuiltinBackbone", "OnnxBackbone", "FunctionBackbone",
           "TorchBackbone", "IMAGENET_MEAN", "IMAGENET_STD", "cache_dir"]
