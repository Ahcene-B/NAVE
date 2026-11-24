# SPDX-License-Identifier: MIT
"""Backbones: anything that maps a normalised image to block feature maps.

A backbone is called with a float32 array (1, 3, H, W), already normalised with its
`mean` and `std`, and the names of the blocks wanted. It returns {name: (1, C, h, w)}.
"""
import os

import numpy as np

from .. import interp
from . import weights as W

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BUILTIN = {      # default layers: N34 for every ResNet, the last block for ViTs
    "vitt16": {"blocks": 12, "trainings": ("random", "imagenet"), "default": [12], "multiple": 16},
    "resnet18": {"blocks": 4, "trainings": ("random", "imagenet"), "default": [3, 4], "multiple": 32},
    "resnet50": {"blocks": 4, "trainings": ("random", "imagenet", "dino"), "default": [3, 4], "multiple": 32},
    "vits16": {"blocks": 12, "trainings": ("random", "imagenet", "dino"), "default": [12], "multiple": 16},
}


def parse_paper_notation(text, n_blocks):
    """'N234' -> [2, 3, 4];  'N:9-12' -> [9, 12] (as in the paper, '-' separates blocks)."""
    s = text.strip().upper()
    s = s[1:] if s.startswith("N") else s
    if s.startswith(":"):
        blocks = [int(p) for p in s[1:].replace(",", "-").split("-") if p]
    elif n_blocks <= 9:
        blocks = [int(c) for c in s if c.isdigit()]
    else:
        blocks = [int(p) for p in s.replace(",", "-").split("-") if p]
    return blocks


class Backbone:
    """Base class. Subclasses set `blocks` (names, shallow to deep) and implement __call__."""

    name = "custom"
    blocks = []
    default_blocks = None          # None: the deepest block
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    multiple = 32                  # input height and width must be multiples of this
    fixed_size = None              # (height, width) if the model only accepts one size
    extra_outputs = ()             # outputs needed for logits() besides the blocks

    def __call__(self, x, names):
        raise NotImplementedError

    def resolve_blocks(self, blocks):
        """Block names, shallowest first, from None, ints (1-based), names or paper notation."""
        if blocks is None:
            blocks = self.default_blocks or [len(self.blocks)]
        if isinstance(blocks, (str, int)):
            blocks = [blocks]
        names = []
        for b in blocks:
            if isinstance(b, str) and b.upper().startswith("N") and b not in self.blocks:
                names += [self._index_name(i) for i in parse_paper_notation(b, len(self.blocks))]
            elif isinstance(b, (int, np.integer)):
                names.append(self._index_name(int(b)))
            elif b in self.blocks or self._accepts(b):
                names.append(b)
            else:
                raise ValueError(f"Unknown block {b!r}. Available: {self.blocks[:12]}"
                                 f"{' ...' if len(self.blocks) > 12 else ''}")
        if not names:
            raise ValueError("Choose at least one block.")
        order = {n: i for i, n in enumerate(self.blocks)}
        order["b0"] = -1                                   # the input of block 1 comes first
        return sorted(dict.fromkeys(names), key=lambda n: order.get(n, len(order)))

    def _index_name(self, i):
        if not 1 <= i <= len(self.blocks):
            raise ValueError(f"Block {i} does not exist; this model has blocks 1-{len(self.blocks)}.")
        return self.blocks[i - 1]

    def _accepts(self, name):
        return False


    # -- classifier support (saliency) --------------------------------------
    def head(self):
        """{"weight": (classes, C), "bias": (classes,) or None} of the model's classifier, or None."""
        return None

    def head_features(self, feature):
        """Deepest block features (C, h, w) as the classifier sees them (e.g. after a final norm)."""
        return feature

    def logits(self, outputs, valid=None):
        """Class scores for the image, or None when unknown. `valid` = (rows, cols) of the
        deepest grid that cover the image (the rest is padding)."""
        head = self.head()
        if head is None:
            return None
        feat = self.head_features(outputs[self.blocks[-1]][0])
        if valid is not None:
            feat = feat[:, :valid[0], :valid[1]]
        pooled = feat.reshape(feat.shape[0], -1).mean(1)
        return head["weight"] @ pooled + (0 if head["bias"] is None else head["bias"])


def _layer_norm(x, weight, bias, eps=1e-6):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * weight + bias


def _session(path, names, values):
    """onnxruntime session with injected weights. If an onnxruntime version rejects the graph
    after its own optimisations (some versions do), retry with basic optimisations only."""
    import onnxruntime as ort

    def make(level):
        options = ort.SessionOptions()
        options.graph_optimization_level = level
        options.add_external_initializers(names, values)
        return ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
    try:
        return make(ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
    except Exception:
        return make(ort.GraphOptimizationLevel.ORT_ENABLE_BASIC)


class BuiltinBackbone(Backbone):
    """ResNet18, ResNet50 or ViT-S/16 run by onnxruntime with injected weights."""

    def _accepts(self, name):
        return _builtin_accepts(self, name)

    def __init__(self, model, training="imagenet", seed=0, progress=None):
        import onnxruntime as ort
        if model not in BUILTIN:
            raise ValueError(f"Unknown model {model!r}. Built-in models: {', '.join(BUILTIN)}.")
        info = BUILTIN[model]
        if training not in info["trainings"]:
            raise ValueError(f"No {training!r} weights for {model}; available: {', '.join(info['trainings'])}.")
        self.name, self.training = model, training
        self.blocks = [f"b{i}" for i in range(1, info["blocks"] + 1)]
        self.default_blocks = info["default"]
        self.multiple = info["multiple"]
        self.extra_outputs = ("cls",) if model.startswith("vit") else ()
        spec = W.load_spec(model)
        if training == "random":
            weights, self._head = W.random_weights(spec, seed), None
            dim = next((w["shape"][2] for w in spec["weights"] if w["name"] == "cls_token"), None)
            pos = W.random_pos_embed(seed, dim) if model.startswith("vit") else None
        else:
            weights, self._head = W.pretrained_weights(model, training, spec, progress)
            pos = weights.pop("pos_embed", None)
        self._pos, self._pos_cache = pos, {}
        self._weights = weights                         # must outlive the session
        names = list(weights)
        self._values = [ort.OrtValue.ortvalue_from_numpy(weights[n]) for n in names]
        self.session = _session(os.path.join(W.GRAPH_DIR, f"{model}.onnx"), names, self._values)

    def _pos_embed(self, gh, gw):
        """Position embedding resized (bicubic) to a gh x gw patch grid."""
        if (gh, gw) not in self._pos_cache:
            cls, grid = self._pos[:, :1], self._pos[0, 1:]
            g = int(round(np.sqrt(grid.shape[0])))
            grid = grid.reshape(g, g, -1)
            if (g, g) != (gh, gw):
                grid = interp.resample(grid, gh, gw, "cubic")
            self._pos_cache = {(gh, gw): np.ascontiguousarray(
                np.concatenate([cls, grid.reshape(1, gh * gw, -1)], axis=1), np.float32)}
        return self._pos_cache[(gh, gw)]

    def __call__(self, x, names):
        feeds = {"image": np.ascontiguousarray(x, np.float32)}
        if self._pos is not None:
            feeds["pos_embed"] = self._pos_embed(x.shape[2] // 16, x.shape[3] // 16)
        names = list(names)
        return dict(zip(names, self.session.run(names, feeds)))

    def head(self):
        return self._head

    def head_features(self, feature):
        head = self.head()
        if head is None or "norm_weight" not in head:
            return feature
        t = _layer_norm(feature.transpose(1, 2, 0), head["norm_weight"], head["norm_bias"])
        return np.ascontiguousarray(t.transpose(2, 0, 1))

    def logits(self, outputs, valid=None):
        head = self.head()
        if head is None:
            return None
        if "cls" in outputs and "norm_weight" in head:   # ViT: the classifier reads the class token
            cls = _layer_norm(outputs["cls"][0, 0], head["norm_weight"], head["norm_bias"])
            return head["weight"] @ cls + head["bias"]
        return super().logits(outputs, valid)


class OnnxBackbone(Backbone):
    """Any ONNX model whose outputs are block feature maps (1, C, h, w).

    Parameters
    ----------
    path : the .onnx file; its first input takes the normalised image (1, 3, H, W)
    blocks : output names to use as blocks, shallow to deep (default: all outputs)
    mean, std : input normalisation (default: ImageNet)
    multiple : input sides must be multiples of this
    fixed_size : (height, width) if the model only accepts one size
    """

    def __init__(self, path, blocks=None, mean=IMAGENET_MEAN, std=IMAGENET_STD, multiple=32, fixed_size=None):
        import onnxruntime as ort
        self.name = os.path.basename(path)
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.blocks = list(blocks) if blocks else [o.name for o in self.session.get_outputs()]
        self.mean, self.std, self.multiple, self.fixed_size = mean, std, multiple, fixed_size

    def __call__(self, x, names):
        names = list(names)
        return dict(zip(names, self.session.run(names, {self.input_name: np.ascontiguousarray(x, np.float32)})))


class FunctionBackbone(Backbone):
    """Wrap a function `fn(x) -> {name: (1, C, h, w)}` or a list of such arrays.

    Use it for any framework (JAX, TensorFlow, ...): convert inside `fn`.
    """

    def __init__(self, fn, blocks=None, mean=IMAGENET_MEAN, std=IMAGENET_STD, multiple=1, fixed_size=None):
        self.fn, self.name = fn, getattr(fn, "__name__", "function")
        self.blocks = list(blocks) if blocks else []
        self.mean, self.std, self.multiple, self.fixed_size = mean, std, multiple, fixed_size

    def _accepts(self, name):
        return not self.blocks

    def __call__(self, x, names):
        out = self.fn(x)
        if not isinstance(out, dict):
            out = {str(i): o for i, o in enumerate(out)}
        if not self.blocks:
            self.blocks = list(out)
        return {n: np.asarray(out[n], np.float32) for n in names}


def _builtin_accepts(self, name):
    """Built-in models also expose the input of block 1 ("b0") and, for ViTs, the keys ("k1", ...)."""
    return name == "b0" or (self.name.startswith("vit") and name[:1] == "k" and name[1:].isdigit())
