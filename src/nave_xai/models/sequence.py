# SPDX-License-Identifier: MIT
"""Encoders of sequences (time series, text): each layer's output as (C, t) feature maps.

A sequence backbone takes one input and returns {layer name: (C_b, t_b) array}, where t_b runs
along the sequence (time steps, tokens), possibly downsampled. Optionally it also returns the
model's class scores under the key "logits".
"""
import numpy as np


class SequenceBackbone:
    """Base class. `blocks`: the layers that can be chosen, in order; `multiple`: inputs are padded
    to a length that is a multiple of this (the model's total downsampling)."""

    blocks = []
    multiple = 1

    def __call__(self, x, names):
        raise NotImplementedError


class FunctionSequenceBackbone(SequenceBackbone):
    """Any function: fn(x) -> {name: (C, t) or (1, C, t) array, ..., optional "logits"}.

    For series, x is a float32 (1, C, T) array; for text, a list of tokens."""

    def __init__(self, fn, blocks, multiple=1):
        self.fn, self.blocks, self.multiple = fn, list(blocks), multiple

    def __call__(self, x, names):
        out = self.fn(x)
        res = {}
        for n in list(names) + (["logits"] if "logits" in out else []):
            a = np.asarray(out[n], np.float32)
            res[n] = a[0] if (a.ndim == 3 and n != "logits") or (a.ndim == 2 and n == "logits") else a
        return res


class TorchSequenceBackbone(SequenceBackbone):
    """A PyTorch model. Layers are module names; their outputs are captured with forward hooks.

    channels_last : False for convolutional layers ((N, C, t), default), True for transformer
        layers ((N, t, C)); prefix_tokens : leading tokens to drop (class tokens)
    The model's own output is used as class scores when it is 2D (N, classes).
    """

    def __init__(self, model, multiple=1, channels_last=False, prefix_tokens=0, device="cpu"):
        import torch
        self.torch, self.model, self.device = torch, model.eval().to(device), device
        self.multiple, self.channels_last, self.prefix = multiple, channels_last, prefix_tokens
        self.blocks = [n for n, _ in model.named_modules() if n]
        self._modules = dict(model.named_modules())

    def __call__(self, x, names):
        torch = self.torch
        captured, hooks = {}, []
        for n in names:
            if n not in self._modules:
                raise ValueError(f"No module named {n!r} in this model.")
            hooks.append(self._modules[n].register_forward_hook(
                lambda m, i, o, n=n: captured.__setitem__(n, o[0] if isinstance(o, (tuple, list)) else o)))
        try:
            with torch.no_grad():
                out = self.model(torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device))
        finally:
            for h in hooks:
                h.remove()
        res = {}
        for n in names:
            a = captured[n].detach().float().cpu().numpy()[0]
            if self.channels_last:
                a = a[self.prefix:].T
            res[n] = a
        if hasattr(out, "dim") and out.dim() == 2:
            res["logits"] = out.detach().float().cpu().numpy()[0]
        return res


class OnnxSequenceBackbone(SequenceBackbone):
    """An ONNX file whose first input takes (1, C, T) and whose named outputs are (1, C_b, t_b)."""

    def __init__(self, path, blocks, multiple=1, logits=None):
        import onnxruntime as ort
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.blocks, self.multiple, self.logits_name = list(blocks), multiple, logits

    def __call__(self, x, names):
        wanted = list(names) + ([self.logits_name] if self.logits_name else [])
        outs = self.session.run(wanted, {self.input: np.asarray(x, np.float32)})
        res = {n: o[0] for n, o in zip(names, outs)}
        if self.logits_name:
            res["logits"] = outs[-1][0]
        return res


def as_sequence_backbone(model, layers=None, multiple=1, device="cpu", channels_last=False):
    """A SequenceBackbone from a backbone, a torch.nn.Module, an .onnx path or a function."""
    if isinstance(model, SequenceBackbone):
        return model
    if hasattr(model, "named_modules") and hasattr(model, "forward"):
        return TorchSequenceBackbone(model, multiple, channels_last, device=device)
    if isinstance(model, str) and model.endswith(".onnx"):
        if not layers:
            raise ValueError("Name the ONNX outputs to use with layers=.")
        return OnnxSequenceBackbone(model, layers, multiple)
    if callable(model):
        if not layers:
            raise ValueError("Name the layers your function returns with layers=.")
        return FunctionSequenceBackbone(model, layers, multiple)
    raise TypeError("model must be a torch.nn.Module, an .onnx path, a function or a SequenceBackbone.")
