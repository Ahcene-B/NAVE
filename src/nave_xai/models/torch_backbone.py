# SPDX-License-Identifier: MIT
"""PyTorch models as backbones: block outputs are captured with forward hooks.

Blocks are module names as given by `model.named_modules()` (e.g. "layer3" for a
torchvision ResNet, "blocks.11" for a timm ViT, "encoder.layer.11" for Hugging Face).
Outputs may be feature maps (N, C, h, w) or token sequences (N, L, C); tokens are
reshaped to the patch grid after dropping class/register tokens.
"""
import numpy as np

from .backbone import IMAGENET_MEAN, IMAGENET_STD, Backbone


def _first_tensor(out):
    """Module outputs can be tuples or ModelOutput objects (Hugging Face): take the first tensor."""
    import torch
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, dict):
        out = list(out.values())
    if isinstance(out, (tuple, list)):
        for o in out:
            if isinstance(o, torch.Tensor):
                return o
    raise TypeError(f"Cannot read a tensor from a module output of type {type(out).__name__}.")


class TorchBackbone(Backbone):
    """Wrap a `torch.nn.Module`.

    Parameters
    ----------
    module : the model (put in eval mode; gradients are not tracked)
    mean, std : input normalisation (default: ImageNet)
    multiple : input sides must be multiples of this (e.g. 32 for CNNs, the patch size for ViTs)
    fixed_size : (height, width) if the model only accepts one input size
    patch_size : for token outputs; default: found from the number of tokens
    prefix_tokens : class/register tokens before the patch tokens; default: found automatically
    classifier : classifier for saliency: a module name (e.g. "fc", "head"), a torch.nn.Linear,
        or None to look for a Linear named fc / head / classifier
    final_norm : module name of the norm applied before the classifier (e.g. "norm"), if any
    device : "cpu", "cuda", ...
    """

    def __init__(self, module, mean=IMAGENET_MEAN, std=IMAGENET_STD, multiple=32, fixed_size=None,
                 patch_size=None, prefix_tokens=None, classifier=None, final_norm=None, device="cpu"):
        import torch
        self.torch = torch
        self.module = module.eval().to(device)
        self.device = device
        self.name = type(module).__name__
        self.modules = dict(module.named_modules())
        self.blocks = [n for n in self.modules if n]
        self.mean, self.std, self.multiple, self.fixed_size = mean, std, multiple, fixed_size
        self.patch_size, self.prefix_tokens = patch_size, prefix_tokens
        self._classifier, self._final_norm = classifier, final_norm
        self._last_output = None

    def resolve_blocks(self, blocks):
        if blocks is None:
            raise ValueError("Pass blocks= as module names for a PyTorch model, e.g. ['layer3', 'layer4'].")
        items = [blocks] if isinstance(blocks, (str, int)) else list(blocks)
        if any(isinstance(b, (int, np.integer)) for b in items):
            raise ValueError("Give PyTorch blocks as module names (see model.named_modules()), not numbers.")
        return super().resolve_blocks(items)

    def __call__(self, x, names):
        torch = self.torch
        captured, handles = {}, []
        for n in names:
            if n not in self.modules:
                raise ValueError(f"Module {n!r} not found in {self.name}.")
            handles.append(self.modules[n].register_forward_hook(
                lambda mod, inp, out, n=n: captured.__setitem__(n, _first_tensor(out).detach())))
        try:
            with torch.no_grad():
                out = self.module(torch.from_numpy(np.ascontiguousarray(x, np.float32)).to(self.device))
        finally:
            for h in handles:
                h.remove()
        try:
            self._last_output = _first_tensor(out).float().cpu().numpy()
        except TypeError:
            self._last_output = None
        return {n: self._to_map(captured[n], x.shape[2], x.shape[3]) for n in names}

    def _to_map(self, t, H, W):
        t = t.float().cpu().numpy()
        if t.ndim == 4:
            return t
        if t.ndim != 3:
            raise ValueError(f"Block output of shape {t.shape} is neither a feature map nor tokens.")
        n, L, C = t.shape
        gh, gw, prefix = self._token_grid(L, H, W)
        grid = t[:, prefix:prefix + gh * gw].reshape(n, gh, gw, C)
        return np.ascontiguousarray(grid.transpose(0, 3, 1, 2))

    def _token_grid(self, L, H, W):
        sizes = [self.patch_size] if self.patch_size else [16, 14, 8, 32, 4]
        for p in sizes:
            gh, gw = H // p, W // p
            prefix = L - gh * gw if self.prefix_tokens is None else self.prefix_tokens
            if gh and gw and 0 <= prefix <= 16 and L - prefix == gh * gw:
                return gh, gw, prefix
        raise ValueError(f"Cannot fit {L} tokens on a patch grid for a {H}x{W} input; "
                         "pass patch_size= and prefix_tokens= to TorchBackbone.")

    # -- classifier support --------------------------------------------------
    def _linear(self):
        torch = self.torch
        c = self._classifier
        if isinstance(c, torch.nn.Module):
            return c if isinstance(c, torch.nn.Linear) else None
        candidates = [c] if isinstance(c, str) else ["fc", "head", "classifier", "heads.head"]
        for name in candidates:
            mod = self.modules.get(name)
            if isinstance(mod, torch.nn.Sequential):
                mod = next((m for m in reversed(mod) if isinstance(m, torch.nn.Linear)), None)
            if isinstance(mod, torch.nn.Linear):
                return mod
        return None

    def head(self):
        lin = self._linear()
        if lin is None:
            return None
        bias = None if lin.bias is None else lin.bias.detach().float().cpu().numpy()
        return {"weight": lin.weight.detach().float().cpu().numpy(), "bias": bias}

    def head_features(self, feature):
        if not self._final_norm:
            return feature
        torch = self.torch
        norm = self.modules[self._final_norm]
        with torch.no_grad():
            t = torch.from_numpy(np.ascontiguousarray(feature.transpose(1, 2, 0))).to(self.device)
            t = norm(t).float().cpu().numpy()
        return np.ascontiguousarray(t.transpose(2, 0, 1))

    def logits(self, outputs, valid=None):
        out = self._last_output
        if out is not None and out.ndim == 2 and out.shape[0] == 1:
            return out[0]                                    # the model's own prediction
        return super().logits(outputs, valid)
