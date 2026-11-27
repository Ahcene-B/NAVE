# SPDX-License-Identifier: MIT
"""Weights for the built-in models, without torch.

Pretrained weights come from (in this order) the nave-xai cache, the Hugging Face hub
cache, the torch hub cache, or a download. `.safetensors` files are parsed with numpy;
`.pth` files (official DINO and DeiT releases, torchvision) are read with a restricted
unpickler that accepts only plain tensors. Random weights use each architecture's usual
initialisation.
"""
import json
import math
import os
import pickle
import struct
import sys
import urllib.request
import zipfile
from collections import OrderedDict

import numpy as np

GRAPH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")

HF_REPOS = {
    ("vitt16", "imagenet"): "timm/deit_tiny_patch16_224.fb_in1k",
    ("resnet18", "imagenet"): "timm/resnet18.tv_in1k",
    ("resnet50", "imagenet"): "timm/resnet50.tv_in1k",
    ("vits16", "imagenet"): "timm/deit_small_patch16_224.fb_in1k",
    ("vits16", "dino"): "timm/vit_small_patch16_224.dino",
}
TORCH_HUB_FILES = {
    ("vitt16", "imagenet"): ["deit_tiny_patch16_224-a1311bcf.pth"],
    ("resnet18", "imagenet"): ["resnet18-f37072fd.pth", "resnet18-5c106cde.pth"],
    ("resnet50", "imagenet"): ["resnet50-0676ba61.pth", "resnet50-19c8e357.pth"],
    ("resnet50", "dino"): ["dino_resnet50_pretrain.pth"],
    ("vits16", "imagenet"): ["deit_small_patch16_224-cd65a155.pth"],
    ("vits16", "dino"): ["dino_deitsmall16_pretrain.pth"],
}
DOWNLOADS = {key: f"https://huggingface.co/{repo}/resolve/main/model.safetensors" for key, repo in HF_REPOS.items()}
DOWNLOADS[("resnet50", "dino")] = \
    "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth"

# Classifier head and final norm stored next to the encoder weights (ImageNet models only)
HEADS = {"resnet18": ("fc.weight", "fc.bias", None), "resnet50": ("fc.weight", "fc.bias", None),
         "vits16": ("head.weight", "head.bias", "norm"), "vitt16": ("head.weight", "head.bias", "norm")}

def cache_dir():
    """Where nave-xai keeps downloaded weights (NAVE_XAI_CACHE overrides it)."""
    if os.environ.get("NAVE_XAI_CACHE"):
        return os.environ["NAVE_XAI_CACHE"]
    if sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Caches")
    elif os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "nave-xai")


def _home_cache():
    return os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")


def find_existing(model, training):
    """A checkpoint already on disk: Hugging Face hub cache, then torch hub cache."""
    repo = HF_REPOS.get((model, training))
    if repo:
        hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
            os.environ.get("HF_HOME") or os.path.join(_home_cache(), "huggingface"), "hub")
        snaps = os.path.join(hub, "models--" + repo.replace("/", "--"), "snapshots")
        if os.path.isdir(snaps):
            files = [os.path.join(snaps, d, "model.safetensors") for d in os.listdir(snaps)]
            files = [f for f in files if os.path.exists(f)]
            if files:
                return max(files, key=os.path.getmtime)
    torch_home = os.environ.get("TORCH_HOME") or os.path.join(_home_cache(), "torch")
    for name in TORCH_HUB_FILES.get((model, training), []):
        path = os.path.join(torch_home, "hub", "checkpoints", name)
        if os.path.exists(path):
            return path
    return None


def load_spec(model):
    with open(os.path.join(GRAPH_DIR, f"{model}.json")) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Random weights
# --------------------------------------------------------------------------
def _trunc_normal(rng, shape, std):
    return np.clip(rng.normal(0.0, std, shape), -2 * std, 2 * std)


def random_weights(spec, seed):
    """Fixed-size weights with each architecture's usual initialisation."""
    rng = np.random.default_rng(seed)
    out = {}
    for w in spec["weights"]:
        if w.get("input"):
            continue
        shape, kind = tuple(w["shape"]), w["kind"]
        if kind == "conv":                       # Kaiming normal, fan_out (torchvision ResNet)
            a = rng.normal(0.0, math.sqrt(2.0 / (shape[0] * shape[2] * shape[3])), shape)
        elif kind == "patch":                    # PyTorch default conv init
            bound = 1.0 / math.sqrt(shape[1] * shape[2] * shape[3])
            a = rng.uniform(-bound, bound, shape)
        elif kind == "linear":                   # timm ViT: truncated normal, std 0.02
            a = _trunc_normal(rng, shape, 0.02)
        elif kind == "cls":
            a = rng.normal(0.0, 1e-6, shape)
        elif kind == "ones":
            a = np.ones(shape)
        else:
            a = np.zeros(shape)
        out[w["name"]] = a.astype(np.float32)
    return out


def random_pos_embed(seed, dim=384, grid=14):
    """A random ViT position embedding on a grid x grid patch grid (+ class token)."""
    rng = np.random.default_rng([seed, 1])
    return _trunc_normal(rng, (1, grid * grid + 1, dim), 0.02).astype(np.float32)


# --------------------------------------------------------------------------
# Checkpoint readers (no torch)
# --------------------------------------------------------------------------
_ST_DTYPES = {"F32": np.float32, "F16": np.float16, "F64": np.float64, "I64": np.int64, "I32": np.int32}


def read_safetensors(path, names=None):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, info in header.items():
            if name == "__metadata__" or (names is not None and name not in names):
                continue
            start, end = info["data_offsets"]
            f.seek(base + start)
            raw = f.read(end - start)
            if info["dtype"] == "BF16":
                a = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
            else:
                a = np.frombuffer(raw, _ST_DTYPES[info["dtype"]])
            out[name] = a.reshape(info["shape"])
    return out


_TORCH_DTYPES = {"FloatStorage": np.float32, "HalfStorage": np.float16, "DoubleStorage": np.float64,
                 "LongStorage": np.int64, "IntStorage": np.int32, "BFloat16Storage": "bf16"}


class _Storage:
    def __init__(self, dtype, key):
        self.dtype, self.key = dtype, key


def _rebuild_tensor(storage, offset, size, stride, *_):
    return ("tensor", storage, offset, tuple(size), tuple(stride))


def _rebuild_parameter(data, *_):
    return data


class _TorchUnpickler(pickle.Unpickler):
    """Allows only what a plain state_dict needs; anything else is refused."""

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "torch" and name in _TORCH_DTYPES:
            return name
        raise pickle.UnpicklingError(f"Refusing to load {module}.{name} from checkpoint")

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        return _Storage(_TORCH_DTYPES[pid[1]], pid[2])


def read_pth(path):
    if not zipfile.is_zipfile(path):
        raise ValueError(f"{path} is a legacy (non-zip) PyTorch checkpoint; not supported without torch.")
    with zipfile.ZipFile(path) as z:
        pkl = next(n for n in z.namelist() if n.endswith("data.pkl"))
        root = pkl[: -len("data.pkl")]
        obj = _TorchUnpickler(z.open(pkl)).load()
        for k in ("state_dict", "model", "teacher"):
            if isinstance(obj, dict) and k in obj and isinstance(obj[k], dict):
                obj = obj[k]
        cache, out = {}, {}
        for name, t in obj.items():
            if not (isinstance(t, tuple) and t and t[0] == "tensor"):
                continue
            _, st, offset, size, stride = t
            if st.key not in cache:
                raw = z.read(f"{root}data/{st.key}")
                if st.dtype == "bf16":
                    cache[st.key] = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
                else:
                    cache[st.key] = np.frombuffer(raw, st.dtype)
            buf = cache[st.key]
            item = buf.itemsize
            a = np.lib.stride_tricks.as_strided(buf[offset:], shape=size,
                                                strides=tuple(s * item for s in stride))
            for prefix in ("module.", "backbone."):
                if name.startswith(prefix):
                    name = name[len(prefix):]
            out[name] = np.array(a)
    return out


# --------------------------------------------------------------------------
# Download and assembly
# --------------------------------------------------------------------------
def download(url, dest, progress=None):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    request = urllib.request.Request(url, headers={"User-Agent": "nave-xai"})
    try:
        with urllib.request.urlopen(request, timeout=60) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if progress and total:
                    progress(0.1 * done / total, f"downloading weights {done >> 20}/{total >> 20} MB")
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        hint = ""
        if "CERTIFICATE" in str(e).upper():
            hint = ("\nThis Python has no SSL certificates. On macOS with a python.org install, "
                    "run 'Install Certificates.command' from its Applications folder.")
        raise RuntimeError(f"Download failed: {url}\n{e}{hint}") from e
    os.replace(tmp, dest)


def checkpoint_path(model, training, progress=None):
    """Path of a local checkpoint for (model, training), downloading it if needed."""
    if (model, training) not in DOWNLOADS:
        raise ValueError(f"There are no {training!r} weights for {model!r}.")
    url = DOWNLOADS[(model, training)]
    own = os.path.join(cache_dir(), f"{model}_{training}_{os.path.basename(url)}")
    if os.path.exists(own):
        return own
    found = find_existing(model, training)
    if found:
        return found
    download(url, own, progress)
    return own


def pretrained_weights(model, training, spec, progress=None):
    """Encoder weights for the graph, plus the classifier head when the checkpoint has one.

    Returns (weights, head) where head is None or a dict with "weight", "bias" and,
    for ViT, "norm_weight"/"norm_bias" (the final layer norm before the classifier).
    """
    path = checkpoint_path(model, training, progress)
    head_names = HEADS[model]
    extra = {head_names[0], head_names[1]}
    if head_names[2]:
        extra |= {head_names[2] + ".weight", head_names[2] + ".bias"}
    wanted = {w["name"] for w in spec["weights"]} | extra
    if path.endswith(".safetensors"):
        raw = read_safetensors(path, wanted)
    elif path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            raw = {name: z[name] for name in z.files}
    else:
        raw = read_pth(path)
    weights = {}
    for w in spec["weights"]:
        name = w["name"]
        if name not in raw:
            raise RuntimeError(f"Weight {name!r} is missing from {path}. "
                               "If nave-xai downloaded this file, delete it to download it again.")
        a = raw[name].astype(np.float32)
        if not w.get("input") and a.shape != tuple(w["shape"]):
            raise RuntimeError(f"Weight {name!r} in {path} has shape {a.shape}, expected {tuple(w['shape'])}.")
        weights[name] = np.ascontiguousarray(a)
    head = None
    if head_names[0] in raw and raw[head_names[0]].ndim == 2:
        head = {"weight": raw[head_names[0]].astype(np.float32),
                "bias": raw[head_names[1]].astype(np.float32) if head_names[1] in raw else None}
        if head_names[2]:
            head["norm_weight"] = raw[head_names[2] + ".weight"].astype(np.float32)
            head["norm_bias"] = raw[head_names[2] + ".bias"].astype(np.float32)
    return weights, head


def download_weights(model="vitt16", training="imagenet", progress=None):
    """Download a built-in model's pretrained weights now, and return where they are.

    nave-xai otherwise downloads them the first time a model needs them; call this ahead of time,
    for instance before working offline. They are kept in `cache_dir()` (`NAVE_XAI_CACHE` sets it).

    >>> from nave_xai import download_weights
    >>> download_weights("vitt16", "imagenet")        # ViT-Ti/16, about 11 MB, once
    """
    return checkpoint_path(model, training, progress)
