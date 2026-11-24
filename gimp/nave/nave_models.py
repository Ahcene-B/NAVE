# SPDX-License-Identifier: MIT
"""
NAVE backbones: weights and onnxruntime sessions, without torch.

Weights come from one of three places:
  * random  -> generated from the seed with each architecture's usual init
  * .safetensors (timm, Hugging Face) -> parsed with numpy only
  * .pth (official DINO ResNet50) -> zip + restricted unpickler, no torch
and are injected into the weight-less graphs in graphs/ with
SessionOptions.add_external_initializers.
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

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH_DIR = os.path.join(HERE, "graphs")

# Files these weights usually already have on a machine that uses timm / torchvision / torch.hub.
HF_REPOS = {
    ("resnet18", "imagenet"): "timm/resnet18.tv_in1k",
    ("resnet50", "imagenet"): "timm/resnet50.tv_in1k",
    ("vits16", "imagenet"): "timm/deit_small_patch16_224.fb_in1k",
    ("vits16", "dino"): "timm/vit_small_patch16_224.dino",
}
TORCH_HUB_FILES = {
    ("resnet18", "imagenet"): ["resnet18-f37072fd.pth", "resnet18-5c106cde.pth"],   # torchvision IMAGENET1K_V1
    ("resnet50", "imagenet"): ["resnet50-0676ba61.pth", "resnet50-19c8e357.pth"],   # torchvision IMAGENET1K_V1
    ("resnet50", "dino"): ["dino_resnet50_pretrain.pth"],
    ("vits16", "imagenet"): ["deit_small_patch16_224-cd65a155.pth"],
    ("vits16", "dino"): ["dino_deitsmall16_pretrain.pth"],
}

SOURCES = {
    ("resnet18", "imagenet"): "https://huggingface.co/timm/resnet18.tv_in1k/resolve/main/model.safetensors",
    ("resnet50", "imagenet"): "https://huggingface.co/timm/resnet50.tv_in1k/resolve/main/model.safetensors",
    ("resnet50", "dino"): "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth",
    ("vits16", "imagenet"): "https://huggingface.co/timm/deit_small_patch16_224.fb_in1k/resolve/main/model.safetensors",
    ("vits16", "dino"): "https://huggingface.co/timm/vit_small_patch16_224.dino/resolve/main/model.safetensors",
}


def cache_dir():
    if sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Caches")
    elif os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "nave")


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
        f = os.path.join(torch_home, "hub", "checkpoints", name)
        if os.path.exists(f):
            return f
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
    """Fixed-size weights only; run-time inputs (ViT pos_embed) are made in Backbone."""
    rng = np.random.default_rng(seed)
    out = {}
    for w in spec["weights"]:
        if w.get("input"):
            continue
        shape, kind = tuple(w["shape"]), w["kind"]
        if kind == "conv":                      # Kaiming normal, fan_out, ReLU (torchvision ResNet)
            a = rng.normal(0.0, math.sqrt(2.0 / (shape[0] * shape[2] * shape[3])), shape)
        elif kind == "patch":                   # PyTorch default conv init
            bound = 1.0 / math.sqrt(shape[1] * shape[2] * shape[3])
            a = rng.uniform(-bound, bound, shape)
        elif kind == "linear":                  # timm ViT: trunc normal, std 0.02
            a = _trunc_normal(rng, shape, 0.02)
        elif kind == "cls":
            a = rng.normal(0.0, 1e-6, shape)
        elif kind == "ones":
            a = np.ones(shape)
        else:
            a = np.zeros(shape)
        out[w["name"]] = a.astype(np.float32)
    return out


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
# Download + assembly
# --------------------------------------------------------------------------
def download(url, dest, progress):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "nave-gimp-plugin"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    progress(0.2 * done / total, f"NAVE: downloading weights {done >> 20}/{total >> 20} MB")
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        hint = ""
        if "CERTIFICATE" in str(e).upper():
            hint = ("\nThis Python has no SSL certificates. On macOS with a python.org install, "
                    "run 'Install Certificates.command' from its Applications folder.")
        raise RuntimeError(f"Download failed: {url}\n{e}{hint}")
    os.replace(tmp, dest)


def random_pos_embed(seed, grid, dim=384):
    rng = np.random.default_rng([seed, 1])
    return _trunc_normal(rng, (1, grid * grid + 1, dim), 0.02).astype(np.float32)


def _resize_pos_embed(pe, n_target):
    """(1, 1+g*g, C) -> (1, n_target, C), bicubic on the patch grid."""
    from nave_core import interp_matrix, resize_coords, apply_separable
    cls, grid = pe[:, :1], pe[0, 1:]
    g_in, g_out = int(round(math.sqrt(grid.shape[0]))), int(round(math.sqrt(n_target - 1)))
    grid = grid.reshape(g_in, g_in, -1)
    A = interp_matrix(resize_coords(g_in, g_out), g_in, "cubic")
    grid = apply_separable(A, A, grid).reshape(1, g_out * g_out, -1)
    return np.concatenate([cls, grid], axis=1)


def pretrained_weights(model, training, spec, progress):
    if (model, training) not in SOURCES:
        raise RuntimeError(f"There are no {training} weights for {model}.")
    url = SOURCES[(model, training)]
    dest = os.path.join(cache_dir(), "weights", f"{model}_{training}_{os.path.basename(url)}")
    if not os.path.exists(dest):
        found = find_existing(model, training)
        if found:
            progress(0.1, f"NAVE: using {found}")
            dest = found
        else:
            download(url, dest, progress)
    names = {w["name"] for w in spec["weights"]}
    raw = read_safetensors(dest, names) if dest.endswith(".safetensors") else read_pth(dest)
    out = {}
    for w in spec["weights"]:
        name, shape = w["name"], tuple(w["shape"])
        if name not in raw:
            raise RuntimeError(f"Weight '{name}' missing from {dest}. "
                               "If the plugin downloaded this file, delete it to download it again.")
        a = raw[name].astype(np.float32)
        if w.get("input"):                           # sized to the window in Backbone
            out[name] = a
            continue
        if a.shape != shape:
            raise RuntimeError(f"Weight '{name}' has shape {a.shape}, expected {shape}.")
        out[name] = np.ascontiguousarray(a)
    return out


class Backbone:
    """onnxruntime session for one model, for square windows of `window` px."""

    def __init__(self, model, training, seed=0, window=448, progress=lambda f, t="": None):
        import onnxruntime as ort
        spec = load_spec(model)
        self.window = window
        if training == "random":
            weights = random_weights(spec, seed)
        else:
            weights = pretrained_weights(model, training, spec, progress)
        progress(0.2, "NAVE: loading model")
        self.feeds = {}
        for w in spec["weights"]:                     # run-time inputs, sized to the window
            if w.get("input") and w["name"] == "pos_embed":
                grid = window // 16
                if training == "random":
                    pe = random_pos_embed(seed, grid)
                else:
                    pe = weights[w["name"]]
                    if pe.shape[1] != grid * grid + 1:
                        pe = _resize_pos_embed(pe, grid * grid + 1)
                self.feeds["pos_embed"] = np.ascontiguousarray(pe, np.float32)
                weights.pop(w["name"], None)
        self._weights = weights                       # keep alive for the session's lifetime
        names = list(weights)
        values = [ort.OrtValue.ortvalue_from_numpy(weights[n]) for n in names]
        def make(level):
            so = ort.SessionOptions()
            so.graph_optimization_level = level
            so.add_external_initializers(names, values)
            return ort.InferenceSession(os.path.join(GRAPH_DIR, f"{model}.onnx"), so, providers=["CPUExecutionProvider"])
        try:
            self.sess = make(ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
        except Exception:                             # some onnxruntime versions reject the full optimisation
            self.sess = make(ort.GraphOptimizationLevel.ORT_ENABLE_BASIC)
        self._values = values
        # only the block outputs b1, b2, ... (the graphs may also offer b0, keys and the class token)
        self.output_names = [o.name for o in self.sess.get_outputs()
                             if o.name[:1] == "b" and o.name[1:].isdigit() and o.name != "b0"]

    def __call__(self, x):
        return dict(zip(self.output_names, self.sess.run(self.output_names, {"image": x, **self.feeds})))
