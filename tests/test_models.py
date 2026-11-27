import io
import json
import pickle
import struct
import sys
import types
import zipfile
from collections import OrderedDict

import numpy as np
import onnxruntime as ort
import pytest

from nave_xai import models
from nave_xai.models import weights as W


# -- graphs against an independent numpy implementation ---------------------
def _conv(x, w, b=None, s=1, p=0):
    C, H, Wd = x.shape
    O, _, k, _ = w.shape
    x = np.pad(x, ((0, 0), (p, p), (p, p)))
    Ho, Wo = (H + 2 * p - k) // s + 1, (Wd + 2 * p - k) // s + 1
    cols = np.empty((C, k, k, Ho, Wo), np.float32)
    for i in range(k):
        for j in range(k):
            cols[:, i, j] = x[:, i:i + s * Ho:s, j:j + s * Wo:s]
    y = (w.reshape(O, -1) @ cols.reshape(-1, Ho * Wo)).reshape(O, Ho, Wo)
    return y if b is None else y + b[:, None, None]


def _bn(x, Wt, p):
    return ((x - Wt[p + ".running_mean"][:, None, None]) / np.sqrt(Wt[p + ".running_var"][:, None, None] + 1e-5)
            * Wt[p + ".weight"][:, None, None] + Wt[p + ".bias"][:, None, None])


def _resnet18(x, Wt):
    relu = lambda v: np.maximum(v, 0)
    x = relu(_bn(_conv(x, Wt["conv1.weight"], s=2, p=3), Wt, "bn1"))
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)), constant_values=-np.inf)
    h, w = x.shape[1] // 2, x.shape[2] // 2
    x = np.max(np.stack([xp[:, i:i + 2 * h:2, j:j + 2 * w:2] for i in range(3) for j in range(3)]), 0)
    outs = []
    for li, s in enumerate([1, 2, 2, 2], 1):
        for bi in range(2):
            p, st = f"layer{li}.{bi}", s if bi == 0 else 1
            y = relu(_bn(_conv(x, Wt[p + ".conv1.weight"], s=st, p=1), Wt, p + ".bn1"))
            y = _bn(_conv(y, Wt[p + ".conv2.weight"], p=1), Wt, p + ".bn2")
            ds = p + ".downsample.0.weight"
            x = relu(y + (_bn(_conv(x, Wt[ds], s=st), Wt, p + ".downsample.1") if ds in Wt else x))
        outs.append(x)
    return outs


def _vit(x, Wt, pos):
    from math import erf
    verf = np.vectorize(erf)

    def ln(v, w, b):
        m = v.mean(-1, keepdims=True)
        return (v - m) / np.sqrt(((v - m) ** 2).mean(-1, keepdims=True) + 1e-6) * w + b
    t = _conv(x, Wt["patch_embed.proj.weight"], Wt["patch_embed.proj.bias"], s=16)
    C, gh, gw = t.shape
    t = np.concatenate([Wt["cls_token"][0], t.reshape(C, -1).T]) + pos[0]
    outs = []
    for i in range(12):
        p = f"blocks.{i}."
        h = ln(t, Wt[p + "norm1.weight"], Wt[p + "norm1.bias"])
        q, k, v = (h @ Wt[p + "attn.qkv.weight"].T + Wt[p + "attn.qkv.bias"]).reshape(-1, 3, 6, 64).transpose(1, 2, 0, 3)
        a = q @ k.transpose(0, 2, 1) / 8.0
        a = np.exp(a - a.max(-1, keepdims=True))
        a /= a.sum(-1, keepdims=True)
        t = t + (a @ v).transpose(1, 0, 2).reshape(-1, C) @ Wt[p + "attn.proj.weight"].T + Wt[p + "attn.proj.bias"]
        h = ln(t, Wt[p + "norm2.weight"], Wt[p + "norm2.bias"]) @ Wt[p + "mlp.fc1.weight"].T + Wt[p + "mlp.fc1.bias"]
        t = t + (0.5 * h * (1 + verf(h / np.sqrt(2)))) @ Wt[p + "mlp.fc2.weight"].T + Wt[p + "mlp.fc2.bias"]
        outs.append(t[1:].T.reshape(C, gh, gw))
    return outs


@pytest.mark.parametrize("model", ["resnet18", "vits16"])
def test_graph_matches_reference(model, rng):
    spec = W.load_spec(model)
    Wt = W.random_weights(spec, 1)
    for k in Wt:                                       # exercise every parameter, BN statistics included
        if k.endswith("running_var"):
            Wt[k] = rng.uniform(0.5, 2, Wt[k].shape).astype(np.float32)
        elif Wt[k].ndim == 1:
            Wt[k] = (Wt[k] + rng.normal(0, 0.1, Wt[k].shape)).astype(np.float32)
    x = rng.normal(size=(1, 3, 160, 224)).astype(np.float32)
    feeds = {"image": x}
    if model == "vits16":
        feeds["pos_embed"] = rng.normal(0, 0.02, (1, 10 * 14 + 1, 384)).astype(np.float32)
    names = list(Wt)
    options = ort.SessionOptions()
    options.add_external_initializers(names, [ort.OrtValue.ortvalue_from_numpy(Wt[n]) for n in names])
    session = ort.InferenceSession(f"{W.GRAPH_DIR}/{model}.onnx", options, providers=["CPUExecutionProvider"])
    blocks = [o.name for o in session.get_outputs() if o.name.startswith("b") and o.name != "b0"]   # block outputs
    out = session.run(blocks, feeds)
    ref = _resnet18(x[0], Wt) if model == "resnet18" else _vit(x[0], Wt, feeds["pos_embed"])
    for o, r in zip(out, ref):
        assert np.abs(o[0] - r).max() / np.abs(r).max() < 1e-4


def test_builtin_backbones_any_size():
    for name, info in models.available().items():
        bb = models.get_backbone(name, "random")
        assert bb.resolve_blocks(None) == [f"b{i}" for i in info["default_blocks"]]
        out = bb(np.zeros((1, 3, 96, 160), np.float32), bb.blocks)
        assert len(out) == info["blocks"]
    vit = models.get_backbone("vits16", "random")
    assert vit.resolve_blocks("N:9-12") == ["b9", "b12"]
    assert models.get_backbone("resnet50", "random").resolve_blocks("N234") == ["b2", "b3", "b4"]


def test_function_backbone_end_to_end(rng):
    from nave_xai import NAVEImage

    def features(x):                                  # any framework: return numpy feature maps
        return {"fine": x[:, :, ::4, ::4], "coarse": np.abs(x[:, :, ::8, ::8])}
    fb = models.FunctionBackbone(features, blocks=["fine", "coarse"], multiple=8)
    image = (rng.random((96, 120, 3)) * 255).astype(np.uint8)
    exp = NAVEImage(fb, layers=["fine", "coarse"], k=4).explain(image)
    assert exp.explanation.shape == (96, 120)


# -- checkpoint readers -------------------------------------------------------
def test_safetensors_reader(tmp_path, rng):
    a = rng.normal(size=(3, 4)).astype(np.float32)
    bf = (np.frombuffer(a.tobytes(), np.uint32) >> 16).astype(np.uint16)
    header = json.dumps({"a": {"dtype": "F32", "shape": [3, 4], "data_offsets": [0, 48]},
                         "b": {"dtype": "BF16", "shape": [3, 4], "data_offsets": [48, 72]}}).encode()
    path = tmp_path / "w.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + a.tobytes() + bf.tobytes())
    out = W.read_safetensors(str(path))
    assert np.array_equal(out["a"], a)
    assert np.allclose(out["b"], a, rtol=1e-2)


def _fake_pth(path, tensors):
    """Write a zip checkpoint laid out like torch.save, without torch."""
    torch, utils = types.ModuleType("torch"), types.ModuleType("torch._utils")

    class FloatStorage:
        pass
    FloatStorage.__module__, FloatStorage.__qualname__ = "torch", "FloatStorage"
    torch.FloatStorage = FloatStorage

    def _rebuild_tensor_v2(*a):
        pass
    _rebuild_tensor_v2.__module__, _rebuild_tensor_v2.__qualname__ = "torch._utils", "_rebuild_tensor_v2"
    utils._rebuild_tensor_v2 = _rebuild_tensor_v2

    class Storage:
        def __init__(self, key):
            self.key = key

    class Tensor:
        def __init__(self, key, shape):
            self.key, self.shape = key, shape

        def __reduce__(self):
            stride = tuple(int(np.prod(self.shape[i + 1:])) for i in range(len(self.shape)))
            return _rebuild_tensor_v2, (Storage(self.key), 0, tuple(self.shape), stride, False, {})

    class Pickler(pickle.Pickler):
        def persistent_id(self, obj):
            return ("storage", FloatStorage, obj.key, "cpu", 0) if isinstance(obj, Storage) else None

    saved = dict(sys.modules)
    sys.modules.update({"torch": torch, "torch._utils": utils})
    try:
        buf = io.BytesIO()
        Pickler(buf, protocol=2).dump(OrderedDict((f"module.{n}", Tensor(str(i), a.shape))
                                                  for i, (n, a) in enumerate(tensors.items())))
    finally:
        sys.modules.clear()
        sys.modules.update(saved)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("archive/data.pkl", buf.getvalue())
        for i, a in enumerate(tensors.values()):
            z.writestr(f"archive/data/{i}", a.astype(np.float32).tobytes())


def test_pth_reader_and_safety(tmp_path, rng):
    tensors = {"conv.weight": rng.normal(size=(2, 3, 3, 3)), "fc.bias": rng.normal(size=(5,))}
    _fake_pth(tmp_path / "ok.pth", tensors)
    out = W.read_pth(str(tmp_path / "ok.pth"))
    assert np.allclose(out["conv.weight"], tensors["conv.weight"]) and np.allclose(out["fc.bias"], tensors["fc.bias"])
    evil = io.BytesIO()
    pickle.dump(OrderedDict(x=print), evil, protocol=2)
    with zipfile.ZipFile(tmp_path / "evil.pth", "w") as z:
        z.writestr("archive/data.pkl", evil.getvalue())
    with pytest.raises(pickle.UnpicklingError):
        W.read_pth(str(tmp_path / "evil.pth"))


def test_existing_caches_are_found(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    for var in ("HF_HUB_CACHE", "HF_HOME", "TORCH_HOME"):
        monkeypatch.delenv(var, raising=False)
    snap = tmp_path / "huggingface/hub/models--timm--vit_small_patch16_224.dino/snapshots/abc"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"")
    hub = tmp_path / "torch/hub/checkpoints"
    hub.mkdir(parents=True)
    (hub / "dino_resnet50_pretrain.pth").write_bytes(b"")
    assert W.find_existing("vits16", "dino").endswith("model.safetensors")
    assert W.find_existing("resnet50", "dino").endswith("dino_resnet50_pretrain.pth")
    assert W.find_existing("vits16", "imagenet") is None


def test_b0_is_the_input_of_block_1(rng):
    from nave_xai.models import weights as W
    from nave_xai.models.backbone import BuiltinBackbone
    x = rng.normal(size=(1, 3, 64, 64)).astype(np.float32)
    res = BuiltinBackbone("resnet18", "random", seed=0)                 # ResNet: the stem, up to the max-pool
    Wt = W.random_weights(W.load_spec("resnet18"), 0)
    s = np.maximum(_bn(_conv(x[0], Wt["conv1.weight"], s=2, p=3), Wt, "bn1"), 0)
    sp = np.pad(s, ((0, 0), (1, 1), (1, 1)), constant_values=-np.inf)
    h, w = s.shape[1] // 2, s.shape[2] // 2
    stem = np.max(np.stack([sp[:, i:i + 2 * h:2, j:j + 2 * w:2] for i in range(3) for j in range(3)]), 0)
    assert np.allclose(res(x, ["b0"])["b0"][0], stem, atol=1e-4)
    vit = BuiltinBackbone("vitt16", "random", seed=0)                    # ViT: patch embedding + position
    Wv = W.random_weights(W.load_spec("vitt16"), 0)
    def embedding(img):
        p = img[0].reshape(3, 4, 16, 4, 16).transpose(1, 3, 0, 2, 4).reshape(16, -1)
        return p @ Wv["patch_embed.proj.weight"].reshape(192, -1).T + Wv["patch_embed.proj.bias"]
    y = rng.normal(size=(1, 3, 64, 64)).astype(np.float32)
    pos_x = vit(x, ["b0"])["b0"][0].reshape(192, -1).T - embedding(x)
    pos_y = vit(y, ["b0"])["b0"][0].reshape(192, -1).T - embedding(y)
    assert np.allclose(pos_x, pos_y, atol=1e-4)                         # what remains is the position embedding
    assert vit.resolve_blocks(["b3", "b0", 1]) == ["b0", "b1", "b3"]


def test_download_weights_names_what_it_cannot_download():
    import nave_xai
    assert callable(nave_xai.download_weights)
    with pytest.raises(ValueError):
        nave_xai.download_weights("resnet18", "random")          # random weights have nothing to download
