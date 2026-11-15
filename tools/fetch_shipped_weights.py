# SPDX-License-Identifier: MIT
"""Developer tool: put the weights of the shipped model inside the package.

Fetches DeiT-Tiny (ViT-Ti/16, ImageNet, Apache-2.0) from the Hugging Face hub (or uses a copy
already in your Hugging Face / torch hub cache), keeps the tensors nave-xai needs (encoder,
classifier head, final norm), converts them to float16 and writes

    src/nave_xai/models/weights/vitt16_imagenet.npz        (about 11 MB)

Run it once before `python -m build`:

    python tools/fetch_shipped_weights.py
    python tools/fetch_shipped_weights.py --source path/to/model.safetensors   # an existing file
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from nave_xai.models import weights as W  # noqa: E402

MODEL, TRAINING = "vitt16", "imagenet"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="an existing .safetensors or .pth checkpoint of DeiT-Tiny")
    args = ap.parse_args()

    if args.source:
        path = args.source
    else:
        path = W.find_existing(MODEL, TRAINING)
        if path is None:
            path = os.path.join(W.cache_dir(), "vitt16_imagenet_model.safetensors")
            if not os.path.exists(path):
                print("downloading", W.DOWNLOADS[(MODEL, TRAINING)])
                W.download(W.DOWNLOADS[(MODEL, TRAINING)], path)
    print("reading", path)
    raw = W.read_safetensors(path) if path.endswith(".safetensors") else W.read_pth(path)

    spec = W.load_spec(MODEL)
    head_w, head_b, norm = W.HEADS[MODEL]
    names = [w["name"] for w in spec["weights"]] + [head_w, head_b, norm + ".weight", norm + ".bias"]
    missing = [n for n in names if n not in raw]
    if missing:
        sys.exit(f"Missing tensors in {path}: {missing[:5]}")
    for w in spec["weights"]:
        if not w.get("input") and tuple(raw[w["name"]].shape) != tuple(w["shape"]):
            sys.exit(f"{w['name']} has shape {raw[w['name']].shape}, expected {tuple(w['shape'])}: not DeiT-Tiny?")

    out = os.path.join(W.SHIPPED_DIR, W.SHIPPED[(MODEL, TRAINING)])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **{n: np.asarray(raw[n], np.float32).astype(np.float16) for n in names})
    print(f"wrote {out} ({os.path.getsize(out) / 2 ** 20:.1f} MB)")


if __name__ == "__main__":
    main()
