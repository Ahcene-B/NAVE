#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Developer tool: build the weight-less ONNX graphs shipped in nave_xai/models/graphs.
Needs the `onnx` package (pip install onnx). Users never run this.

    python tools/build_graphs.py           # resnet18, resnet50, vits16 (ViT-S/16), vitt16 (ViT-Ti/16)

Each graph takes a 1x3xHxW ImageNet-normalized window (any size; multiples of 32)
and returns every block's output as b1..bN with shape (1, C, h, w). The ViT graph
also takes its position embedding as an input sized to the window, and returns the
class token after the last block as `cls` (for the classifier). Weights are external-data
placeholders named exactly like the timm / torchvision state-dict keys; the
worker supplies them at load time with SessionOptions.add_external_initializers.
The .json next to each graph lists every weight: name, shape, init kind.
"""
import json
import os

import onnx
from onnx import TensorProto, helper

OPSET, IR = 17, 8
HERE = os.path.dirname(os.path.abspath(__file__))


class Graph:
    def __init__(self):
        self.nodes, self.inits, self.spec, self.n, self.inputs = [], [], [], 0, []

    def tmp(self, base="t"):
        self.n += 1
        return f"{base}_{self.n}"

    def weight(self, name, shape, kind):
        t = TensorProto()
        t.name, t.data_type = name, TensorProto.FLOAT
        t.dims.extend(shape)
        t.data_location = TensorProto.EXTERNAL
        e = t.external_data.add()
        e.key, e.value = "location", "weights.bin"      # never read: weights are injected
        self.inits.append(t)
        self.spec.append({"name": name, "shape": list(shape), "kind": kind})
        return name

    def input(self, name, shape, kind):
        """A weight fed at run time (its size depends on the window)."""
        self.inputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))
        self.spec.append({"name": name, "shape": shape, "kind": kind, "input": True})
        return name

    def const(self, name, values, dtype=TensorProto.INT64):
        self.inits.append(helper.make_tensor(name, dtype, [len(values)], values))
        return name

    def op(self, op_type, inputs, out=None, **attrs):
        out = out or self.tmp(op_type.lower())
        # every node gets a unique name: some onnxruntime versions derive fused node names from it
        self.nodes.append(helper.make_node(op_type, inputs, [out], name=f"n_{out}", **attrs))
        return out

    def split(self, x, n, axis):
        outs = [self.tmp("split") for _ in range(n)]
        self.nodes.append(helper.make_node("Split", [x], outs, name=f"n_{outs[0]}", axis=axis, num_outputs=n)
                          if OPSET >= 18 else helper.make_node("Split", [x], outs, name=f"n_{outs[0]}", axis=axis))
        return outs

    def save(self, path, outputs, out_shapes):
        g = helper.make_graph(
            self.nodes, os.path.basename(path),
            [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, "H", "W"])] + self.inputs,
            [helper.make_tensor_value_info(o, TensorProto.FLOAT, s) for o, s in zip(outputs, out_shapes)],
            initializer=self.inits)
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", OPSET)], ir_version=IR)
        onnx.save(m, path)
        with open(path[:-5] + ".json", "w") as f:
            json.dump({"weights": self.spec}, f, indent=0)
        print("wrote", path, os.path.getsize(path), "bytes,", len(self.spec), "weights")


# ---------------------------------------------------------------- ResNet50
def conv_bn(g, x, prefix_conv, prefix_bn, cin, cout, k, stride, relu):
    w = g.weight(f"{prefix_conv}.weight", [cout, cin, k, k], "conv")
    y = g.op("Conv", [x, w], kernel_shape=[k, k], strides=[stride, stride], pads=[k // 2] * 4)
    bn = [g.weight(f"{prefix_bn}.{n}", [cout], kind)
          for n, kind in [("weight", "ones"), ("bias", "zeros"), ("running_mean", "zeros"), ("running_var", "ones")]]
    y = g.op("BatchNormalization", [y] + bn, epsilon=1e-5)
    return g.op("Relu", [y]) if relu else y


def build_resnet50(path):
    g = Graph()
    x = conv_bn(g, "image", "conv1", "bn1", 3, 64, 7, 2, True)
    x = g.op("MaxPool", [x], kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])
    cin, outs, shapes = 64, [], []
    for li, (blocks, planes, stride) in enumerate([(3, 64, 1), (4, 128, 2), (6, 256, 2), (3, 512, 2)], 1):
        for bi in range(blocks):
            p, s = f"layer{li}.{bi}", stride if bi == 0 else 1
            y = conv_bn(g, x, f"{p}.conv1", f"{p}.bn1", cin, planes, 1, 1, True)
            y = conv_bn(g, y, f"{p}.conv2", f"{p}.bn2", planes, planes, 3, s, True)   # torchvision: stride on 3x3
            y = conv_bn(g, y, f"{p}.conv3", f"{p}.bn3", planes, planes * 4, 1, 1, False)
            sc = x
            if bi == 0:
                sc = conv_bn(g, x, f"{p}.downsample.0", f"{p}.downsample.1", cin, planes * 4, 1, s, False)
            x = g.op("Relu", [g.op("Add", [y, sc])], out=f"b{li}" if bi == blocks - 1 else None)
            cin = planes * 4
        outs.append(f"b{li}")
        shapes.append([1, cin, f"h{li}", f"w{li}"])
    g.save(path, outs, shapes)


def build_resnet18(path):
    """torchvision ResNet18: BasicBlocks of two 3x3 convs, [2, 2, 2, 2]."""
    g = Graph()
    x = conv_bn(g, "image", "conv1", "bn1", 3, 64, 7, 2, True)
    x = g.op("MaxPool", [x], kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])
    cin, outs, shapes = 64, [], []
    for li, (planes, stride) in enumerate([(64, 1), (128, 2), (256, 2), (512, 2)], 1):
        for bi in range(2):
            p, s = f"layer{li}.{bi}", stride if bi == 0 else 1
            y = conv_bn(g, x, f"{p}.conv1", f"{p}.bn1", cin, planes, 3, s, True)
            y = conv_bn(g, y, f"{p}.conv2", f"{p}.bn2", planes, planes, 3, 1, False)
            sc = x
            if bi == 0 and (s != 1 or cin != planes):
                sc = conv_bn(g, x, f"{p}.downsample.0", f"{p}.downsample.1", cin, planes, 1, s, False)
            x = g.op("Relu", [g.op("Add", [y, sc])], out=f"b{li}" if bi == 1 else None)
            cin = planes
        outs.append(f"b{li}")
        shapes.append([1, cin, f"h{li}", f"w{li}"])
    g.save(path, outs, shapes)


# ---------------------------------------------------------------- ViT-S/16
def layer_norm(g, x, prefix, dim):
    w = g.weight(f"{prefix}.weight", [dim], "ones")
    b = g.weight(f"{prefix}.bias", [dim], "zeros")
    return g.op("LayerNormalization", [x, w, b], axis=-1, epsilon=1e-6)


def linear(g, x, prefix, din, dout):
    w = g.weight(f"{prefix}.weight", [dout, din], "linear")
    b = g.weight(f"{prefix}.bias", [dout], "zeros")
    return g.op("Add", [g.op("MatMul", [x, g.op("Transpose", [w], perm=[1, 0])]), b])


def build_vit(path, dim=384, depth=12, heads=6, mlp=1536, patch=16):
    """ViT with timm's layout (ViT-S/16: dim 384, 6 heads; ViT-Ti/16: dim 192, 3 heads)."""
    g = Graph()
    hd = dim // heads
    w = g.weight("patch_embed.proj.weight", [dim, 3, patch, patch], "patch")
    b = g.weight("patch_embed.proj.bias", [dim], "zeros")
    grid = g.op("Conv", ["image", w, b], kernel_shape=[patch, patch], strides=[patch, patch])  # 1, dim, h, w
    grid_shape = g.op("Shape", [grid])
    x = g.op("Transpose", [g.op("Reshape", [grid, g.const("s_flat", [1, dim, -1])])], perm=[0, 2, 1])
    cls = g.weight("cls_token", [1, 1, dim], "cls")
    x = g.op("Concat", [cls, x], axis=1)
    x = g.op("Add", [x, g.input("pos_embed", [1, "n", dim], "pos")])

    s_qkv = g.const("s_qkv", [0, 0, 3, heads, hd])
    s_tok = g.const("s_tok", [0, 0, dim])
    sl = [g.const(k, v) for k, v in [("sl_start", [1]), ("sl_end", [2 ** 62]), ("sl_axes", [1])]]
    scale = g.const("attn_scale", [hd ** -0.5], TensorProto.FLOAT)
    half = g.const("c_half", [0.5], TensorProto.FLOAT)
    one = g.const("c_one", [1.0], TensorProto.FLOAT)
    rsqrt2 = g.const("c_rsqrt2", [2 ** -0.5], TensorProto.FLOAT)
    idx = [g.const("i0", [0])]

    outs, shapes = [], []
    for i in range(depth):
        p = f"blocks.{i}"
        h = layer_norm(g, x, f"{p}.norm1", dim)
        qkv = g.op("Transpose", [g.op("Reshape", [linear(g, h, f"{p}.attn.qkv", dim, 3 * dim), s_qkv])],
                   perm=[2, 0, 3, 1, 4])                                         # 3, 1, heads, n, hd
        q, k, v = [g.op("Squeeze", [part, idx[0]]) for part in g.split(qkv, 3, axis=0)]
        keys = g.op("Reshape", [g.op("Transpose", [k], perm=[0, 2, 1, 3]), s_tok])   # 1, n, dim (heads concatenated)
        g.op("Reshape", [g.op("Transpose", [g.op("Slice", [keys] + sl)], perm=[0, 2, 1]), grid_shape], out=f"k{i + 1}")
        outs.append(f"k{i + 1}")
        shapes.append([1, dim, "h", "w"])
        att = g.op("Mul", [g.op("MatMul", [q, g.op("Transpose", [k], perm=[0, 1, 3, 2])]), scale])
        o = g.op("MatMul", [g.op("Softmax", [att], axis=-1), v])                  # 1, heads, n, hd
        o = g.op("Reshape", [g.op("Transpose", [o], perm=[0, 2, 1, 3]), s_tok])
        x = g.op("Add", [x, linear(g, o, f"{p}.attn.proj", dim, dim)])
        h = linear(g, layer_norm(g, x, f"{p}.norm2", dim), f"{p}.mlp.fc1", dim, mlp)
        h = g.op("Mul", [g.op("Mul", [h, half]), g.op("Add", [g.op("Erf", [g.op("Mul", [h, rsqrt2])]), one])])
        x = g.op("Add", [x, linear(g, h, f"{p}.mlp.fc2", mlp, dim)])
        tok = g.op("Slice", [x] + sl)                                             # drop cls token
        g.op("Reshape", [g.op("Transpose", [tok], perm=[0, 2, 1]), grid_shape], out=f"b{i + 1}")
        outs.append(f"b{i + 1}")
        shapes.append([1, dim, "h", "w"])
    g.op("Slice", [x, g.const("cls_start", [0]), g.const("cls_end", [1]), sl[2]], out="cls")   # 1, 1, dim
    outs.append("cls")
    shapes.append([1, 1, dim])
    g.save(path, outs, shapes)


if __name__ == "__main__":
    out = os.path.join(HERE, "..", "src", "nave_xai", "models", "graphs")
    os.makedirs(out, exist_ok=True)
    build_resnet18(os.path.join(out, "resnet18.onnx"))
    build_resnet50(os.path.join(out, "resnet50.onnx"))
    build_vit(os.path.join(out, "vits16.onnx"), dim=384, heads=6, mlp=1536)
    build_vit(os.path.join(out, "vitt16.onnx"), dim=192, heads=3, mlp=768)
