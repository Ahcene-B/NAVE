# Your own models

Any encoder that turns an image into feature maps can be explained. Pass it as `model=` and name
its layers.

## PyTorch

```bash
pip install "nave-xai[torch]"
```

```python
import torchvision
from nave_xai import NAVEImage

model = torchvision.models.resnet101(weights="DEFAULT")
exp = NAVEImage(model, layers=["layer3", "layer4"], k=10).explain(image)
```

Layers are module names from `model.named_modules()`: `"layer3"` (torchvision ResNet),
`"features.5"` (torchvision ConvNeXt), `"blocks.11"` (timm ViT), `"encoder.layer.11"`
(Hugging Face ViT). Their outputs are captured with forward hooks during one forward pass.
Token outputs `(N, L, C)` are reshaped to the patch grid after dropping class and register tokens.

For more control, build the backbone yourself:

```python
from nave_xai.models import TorchBackbone

backbone = TorchBackbone(
    model,
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),   # the model's normalisation
    multiple=16,             # input sides must be multiples of this (patch size for ViTs)
    patch_size=16,           # token outputs: patch size (found automatically if omitted)
    prefix_tokens=1,         # class + register tokens before the patches (found automatically)
    classifier="head",       # for classes and saliency: module name or a torch.nn.Linear
    final_norm="norm",       # norm the classifier reads after (ViTs)
    device="cuda",
)
NAVEImage(backbone, layers=["blocks.8", "blocks.11"])
```

## ONNX

```python
from nave_xai.models import OnnxBackbone
NAVEImage(OnnxBackbone("encoder.onnx", blocks=["stage3", "stage4"], multiple=32), layers=["stage3", "stage4"])
```

The model's first input takes the normalised image `(1, 3, H, W)`; the named outputs are the
layers, each `(1, C, h, w)`.

## Any function (JAX, TensorFlow, ...)

```python
from nave_xai.models import FunctionBackbone

def features(x):                       # x: float32 (1, 3, H, W), already normalised
    return {"mid": mid_features(x), "deep": deep_features(x)}     # numpy (1, C, h, w)

NAVEImage(FunctionBackbone(features, blocks=["mid", "deep"], multiple=32), layers=["mid", "deep"])
```
