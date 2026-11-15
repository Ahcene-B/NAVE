# nave-xai

**NAVE explanations of vision encoders: which concepts an encoder extracts, and where.**

nave-xai is the official package of **NAVE** (Neuro-Activated Vision Explanations), from
[*Multi-Depth Concept Extraction for Post-Hoc Vision Encoder Explanation*](https://arxiv.org/abs/2411.19700)
by Ahcène Boubekki, Samuel G. Fadel and Sebastian Mair.

NAVE explains the encoder, not a prediction. It combines a frozen encoder's feature activations
at several depths, extracts concepts from them, and returns an explanation map of how the encoder
decomposes the image, before any classifier. Post-hoc, unsupervised, for CNNs and ViTs.

!!! note "Citation"
    If you use nave-xai, please cite the paper:

    ```bibtex
    @article{boubekki2024nave,
      title   = {Multi-Depth Concept Extraction for Post-Hoc Vision Encoder Explanation},
      author  = {Boubekki, Ahc{\`e}ne and Fadel, Samuel G. and Mair, Sebastian},
      journal = {arXiv preprint arXiv:2411.19700},
      year    = {2024}
    }
    ```

```bash
pip install nave-xai
```

```python
from nave_xai import NAVEImage
exp = NAVEImage().explain(image)
exp.explanation          # (H, W) the concept of every pixel
```

- New here? Start with [Getting started](getting-started.md).
- The method: [How NAVE works](concepts.md).
- NAVE as a diagnostic tool: [Inspecting models](inspecting.md).
