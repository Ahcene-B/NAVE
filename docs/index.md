# nave-xai

**NAVE explanations of vision encoders: which concepts an encoder extracts, and where.**

nave-xai is the official package of **NAVE** (Neuro-Activated Vision Explanations), from
[*Multi-depth Concept Extraction for Post-Hoc Vision Encoder Explanation*](https://doi.org/10.1007/978-3-032-37664-0_26) (ECML PKDD 2026)
by Ahcène Boubekki, Samuel G. Fadel and Sebastian Mair.

NAVE explains the encoder, not a prediction. It combines a frozen encoder's feature activations
at several depths, extracts concepts from them, and returns an explanation map of how the encoder
decomposes the image, before any classifier. Post-hoc, unsupervised, for CNNs and ViTs.

!!! note "Citation"
    If you use nave-xai, please cite the paper:

    ```bibtex
    @inproceedings{boubekki2026nave,
      author    = {Boubekki, Ahc{\`e}ne and Fadel, Samuel G. and Mair, Sebastian},
      editor    = {Ceci, Michelangelo and Kocev, Dragi and Monreale, Anna and Vens, Celine and Flach, Peter
                   and Manco, Giuseppe and Spiliopoulou, Myra and Masciari, Elio},
      title     = {Multi-depth Concept Extraction for Post-Hoc Vision Encoder Explanation},
      booktitle = {Machine Learning and Knowledge Discovery in Databases. Research Track},
      publisher = {Springer Nature Switzerland},
      address   = {Cham},
      pages     = {449--467},
      year      = {2026},
      doi       = {10.1007/978-3-032-37664-0_26},
      isbn      = {978-3-032-37664-0}
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
