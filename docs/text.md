# Text

`NAVEText` explains text encoders: one concept per token.

```python
from nave_xai import NAVEText

explainer = NAVEText("spacy").fit(documents)            # pip install "nave-xai[text]"
exp = explainer.explain("Oil prices rose after OPEC cut production.", uncertainty=True)
list(zip(exp.tokens, exp.explanation))
explainer.summarize(documents[0], saliency=True, classifier=head)
```

## Inputs and models

A string is one document and a list of strings several; an already tokenized document is a list
of tokens, so several of them are a list of lists.

- `model="spacy"`: spaCy's `tok2vec` encoder of `en_core_web_sm` (hashed embeddings, then four
  residual window-CNN blocks of 96 channels; MIT license). Install it with
  `pip install "nave-xai[text]"`, then `python -m spacy download en_core_web_sm` (about 12 MB).
  Layers are `"block1"` to `"block4"`; by default blocks 2 to 4. Its tokenizer is used.
- Any function `tokens -> {layer: (C_b, n_tokens) array}`, optionally with the class scores under
  `"logits"`. PyTorch and Hugging Face models are wrapped in such a function (tokens to ids inside).

Text needs more concepts than images: the default is `k=20`. With few concepts, the tokens a
classifier relies on most (the topic keywords) tend to share one concept whatever the topic;
`groups=` (one projector per class) or more concepts separate them.

## What `fit` computes

As for images, `prototypes`, `palette`, `classes` and `correlation`; for every concept:

| | |
|---|---|
| `span_counts` | number of spans (runs of consecutive tokens with the concept) |
| `mean_runs` | mean number of consecutive tokens with the concept |
| `words` | its characteristic words: frequency in the concept relative to overall |
| `spans` | its spans closest to the prototype, with context |

`update_spans(texts)` recomputes them on other texts.

## Summary

`summarize(text)` shows the document wrapped, each token on its concept's colour with the
background's opacity = certainty; with `saliency=True`, the font size shows the saliency's magnitude
and black / white its sign. Then the correlation of the concepts with the classes, and one row per
concept: colour, spans and mean run, characteristic words and its two closest spans in context.
`label=` adds the true class next to the model's prediction in the title. `reorder(by="classes", recolor=True)` is recommended for
text: every class gets its colour family, so a document's topic shows at a glance.

See `notebooks/text.ipynb` for a complete example on Reuters news.

Without correlation (one item, or a single class), the summary shows each concept's mean saliency in
the item instead, on [-1, 1] (it needs a classifier).
