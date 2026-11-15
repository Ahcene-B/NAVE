# How NAVE works

NAVE explains a frozen vision encoder: it extracts concepts from the encoder's feature
activations at several depths and maps them back onto the image. The explanation does not depend
on a classifier or a label.

## Layers and composite pixels

An encoder is a stack of layers (blocks). Each outputs a feature map: a grid of positions, each
with a vector of C numbers. A ResNet's four blocks halve the resolution each time:

| Block | Stride | Grid at 448 px | Channels (ResNet18 / 50) |
|---|---|---|---|
| 1 | 4 | 112 x 112 | 64 / 256 |
| 2 | 8 | 56 x 56 | 128 / 512 |
| 3 | 16 | 28 x 28 | 256 / 1024 |
| 4 | 32 | 14 x 14 | 512 / 2048 |

ViT-Ti/16 and ViT-S/16 have 12 blocks, all on the same grid of 16 px patches (192 and 384 channels).

NAVE (Algorithm 1 of the paper) combines the chosen layers per position:

1. **Upsample** every chosen layer to the grid of the shallowest one.
2. **Normalise** each position's vector to length 1 (`normalize=True`).
3. **Scale** each layer by C_min / (1 + C_b), so layers with many channels do not dominate (`scale=True`).
4. **Concatenate**: each position becomes a *composite pixel* mixing detail and semantics.
5. **Project** the composite pixels onto K concepts.

The paper's notation names the layers: N34 is blocks 3 and 4; N:9-12 is blocks 9 and 12.
`normalize=False` and `scale=False` reproduce the paper's ablations.

## Projectors

| `projector=` | |
|---|---|
| `None` / `"kmeans"` | k-means++ seeding and Lloyd iterations; prototypes are the centroids |
| `"bayesian"` | k-means with a Dirichlet prior on the weights, by Gibbs sampling; `k` is an upper bound |
| `"pca"` | principal components; a pixel goes to the component with the largest absolute projection |
| any object | scikit-learn style: clusterers (`predict` + `transform`, `predict_proba`, or `fit_predict`) or projections (`transform` only: PCA, NMF, ICA) |

**Bayesian k-means** is a finite approximation of a Dirichlet-process mixture:
π ~ Dirichlet(α/K), z ~ Categorical(π), x ~ N(μ_z, σ²I) with one shared σ². Assignments are
sampled with the cluster means integrated out, leaving each point out of its own cluster, so
duplicate clusters merge and the number of concepts used is learned. After `bayes_burn_in`
sweeps, posterior means are averaged and pixels assigned with them; concepts below 0.5% of the
posterior weight are dropped. It runs on `sample_size` composite pixels.

## From the grid to pixels

Concepts are extracted on the feature grid, coarser than the image. nave-xai interpolates the
*concept scores* to pixel resolution and takes the best concept per pixel: the explanation map.
With `upsample="nearest"` this gives the paper's blocky maps; `"linear"` and `"cubic"` smooth
boundaries. One concept can appear in several places; `segmentation()` gives the connected regions.

## Uncertainty

With `uncertainty=True`, every pixel gets H(softmax(scores / T)) / log K, the normalised entropy
of its concept assignment, from 0 (one concept clearly wins) to 1 (all equally likely). For
k-means, scores are minus the squared distances to the prototypes, so
1 - uncertainty = 1 - H(softmax(-d² / T)) / log K is the certainty. The temperature T is
`uncertainty_temp`, 0.1 by default; `"auto"` uses T = 1 for probabilistic projectors and otherwise
the median gap between the two best concept scores.

## Ordering and colours

After fitting, concepts are renumbered along a dendrogram of their prototypes (`reorder="ward"`
in `fit`). Ward linkage is exact from prototypes and concept sizes: the cost of merging two
clusters, n_a n_b / (n_a + n_b) |c_a - c_b|², depends only on their centroids and sizes.
`"average"` and `"single"` linkage use distances between prototypes. With groups, concepts are
ordered by group first, and every group gets a colour family.

`reorder(by="classes")` groups concepts under the class they correlate with most (from
`correlation`), unassigned ones last; `recolor=True` gives every class a colour and its concepts
shades of it (`color_vary`, `color_spread`).

## Patches and correlation

A *cell* is a connected region of an explanation map assigned to one concept. For every concept,
the **closest** cell (smallest mean distance of its composite pixels to the prototype) and the
**largest** cell of the fitting set are kept as `patches`: 32 x 32 views, dimmed outside the cell,
with a 1 px white line around it. `correlation` is the Pearson correlation, across images, between
a concept's proportion in an image and the image's class (the groups given to `fit`, else the
model's prediction).

## Concept-based saliency

`saliency(image)` explains a class with the encoder's own regions:

1. At every position of the deepest chosen layer (before pooling), take the cosine between the
   feature vector and the classifier's weight vector of the target class.
2. Explain that layer alone with NAVE and average the cosine over each region.
3. Move one layer shallower: upsample, explain with the cumulative layers (N4, then N34, then
   N234) and average per region again.
4. Upsample the result to the image.
