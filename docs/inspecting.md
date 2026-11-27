# Inspecting models

NAVE explanations reflect what an encoder learned, which makes them a diagnostic tool
(Sections 4 and 5 of the paper). Each recipe below is a notebook in
[`notebooks/inspecting/`](https://github.com/Ahcene-B/NAVE/tree/main/notebooks/inspecting).
Notebooks 4 and 9 need PyTorch; the others run with the built-in encoders.

| Notebook | Question |
|---|---|
| `01_training.ipynb` | How do random, supervised (ImageNet) and self-supervised (DINO) training shape the concepts? |
| `02_architectures.ipynb` | Do a ResNet and a ViT decompose the same image the same way? |
| `03_depth.ipynb` | How do concepts change from shallow to deep blocks, and where is information compressed? |
| `04_vit_registers.ipynb` | Do high-norm token artifacts of ViTs form concepts, and do registers remove them? (PyTorch) |
| `05_shortcuts.ipynb` | Does a shortcut (a watermark) take over the explanation? |
| `06_concept_features.ipynb` | Are concept proportions informative enough for a classifier probe? |
| `07_localization.ipynb` | Can a concept's region localize an object (inner and outer boxes)? |
| `08_uncertainty.ipynb` | Where does the encoder hesitate between concepts? |
| `09_causality.ipynb` | Do the concepts that come with a class actually drive its score? (PyTorch) |

## Concepts and causality

NAVE's correlation tells which concepts come with a class, not whether the model uses them. `09_causality.ipynb`
asks the causal question with interventions, for a pretrained ResNet50 and a ViT-S/16. At one block, a forward hook
replaces the tokens of a concept with those of other images at the same positions (resampling ablation) and lets the
network run on; the drop in the true class's logit is the concept's effect. Replacing as many random tokens the same
way gives a control, and a concept's causal effect is its effect beyond that control, averaged over the images where
it appears, with bootstrap intervals. The notebook then sets correlation against causal effect, concept by concept,
and shows where they disagree: context that comes with a class but is ignored, and features the model uses without
their being specific to a class.

The effect is causal for the model, not for the world, and interventions shift the activations' distribution; the
notebook ends with these caveats.
