# Inspecting models

NAVE explanations reflect what an encoder learned, which makes them a diagnostic tool
(Sections 4 and 5 of the paper). Each recipe below is a notebook in
[`notebooks/inspecting/`](https://github.com/Ahcene-B/NAVE/tree/main/notebooks/inspecting).
Notebook 4 needs PyTorch; the others run with the built-in encoders.

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
