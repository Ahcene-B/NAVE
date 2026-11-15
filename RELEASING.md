# Releasing nave-xai

1. **Ship the ViT-Ti/16 weights** (about 10 MB, needed once per machine, not committed to git):

   ```bash
   python tools/fetch_shipped_weights.py
   ```

   This writes `src/nave_xai/models/weights/vitt16_imagenet.npz`. Without it, the package still
   works but downloads the weights on first use, like the other models.

2. **Rebuild the graphs** only if `tools/build_graphs.py` changed (needs `pip install onnx`):

   ```bash
   python tools/build_graphs.py
   ```

3. **Test**: `pytest`, and `python -m mkdocs build --strict` for the docs.

4. **Version**: raise `version` in `pyproject.toml` and `__version__` in `src/nave_xai/__init__.py`.

5. **Build and upload**:

   ```bash
   rm -rf dist && python -m build
   python -m zipfile -l dist/*.whl | grep vitt16_imagenet.npz     # the weights are in the wheel
   twine check dist/*
   twine upload --repository testpypi dist/*                     # try first
   twine upload dist/*
   ```

6. **Tag**: `git tag v0.1.0 && git push origin v0.1.0`.
