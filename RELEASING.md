# Releasing nave-xai

1. **Rebuild the graphs** only if `tools/build_graphs.py` changed (needs `pip install onnx`):

   ```bash
   python tools/build_graphs.py
   ```

2. **Test**: `pytest`, and `python -m mkdocs build --strict` for the docs.

3. **Version**: raise `version` in `pyproject.toml` and `__version__` in `src/nave_xai/__init__.py`.

4. **Build and upload**:

   ```bash
   rm -rf dist && python -m build
   twine check dist/*
   twine upload --repository testpypi dist/*                     # try first
   twine upload dist/*
   ```

5. **Tag**: `git tag vX.Y.Z && git push origin vX.Y.Z`, with the new version.
