#!/bin/sh

echo "=== Staging Python build files ==="
cp python/pyproject.toml .
cp python/setup.py .
cp python/MANIFEST.in .

echo "=== Building Python Package ==="
# Clean old build artifacts to ensure a fresh run
rm -rf build/ dist/ python/orange_jumpsuit.egg-info/ *.egg-info/
python -m build

echo "=== Cleaning up root directory ==="
rm pyproject.toml setup.py MANIFEST.in
rm -rf build/ *.egg-info/

echo "=== Done! Packages are in the dist/ folder ==="
ls -l dist/
