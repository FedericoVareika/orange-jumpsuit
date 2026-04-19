@echo off
echo === Staging Python build files ===
copy python\pyproject.toml .
copy python\setup.py .
copy python\MANIFEST.in .

echo === Building Python Package ===
REM Clean old build artifacts to ensure a fresh run
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist python\orange_jumpsuit.egg-info rmdir /s /q python\orange_jumpsuit.egg-info
for /d %%i in (*.egg-info) do rmdir /s /q "%%i"

python -m build

echo === Cleaning up root directory ===
del pyproject.toml setup.py MANIFEST.in
if exist build rmdir /s /q build
for /d %%i in (*.egg-info) do rmdir /s /q "%%i"

echo === Done! Packages are in the dist\ folder ===
dir dist\
