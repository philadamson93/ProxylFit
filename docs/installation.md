# Installation Guide

## Step 1: Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Reload your shell so uv is on PATH:

```bash
exec $SHELL
```

Verify:

```bash
uv --version
```

## Step 2: Get the repo

```bash
git clone https://github.com/philadamson93/ProxylFit.git
cd ProxylFit
```

## Step 3: Create virtual environment and install dependencies

```bash
uv venv --python 3.10
uv pip install -r requirements.txt
```

This creates `.venv/` inside the repo.

## Step 4: Run ProxylFit

You can run without "activating" anything:

```bash
uv run python -m proxyl_analysis
```

That's it.

---

## Quick Reference

Every time you want to run ProxylFit:

```bash
cd ProxylFit
uv run python -m proxyl_analysis
```

---

## Troubleshooting

### Qt platform plugin error (macOS)

If you see an error like:
```
qt.qpa.plugin: Could not find the Qt platform plugin "cocoa"
```

**Cause:** Conda is interfering with the uv environment.

**Fix:** Deactivate conda before running:

```bash
conda deactivate
cd ProxylFit
uv run python -m proxyl_analysis
```

Or check if conda is active:
```bash
echo $CONDA_PREFIX
```

If it shows a path, conda is active and needs to be deactivated.

### Alternative: Run in fully isolated mode

If conda continues to interfere, create a fresh environment:

```bash
cd ProxylFit
rm -rf .venv
uv venv --python 3.10
uv pip install -r requirements.txt
uv run python -m proxyl_analysis
```
