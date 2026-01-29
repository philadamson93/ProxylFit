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
