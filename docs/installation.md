# Installation Guide

## Step 1: Install Conda

Download and install Miniconda (recommended) or Anaconda:

- **Mac/Linux**: https://docs.conda.io/en/latest/miniconda.html
- **Windows**: https://docs.conda.io/en/latest/miniconda.html

Run the installer and follow the prompts. Restart your terminal after installation.

Verify conda is installed:
```bash
conda --version
```

## Step 2: Download ProxylFit

```bash
git clone https://github.com/philadamson93/ProxylFit.git
cd ProxylFit
```

Or download and unzip from GitHub: https://github.com/philadamson93/ProxylFit

## Step 3: Create the Conda Environment

```bash
conda create -n proxyl python=3.10
```

When prompted, type `y` to proceed.

## Step 4: Activate the Environment

```bash
conda activate proxyl
```

Your terminal prompt should now show `(proxyl)` at the beginning.

**Note**: You need to activate the environment every time you open a new terminal.

## Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 6: Run ProxylFit

```bash
python -m proxyl_analysis
```

## Quick Reference

Every time you want to run ProxylFit:

```bash
cd ProxylFit
conda activate proxyl
python -m proxyl_analysis
```
