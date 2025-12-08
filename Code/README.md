# Code Folder Guide

This folder contains the training, evaluation, and utility scripts for an image denoising project. The pipeline trains a convolutional model (simple autoencoder or UNet) to remove synthetic noise from images.

## Directory at a glance

```
Code/
├─ train.py          # Main training loop; saves model weights as <model_name>.pth
├─ test.py           # Evaluation/visualization using saved weights; reports MSE and PSNR
├─ dataset.py        # Minimal image dataset wrapper with optional torchvision transforms
├─ models.py         # Denoising models: simple Encoder/Decoder and a UNet with skip connections
├─ utils.py          # Helpers: seeding, noise injection, PSNR, plotting
├─ test_loader.py    # Quick sanity check for data loading and noise functions
├─ Model_check.py    # Quick shape check for the autoencoder model
├─ download_data.py  # Utility to download/unzip a sample dataset into ../data
├─ app.py            # Placeholder for future app/inference entry point
└─ README.md         # This guide
```

## How the pieces fit together

- `dataset.py` defines `COCODataset`, which scans a directory for image files and returns tensors with optional transforms.
- `train.py` builds train/val/test splits, applies online noise with `utils.add_noise`, and trains a model from `models.py` using MSE loss and Adam. It saves weights to `<model.name()>.pth`.
- `test.py` recreates the same split deterministically, loads the saved weights, computes MSE and PSNR (`utils.calculate_psnr`), and saves a visualization via `utils.plot_denoising_result`.
- `models.py` provides two choices:
  - `Denoising_Model`: a simple encoder–decoder with transposed convolutions.
  - `UNet`: a lightweight UNet-style architecture with skip connections and a `name()` helper.
- `utils.py` includes reproducibility seeding, noise functions (`gaussian` and `salt_pepper`), PSNR, and plotting utilities.
- `test_loader.py` and `Model_check.py` are quick checks to ensure data and model tensor shapes behave as expected.
- `download_data.py` fetches and extracts a sample dataset (UT Zappos 50k) into the project’s `data/` directory.

## Data expectations

- Default scripts expect images under `../data/imgs` relative to this `Code/` folder.
- You can either:
  - Place your training images into `Final-Project-Group3/data/imgs/`, or
  - Edit the `DATA_PATH` constant at the top of `train.py` and `test.py` to point to your images folder.

Note: The `download_data.py` script downloads datasets into `../data/`. You may need to move or point `DATA_PATH` to where the images are extracted.

## Quickstart

1) Install dependencies (typical set used by the scripts):

```
pip install torch torchvision pillow matplotlib numpy tqdm requests
```

2) Train a model (UNet by default):

```
python train.py
```

This saves weights as `UNet.pth` (or `Denoising_Model.pth` if you switch models).

3) Evaluate on the test split and save visuals:

```
python test.py
```

This prints MSE/PSNR and saves an image grid like `result-UNet.png`.

4) Optional sanity checks:

```
python test_loader.py   # Verify data loading and noise injection
python Model_check.py   # Verify autoencoder I/O tensor shapes
python download_data.py # Download and extract sample datasets into ../data
```

## Configuration knobs

- In `train.py` and `test.py`:
  - `SEED`: reproducibility seed (must match for identical splits).
  - `DATA_PATH`: path to your images directory (default `../data/imgs`).
  - `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `NOISE_FACTOR`, `NUM_WORKERS`.
- Model selection: swap the model instantiation in `train.py`/`test.py` between `UNet()` and `Denoising_Model()`.
- Transforms (resize, random crop, flip, to tensor) are set near the top of `train.py`/`test.py` via `torchvision.transforms`.

## Notes

- Saved weights are named via `model.name()` to keep training and evaluation in sync.
- Noise is applied on-the-fly during training, so the model sees different perturbations each epoch.
- Images are assumed to be in range `[0, 1]` after `ToTensor()`; the models output with a final `Sigmoid()` to match that range.
