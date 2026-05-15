"""
Task 2: Train CycleGAN on apple2orange with default settings.

Downloads the dataset if not already present, then launches training
using the same hyperparameters as the original CycleGAN paper:
  - ResNet-9blocks generator, PatchGAN (basic) discriminator
  - Instance normalisation, no dropout
  - LSGAN objective, lr=0.0002, Adam (beta1=0.5)
  - 100 epochs fixed lr + 100 epochs linear decay (200 total)
  - lambda_A = lambda_B = 10, lambda_identity = 0.5
  - Image pool size = 50, batch size = 1
"""

import sys
import zipfile
from pathlib import Path

import requests


DATASET_NAME = "apple2orange"
DATASET_URL = f"http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/{DATASET_NAME}.zip"
DATASETS_DIR = Path(__file__).parent / "datasets"
DATASET_DIR = DATASETS_DIR / DATASET_NAME


def download_dataset():
    """Download and unzip apple2orange if not already present."""
    if DATASET_DIR.is_dir() and any(DATASET_DIR.iterdir()):
        print(f"[dataset] '{DATASET_DIR}' already exists – skipping download.")
        return

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATASETS_DIR / f"{DATASET_NAME}.zip"

    print(f"[dataset] Downloading {DATASET_URL} ...")
    response = requests.get(DATASET_URL, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = 100 * downloaded / total
                print(f"\r  {pct:.1f}% ({downloaded >> 20} / {total >> 20} MB)", end="", flush=True)
    print()

    print(f"[dataset] Extracting to {DATASETS_DIR} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATASETS_DIR)
    zip_path.unlink()
    print("[dataset] Done.")


def build_train_args():
    """Return sys.argv-style argument list for train.py with CycleGAN defaults."""
    return [
        "train.py",
        # --- data ---
        "--dataroot", str(DATASET_DIR),
        "--dataset_mode", "unaligned",         # CycleGAN uses unpaired images
        # --- model ---
        "--model", "cycle_gan",
        "--netG", "resnet_9blocks",             # generator: ResNet with 9 residual blocks
        "--netD", "basic",                      # discriminator: 70x70 PatchGAN
        "--norm", "instance",                   # instance normalisation
        "--no_dropout",                         # no dropout (CycleGAN paper default)
        "--gan_mode", "lsgan",                  # least-squares GAN objective
        "--init_type", "normal",
        "--init_gain", "0.02",
        "--ngf", "64",                          # generator base filters
        "--ndf", "64",                          # discriminator base filters
        # --- loss weights ---
        "--lambda_A", "10.0",                   # forward cycle-consistency weight
        "--lambda_B", "10.0",                   # backward cycle-consistency weight
        "--lambda_identity", "0.5",             # identity loss weight
        # --- optimiser ---
        "--lr", "0.0002",
        "--beta1", "0.5",
        "--pool_size", "50",                    # replay buffer to stabilise discriminator
        "--lr_policy", "linear",
        # --- schedule: 100 epochs fixed lr + 100 epochs linear decay ---
        "--n_epochs", "100",
        "--n_epochs_decay", "100",
        # --- data loading ---
        "--batch_size", "1",
        "--load_size", "286",
        "--crop_size", "256",
        "--preprocess", "resize_and_crop",
        "--num_threads", "4",
        # --- logging / saving ---
        "--name", "apple2orange_cyclegan_default",
        "--print_freq", "100",
        "--display_freq", "400",
        "--save_epoch_freq", "5",
        "--save_latest_freq", "5000",
        "--no_html",                            # skip visdom HTML (saves disk space)
    ]


if __name__ == "__main__":
    # Step 1: obtain the dataset
    download_dataset()

    # Step 2: build the argument list for train.py
    args = build_train_args()

    print("\n[train] Starting CycleGAN training with default settings.")
    print("[train] Command: python", " ".join(args), "\n")

    # Step 3: launch train.py as a subprocess so its __main__ guard fires correctly
    import subprocess
    result = subprocess.run(
        [sys.executable] + args,
        cwd=Path(__file__).parent,  # ensure relative paths in train.py resolve correctly
    )
    sys.exit(result.returncode)
