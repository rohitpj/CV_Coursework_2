import sys
import re
import subprocess
import zipfile
import requests
from pathlib import Path

DATASET_URL = "http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/apple2orange.zip"
DATASET_DIR = "./datasets/apple2orange"
CHECKPOINTS_DIR = "./checkpoints"

# Hyperparameter tuning: vary learning rate, everything else at paper default
# (name, lr)
CONFIGS = [
    ("task2_lr0001",                  "0.0001"),
    ("apple2orange_cyclegan_default", "0.0002"),  # paper default
    ("task2_lr0004",                  "0.0004"),
]


def download_dataset():
    if Path(DATASET_DIR).exists():
        print("Dataset already exists, skipping download.")
        return

    Path("./datasets").mkdir(exist_ok=True)
    print("Downloading apple2orange dataset...")
    r = requests.get(DATASET_URL, stream=True)
    with open("./datasets/apple2orange.zip", "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

    print("Extracting...")
    with zipfile.ZipFile("./datasets/apple2orange.zip") as zf:
        zf.extractall("./datasets")
    Path("./datasets/apple2orange.zip").unlink()
    print("Done.")


def cleanup_checkpoints(name):
    checkpoint_dir = Path(CHECKPOINTS_DIR) / name
    for f in checkpoint_dir.glob("*_net_*.pth"):
        if re.match(r"^\d+_net_", f.name):
            f.unlink()
    print(f"  Removed epoch checkpoints, kept latest_net_*.pth.")


def train(name, lr):
    cmd = [
        sys.executable, "train.py",
        "--dataroot",        DATASET_DIR,
        "--name",            name,
        "--model",           "cycle_gan",
        "--dataset_mode",    "unaligned",
        "--netG",            "resnet_9blocks",
        "--netD",            "basic",
        "--ngf",             "64",
        "--ndf",             "64",
        "--norm",            "instance",
        "--no_dropout",
        "--gan_mode",        "lsgan",
        "--lr",              lr,
        "--beta1",           "0.5",
        "--pool_size",       "50",
        "--n_epochs",        "30",
        "--n_epochs_decay",  "30",
        "--batch_size",      "1",
        "--load_size",       "286",
        "--crop_size",       "256",
        "--lambda_A",        "10.0",
        "--lambda_B",        "10.0",
        "--lambda_identity", "0.5",
        "--save_epoch_freq", "10",
        "--no_html",
    ]

    print(f"\nTraining: {name}  (lr={lr})")
    subprocess.run(cmd, cwd=Path(__file__).parent)
    cleanup_checkpoints(name)


if __name__ == "__main__":
    download_dataset()
    for name, lr in CONFIGS:
        train(name, lr)
