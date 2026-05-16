import sys
import re
import shutil
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


def save_best_model(name):
    checkpoint_dir = Path(CHECKPOINTS_DIR) / name
    log_path = checkpoint_dir / "loss_log.txt"
    if not log_path.exists():
        print("  No loss log found, skipping best model save.")
        return

    # Parse per-iteration losses from loss_log.txt and average per epoch
    epoch_losses = {}
    with open(log_path) as f:
        for line in f:
            epoch_m = re.search(r'\(epoch: (\d+)', line)
            if not epoch_m:
                continue
            epoch = int(epoch_m.group(1))
            vals = dict(re.findall(r'(\w+): ([\d.e+-]+)', line))
            g_loss = (float(vals.get("G_A", 0)) + float(vals.get("G_B", 0))
                      + float(vals.get("cycle_A", 0)) + float(vals.get("cycle_B", 0)))
            epoch_losses.setdefault(epoch, []).append(g_loss)

    if not epoch_losses:
        print("  Could not parse losses, skipping best model save.")
        return

    avg = {e: sum(v) / len(v) for e, v in epoch_losses.items()}

    # Find which epoch checkpoints were actually saved to disk
    saved_epochs = set()
    for f in checkpoint_dir.glob("*_net_G_A.pth"):
        m = re.match(r"^(\d+)_net_G_A\.pth$", f.name)
        if m:
            saved_epochs.add(int(m.group(1)))

    if not saved_epochs:
        print("  No epoch checkpoints found, skipping best model save.")
        return

    best_epoch = min(saved_epochs, key=lambda e: avg.get(e, float("inf")))
    print(f"  Best epoch: {best_epoch}  (avg G+cycle loss: {avg.get(best_epoch, 0):.4f})")

    for net in ["G_A", "G_B", "D_A", "D_B"]:
        src = checkpoint_dir / f"{best_epoch}_net_{net}.pth"
        dst = checkpoint_dir / f"best_net_{net}.pth"
        if src.exists():
            shutil.copy2(src, dst)

    # Delete all epoch-numbered checkpoints; keep latest_net_*.pth and best_net_*.pth
    for f in checkpoint_dir.glob("*_net_*.pth"):
        if re.match(r"^\d+_net_", f.name):
            f.unlink()

    print(f"  Saved best_net_*.pth (epoch {best_epoch}), removed epoch checkpoints.")


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
        "--n_epochs",        "100",
        "--n_epochs_decay",  "100",
        "--batch_size",      "1",
        "--load_size",       "286",
        "--crop_size",       "256",
        "--lambda_A",        "10.0",
        "--lambda_B",        "10.0",
        "--lambda_identity", "0.5",
        "--save_epoch_freq", "5",
        "--no_html",
    ]

    print(f"\nTraining: {name}  (lr={lr})")
    subprocess.run(cmd, cwd=Path(__file__).parent)
    save_best_model(name)


if __name__ == "__main__":
    download_dataset()
    for name, lr in CONFIGS:
        train(name, lr)
