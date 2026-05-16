import sys
import re
import zipfile
import subprocess
import requests
from pathlib import Path

from train import main as train_main

_REPO = Path(__file__).resolve().parent
DATASET_URL     = "http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/apple2orange.zip"
DATASET_DIR     = str(_REPO / "datasets" / "apple2orange")
CHECKPOINTS_DIR = str(_REPO / "checkpoints")
RESULTS_CSV     = str(_REPO / "evaluation" / "task2_results.csv")

SEED        = 42
KEEP_EPOCHS = {20, 40, 60, 80, 100}

# Epoch budget: 50 + 50 = 100 total.
# The repo default is 100 + 100 = 200; we reduce it due to compute constraints.
# This budget is used identically in ET1/ET2/ET4/ET5 so all comparisons are
# valid against the Task 2 baseline. State this clearly in the report.
N_EPOCHS       = 50
N_EPOCHS_DECAY = 50

# (name, lr)
# task2_default is the Task 2 deliverable — paper default lr, lsgan, resnet_9blocks.
# The lr variants are supplementary parameter-sensitivity runs; label them as such
# in the report (they are not the Task 2 baseline itself).
CONFIGS = [
    ("task2_lr0001",  "0.0001"),   # lr sensitivity — below default
    ("task2_default", "0.0002"),   # Task 2 baseline
    ("task2_lr0004",  "0.0004"),   # lr sensitivity — above default
]


def download_dataset():
    if Path(DATASET_DIR).exists():
        print("Dataset already exists, skipping download.")
        return

    zip_path = _REPO / "datasets" / "apple2orange.zip"
    (_REPO / "datasets").mkdir(exist_ok=True)
    print(f"Downloading apple2orange dataset from {DATASET_URL} ...")
    try:
        r = requests.get(DATASET_URL, stream=True, timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Download failed: {e}\n"
            "If the server is unreachable, place the dataset manually at "
            f"{DATASET_DIR} (trainA/, trainB/, testA/, testB/ subdirs)."
        ) from e

    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

    print("Verifying and extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            zip_path.unlink()
            raise RuntimeError(
                f"Downloaded zip is corrupt (first bad file: {bad}). "
                "Delete datasets/ and re-run."
            )
        zf.extractall(_REPO / "datasets")
    zip_path.unlink()
    print("Dataset ready.")


def cleanup_checkpoints(name):
    checkpoint_dir = Path(CHECKPOINTS_DIR) / name
    if not checkpoint_dir.exists():
        return
    removed = 0
    for f in checkpoint_dir.glob("*_net_*.pth"):
        m = re.match(r"^(\d+)_net_", f.name)
        if m and int(m.group(1)) not in KEEP_EPOCHS:
            f.unlink()
            removed += 1
    print(f"  Kept epoch checkpoints {sorted(KEEP_EPOCHS)}, removed {removed} intermediate .pth files.")


def evaluate(name):
    cmd = [
        sys.executable, str(_REPO / "evaluate.py"),
        "--name",       name,
        "--dataroot",   DATASET_DIR,
        "--netG",       "resnet_9blocks",
        "--ngf",        "64",
        "--output_csv", RESULTS_CSV,
    ]
    print(f"  Evaluating {name}...")
    result = subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: evaluate.py failed for {name}:\n{result.stderr[-800:]}")
    else:
        for line in result.stdout.splitlines():
            if any(k in line for k in ("FID", "KID", "LPIPS", "SSIM", "Appended", "Results")):
                print(f"  {line.strip()}")


def train(name, lr):
    args = [
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
        "--n_epochs",        str(N_EPOCHS),
        "--n_epochs_decay",  str(N_EPOCHS_DECAY),
        "--batch_size",      "1",
        "--load_size",       "286",
        "--crop_size",       "256",
        "--lambda_A",        "10.0",
        "--lambda_B",        "10.0",
        "--lambda_identity", "0.5",
        "--save_epoch_freq", "5",
        "--seed",            str(SEED),
        "--no_html",
    ]

    print(
        f"\n{'='*70}\n"
        f"Training : {name}\n"
        f"  lr={lr}, epochs={N_EPOCHS}+{N_EPOCHS_DECAY}, seed={SEED}\n"
        f"{'='*70}"
    )
    train_main(args)
    cleanup_checkpoints(name)
    evaluate(name)


if __name__ == "__main__":
    download_dataset()
    for name, lr in CONFIGS:
        train(name, lr)
