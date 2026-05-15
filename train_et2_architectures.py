"""
ET2: Architecture Design Choices

Runs a set of CycleGAN training experiments that vary the generator and
discriminator architecture.  Each config saves to its own checkpoint dir so
that ET3 metrics can be computed on all of them independently.

Generator ablation (generator depth):
  resnet_6blocks   -- 6 residual blocks   (lighter, fewer params)
  resnet_9blocks   -- 9 residual blocks   (paper default / baseline)
  resnet_12blocks  -- 12 residual blocks  (deeper, added to networks.py)
  unet_256         -- U-Net with 8 downsampling levels (skip-connection arch)

Filter-width ablation (on resnet_9blocks):
  ngf=32           -- narrow network, ~4x fewer params than default
  ngf=64           -- default
  ngf=128          -- wide network, ~4x more params than default

All configs use:
  - apple2orange dataset
  - basic (70x70 PatchGAN) discriminator
  - LSGAN objective, lr=0.0002, Adam (beta1=0.5)
  - 50 + 50 = 100 epochs (shorter than full 200 to allow comparison within
    GPU budget; note this in the report when comparing against Task 2 baseline)

How to run:
  python train_et2_architectures.py               # all configs
  python train_et2_architectures.py --configs resnet6 resnet12 unet256
  python train_et2_architectures.py --configs ngf32 ngf128

How to evaluate after training (ET3):
  python evaluation/evaluate.py --name et2_resnet6_ngf64  --dataroot ./datasets/apple2orange
  python evaluation/evaluate.py --name et2_unet256_ngf64  --dataroot ./datasets/apple2orange
  # (repeat for each config name listed below)
"""

import argparse
import sys
import subprocess
import time
from pathlib import Path

DATASET_DIR = Path(__file__).parent / "datasets" / "apple2orange"

# ---------------------------------------------------------------------------
# Experiment configurations
# Each entry becomes one training run.  'key' is used with --configs to
# select a subset on the command line.
# ---------------------------------------------------------------------------

N_EPOCHS       = 50   # fixed-lr phase
N_EPOCHS_DECAY = 50   # linear-decay phase  (100 total)

# Shared flags that are the same across all ET2 runs
COMMON_FLAGS = [
    "--dataroot",      str(DATASET_DIR),
    "--dataset_mode",  "unaligned",
    "--model",         "cycle_gan",
    "--norm",          "instance",
    "--no_dropout",
    "--gan_mode",      "lsgan",
    "--lr",            "0.0002",
    "--beta1",         "0.5",
    "--pool_size",     "50",
    "--lr_policy",     "linear",
    "--n_epochs",      str(N_EPOCHS),
    "--n_epochs_decay", str(N_EPOCHS_DECAY),
    "--batch_size",    "1",
    "--load_size",     "286",
    "--crop_size",     "256",
    "--preprocess",    "resize_and_crop",
    "--num_threads",   "4",
    "--lambda_A",      "10.0",
    "--lambda_B",      "10.0",
    "--lambda_identity", "0.5",
    "--print_freq",    "100",
    "--save_epoch_freq", "5",
    "--save_latest_freq", "5000",
    "--no_html",
]

CONFIGS = [
    # --- generator depth ablation ---
    {
        "key":   "resnet6",
        "name":  "et2_resnet6_ngf64",
        "flags": ["--netG", "resnet_6blocks", "--ngf", "64", "--ndf", "64", "--netD", "basic"],
        "note":  "ResNet-6: 6 residual blocks (lighter than paper default)",
    },
    {
        "key":   "resnet9",
        "name":  "et2_resnet9_ngf64",
        "flags": ["--netG", "resnet_9blocks", "--ngf", "64", "--ndf", "64", "--netD", "basic"],
        "note":  "ResNet-9: paper default architecture (baseline for ET2 comparison)",
    },
    {
        "key":   "resnet12",
        "name":  "et2_resnet12_ngf64",
        "flags": ["--netG", "resnet_12blocks", "--ngf", "64", "--ndf", "64", "--netD", "basic"],
        "note":  "ResNet-12: 12 residual blocks (deeper; added to networks.py for ET2)",
    },
    # --- architectural family change ---
    {
        "key":   "unet256",
        "name":  "et2_unet256_ngf64",
        "flags": ["--netG", "unet_256", "--ngf", "64", "--ndf", "64", "--netD", "basic"],
        "note":  "U-Net-256: skip connections at every resolution vs. ResNet bottleneck",
    },
    # --- filter-width ablation (architecture fixed to resnet_9blocks) ---
    {
        "key":   "ngf32",
        "name":  "et2_resnet9_ngf32",
        "flags": ["--netG", "resnet_9blocks", "--ngf", "32", "--ndf", "32", "--netD", "basic"],
        "note":  "ResNet-9 narrow (ngf=32): ~4x fewer params than default",
    },
    {
        "key":   "ngf128",
        "name":  "et2_resnet9_ngf128",
        "flags": ["--netG", "resnet_9blocks", "--ngf", "128", "--ndf", "64", "--netD", "basic"],
        "note":  "ResNet-9 wide (ngf=128): ~4x more generator params than default",
    },
    # --- optional discriminator ablation (uncomment to include) ---
    # {
    #     "key":   "pixel_d",
    #     "name":  "et2_resnet9_pixel_d",
    #     "flags": ["--netG", "resnet_9blocks", "--ngf", "64", "--ndf", "64", "--netD", "pixel"],
    #     "note":  "ResNet-9 + 1x1 PixelGAN discriminator (no spatial context)",
    # },
    # {
    #     "key":   "nlayer5_d",
    #     "name":  "et2_resnet9_nlayer5_d",
    #     "flags": ["--netG", "resnet_9blocks", "--ngf", "64", "--ndf", "64",
    #               "--netD", "n_layers", "--n_layers_D", "5"],
    #     "note":  "ResNet-9 + deeper PatchGAN discriminator (5 conv layers)",
    # },
]

# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="ET2: Run architecture ablation experiments")
    parser.add_argument(
        "--configs", nargs="+", default=None,
        help="Subset of config keys to run (e.g. resnet6 unet256). "
             "Omit to run all. Available: " + ", ".join(c["key"] for c in CONFIGS),
    )
    return parser.parse_args()


def build_command(config):
    """Assemble the full argument list for train.py."""
    return ["train.py", "--name", config["name"]] + COMMON_FLAGS + config["flags"]


def run_config(config):
    args = build_command(config)
    print(f"\n{'=' * 70}")
    print(f"  Running: {config['name']}")
    print(f"  Note:    {config['note']}")
    print(f"  Command: python {' '.join(args)}")
    print(f"{'=' * 70}\n")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable] + args,
        cwd=Path(__file__).parent,
    )
    elapsed = time.time() - t0

    hours, rem = divmod(int(elapsed), 3600)
    mins, secs = divmod(rem, 60)
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n  [{config['name']}] Finished in {hours:02d}:{mins:02d}:{secs:02d} -- {status}")
    return result.returncode == 0


def check_dataset():
    if not DATASET_DIR.is_dir() or not any(DATASET_DIR.iterdir()):
        print(f"[ERROR] Dataset not found at {DATASET_DIR}")
        print("  Run train_task2_apple2orange.py first (it downloads the dataset).")
        sys.exit(1)


def print_summary(results):
    print(f"\n{'=' * 70}")
    print("  ET2 Training Summary")
    print(f"{'=' * 70}")
    for name, ok in results.items():
        mark = "OK   " if ok else "FAIL "
        print(f"  {mark}  checkpoints/{name}/")
    print(f"{'=' * 70}")
    print("\nTo evaluate with ET3 metrics, run e.g.:")
    for name in results:
        print(f"  python evaluation/evaluate.py --name {name} --dataroot ./datasets/apple2orange")


if __name__ == "__main__":
    args = parse_args()
    check_dataset()

    # Filter to requested subset (or run all)
    if args.configs is not None:
        requested = set(args.configs)
        selected = [c for c in CONFIGS if c["key"] in requested]
        missing = requested - {c["key"] for c in selected}
        if missing:
            print(f"[WARNING] Unknown config keys: {missing}")
    else:
        selected = CONFIGS

    if not selected:
        print("[ERROR] No configs selected.")
        sys.exit(1)

    print(f"Running {len(selected)} ET2 experiment(s): {[c['key'] for c in selected]}")
    print(f"Each run: {N_EPOCHS} + {N_EPOCHS_DECAY} = {N_EPOCHS + N_EPOCHS_DECAY} epochs\n")

    results = {}
    for config in selected:
        ok = run_config(config)
        results[config["name"]] = ok

    print_summary(results)
