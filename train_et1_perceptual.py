"""
ET1: Loss Function Engineering -- VGG Perceptual Cycle Loss

Adds a perceptual loss term to the CycleGAN cycle-consistency loss.
Instead of penalising pixel-wise L1 distance between the reconstructed image
and the original, a frozen VGG16 network extracts intermediate feature maps
and the L1 distance is computed in feature space.  This captures higher-level
texture and semantic similarity that correlates better with human perception.

The perceptual loss is applied to the cycle-reconstruction pairs:
  cycle_A_perceptual = lambda_p * VGG_L1(G_B(G_A(real_A)), real_A)
  cycle_B_perceptual = lambda_p * VGG_L1(G_A(G_B(real_B)), real_B)

Three ablation axes:
  1. lambda_perceptual : how much weight to give the perceptual term
                         (0.1, 1.0, 10.0 -- alongside pixel L1)
  2. perceptual_layers : which VGG layers to extract from
                         shallow = relu1_2 + relu2_2  (texture focus)
                         deep    = relu3_3 + relu4_3  (semantic focus)
                         all     = all four layers
  3. no_cycle_l1       : whether to keep pixel L1 or replace it entirely
                         add     = perceptual added to L1  (default)
                         replace = perceptual replaces L1

Baseline for comparison: apple2orange_cyclegan_default  (Task 2 checkpoint,
lambda_perceptual=0, standard L1 cycle loss).

Usage:
  python train_et1_perceptual.py              # all configs
  python train_et1_perceptual.py --configs lam0p1 lam1 lam10
  python train_et1_perceptual.py --configs shallow deep
  python train_et1_perceptual.py --configs replace

Evaluate with ET3 after training:
  python evaluation/evaluate.py --name et1_lam0p1  --dataroot ./datasets/apple2orange
  python evaluation/evaluate.py --name et1_lam1    --dataroot ./datasets/apple2orange
  # (repeat for each config name below)
"""

import argparse
import sys
import subprocess
import time
from pathlib import Path

DATASET_DIR = Path(__file__).parent / "datasets" / "apple2orange"

N_EPOCHS       = 50
N_EPOCHS_DECAY = 50

# Flags shared by all ET1 runs (same architecture as Task 2 baseline)
COMMON_FLAGS = [
    "--dataroot",        str(DATASET_DIR),
    "--dataset_mode",    "unaligned",
    "--model",           "cycle_gan",
    "--netG",            "resnet_9blocks",
    "--netD",            "basic",
    "--ngf",             "64",
    "--ndf",             "64",
    "--norm",            "instance",
    "--no_dropout",
    "--gan_mode",        "lsgan",
    "--lr",              "0.0002",
    "--beta1",           "0.5",
    "--pool_size",       "50",
    "--lr_policy",       "linear",
    "--n_epochs",        str(N_EPOCHS),
    "--n_epochs_decay",  str(N_EPOCHS_DECAY),
    "--batch_size",      "1",
    "--load_size",       "286",
    "--crop_size",       "256",
    "--preprocess",      "resize_and_crop",
    "--num_threads",     "4",
    "--lambda_A",        "10.0",
    "--lambda_B",        "10.0",
    "--lambda_identity", "0.5",
    "--print_freq",      "100",
    "--save_epoch_freq", "5",
    "--save_latest_freq", "5000",
    "--no_html",
]

# ---------------------------------------------------------------------------
# Ablation configs
# ---------------------------------------------------------------------------

CONFIGS = [
    # -- lambda ablation (all VGG layers, add alongside L1) --
    {
        "key":  "lam0p1",
        "name": "et1_lam0p1",
        "extra": ["--lambda_perceptual", "0.1", "--perceptual_layers", "all"],
        "note": "lambda_p=0.1, all VGG layers, added alongside L1",
    },
    {
        "key":  "lam1",
        "name": "et1_lam1",
        "extra": ["--lambda_perceptual", "1.0", "--perceptual_layers", "all"],
        "note": "lambda_p=1.0, all VGG layers, added alongside L1",
    },
    {
        "key":  "lam10",
        "name": "et1_lam10",
        "extra": ["--lambda_perceptual", "10.0", "--perceptual_layers", "all"],
        "note": "lambda_p=10.0, all VGG layers, added alongside L1",
    },
    # -- layer-depth ablation (lambda=1.0, add alongside L1) --
    {
        "key":  "shallow",
        "name": "et1_shallow",
        "extra": ["--lambda_perceptual", "1.0", "--perceptual_layers", "shallow"],
        "note": "lambda_p=1.0, shallow layers (relu1_2+relu2_2), added alongside L1",
    },
    {
        "key":  "deep",
        "name": "et1_deep",
        "extra": ["--lambda_perceptual", "1.0", "--perceptual_layers", "deep"],
        "note": "lambda_p=1.0, deep layers (relu3_3+relu4_3), added alongside L1",
    },
    # -- replace vs. add ablation (lambda=1.0, all layers) --
    {
        "key":  "replace",
        "name": "et1_replace",
        "extra": ["--lambda_perceptual", "1.0", "--perceptual_layers", "all", "--no_cycle_l1"],
        "note": "lambda_p=1.0, all VGG layers, REPLACES pixel L1 (no L1 cycle loss)",
    },
]


def parse_args():
    keys = [c["key"] for c in CONFIGS]
    parser = argparse.ArgumentParser(description="ET1: VGG perceptual loss ablation")
    parser.add_argument(
        "--configs", nargs="+", default=None,
        choices=keys,
        help=f"Configs to run (default: all). Choices: {keys}",
    )
    return parser.parse_args()


def run_config(config):
    args = ["train.py", "--name", config["name"]] + COMMON_FLAGS + config["extra"]

    print(f"\n{'=' * 68}")
    print(f"  Config : {config['key']}  ({config['name']})")
    print(f"  Note   : {config['note']}")
    print(f"  Epochs : {N_EPOCHS} + {N_EPOCHS_DECAY} = {N_EPOCHS + N_EPOCHS_DECAY} total")
    print(f"{'=' * 68}\n")

    t0 = time.time()
    result = subprocess.run([sys.executable] + args, cwd=Path(__file__).parent)
    elapsed = time.time() - t0

    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n  [{config['name']}] Done in {h:02d}:{m:02d}:{s:02d} -- {status}")
    return result.returncode == 0


def print_summary(results):
    print(f"\n{'=' * 68}")
    print("  ET1 Summary")
    print(f"{'=' * 68}")
    for name, ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'}  checkpoints/{name}/")
    print(f"\n  Baseline (Task 2): checkpoints/apple2orange_cyclegan_default/")
    print(f"\n  Evaluate with ET3:")
    print(f"    python evaluation/evaluate.py --name apple2orange_cyclegan_default "
          f"--dataroot ./datasets/apple2orange")
    for name in results:
        print(f"    python evaluation/evaluate.py --name {name} "
              f"--dataroot ./datasets/apple2orange")
    print(f"{'=' * 68}")


if __name__ == "__main__":
    args = parse_args()

    if not DATASET_DIR.is_dir() or not any(DATASET_DIR.iterdir()):
        print(f"[ERROR] Dataset not found at {DATASET_DIR}")
        print("  Run train_task2_apple2orange.py first to download the dataset.")
        sys.exit(1)

    selected = CONFIGS if args.configs is None else [c for c in CONFIGS if c["key"] in args.configs]

    print(f"Running {len(selected)} ET1 experiment(s): {[c['key'] for c in selected]}")
    print("Note: VGG16 weights will be downloaded on first run (~528 MB).\n")

    results = {}
    for config in selected:
        ok = run_config(config)
        results[config["name"]] = ok

    print_summary(results)
