"""
ET2: Architecture Design Choices

Compares the default ResNet-9blocks generator against a U-Net-256 generator.

ResNet-9blocks (baseline):
  Processes images through an encoder-bottleneck-decoder path with 9 residual
  blocks at the bottleneck.  The bottleneck has no direct path for spatial
  detail from input to output, so the generator must learn all structure from
  compressed features -- giving it more freedom to change style and appearance.

U-Net-256 (ET2 change):
  Uses skip connections that concatenate encoder feature maps directly into
  the decoder at every resolution level.  Spatial detail is preserved by
  construction, which limits how much the generator can change the structure
  of the input image.  This trade-off is interesting for style transfer:
  better structure preservation vs. less stylistic freedom.

Both configs use:
  - apple2orange dataset
  - basic (70x70 PatchGAN) discriminator, ndf=64
  - ngf=64, LSGAN, lr=0.0002, Adam (beta1=0.5)
  - 50 + 50 = 100 epochs total

Usage:
  python train_et2_architectures.py              # run both
  python train_et2_architectures.py --configs resnet9
  python train_et2_architectures.py --configs unet256

Evaluate with ET3 after training:
  python evaluation/evaluate.py --name et2_resnet9  --dataroot ./datasets/apple2orange
  python evaluation/evaluate.py --name et2_unet256  --dataroot ./datasets/apple2orange
"""

import argparse
import sys
import subprocess
import time
from pathlib import Path

DATASET_DIR = Path(__file__).parent / "datasets" / "apple2orange"

N_EPOCHS       = 50
N_EPOCHS_DECAY = 50

COMMON_FLAGS = [
    "--dataroot",        str(DATASET_DIR),
    "--dataset_mode",    "unaligned",
    "--model",           "cycle_gan",
    "--netD",            "basic",
    "--ndf",             "64",
    "--ngf",             "64",
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

CONFIGS = [
    {
        "key":   "resnet9",
        "name":  "et2_resnet9",
        "netG":  "resnet_9blocks",
        "note":  "ResNet-9blocks: paper default, bottleneck architecture (ET2 anchor)",
    },
    {
        "key":   "unet256",
        "name":  "et2_unet256",
        "netG":  "unet_256",
        "note":  "U-Net-256: skip connections at 8 resolution levels (ET2 change)",
    },
]


def parse_args():
    keys = [c["key"] for c in CONFIGS]
    parser = argparse.ArgumentParser(description="ET2: ResNet-9 vs U-Net-256 generator comparison")
    parser.add_argument(
        "--configs", nargs="+", default=None,
        choices=keys,
        help=f"Which configs to run (default: all). Choices: {keys}",
    )
    return parser.parse_args()


def run_config(config):
    args = ["train.py", "--name", config["name"], "--netG", config["netG"]] + COMMON_FLAGS

    print(f"\n{'=' * 68}")
    print(f"  Config : {config['key']}  ({config['name']})")
    print(f"  Note   : {config['note']}")
    print(f"  netG   : {config['netG']}")
    print(f"  Epochs : {N_EPOCHS} fixed + {N_EPOCHS_DECAY} decay = {N_EPOCHS + N_EPOCHS_DECAY} total")
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
    print("  ET2 Summary")
    print(f"{'=' * 68}")
    for name, ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'}  checkpoints/{name}/")
    print(f"\n  Evaluate with ET3:")
    for name in results:
        print(f"    python evaluation/evaluate.py --name {name} --dataroot ./datasets/apple2orange")
    print(f"{'=' * 68}")


if __name__ == "__main__":
    args = parse_args()

    if not DATASET_DIR.is_dir() or not any(DATASET_DIR.iterdir()):
        print(f"[ERROR] Dataset not found at {DATASET_DIR}")
        print("  Run train_task2_apple2orange.py first to download the dataset.")
        sys.exit(1)

    selected = CONFIGS if args.configs is None else [c for c in CONFIGS if c["key"] in args.configs]

    print(f"Running {len(selected)} ET2 experiment(s): {[c['key'] for c in selected]}")

    results = {}
    for config in selected:
        ok = run_config(config)
        results[config["name"]] = ok

    print_summary(results)
