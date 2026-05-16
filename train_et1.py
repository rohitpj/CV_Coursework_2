import sys
import re
import subprocess
from pathlib import Path

from train import train

_REPO = Path(__file__).resolve().parent
DATASET_DIR     = str(_REPO / "datasets" / "apple2orange")
CHECKPOINTS_DIR = str(_REPO / "checkpoints")
RESULTS_CSV     = str(_REPO / "evaluation" / "et1_results.csv")

SEED = 42
KEEP_EPOCHS = {20, 40, 60, 80, 100}

# (name, lambda_perceptual, perceptual_layers, no_cycle_l1, lr)
CONFIGS = [
    ("et1_baseline",        0.0,  "all",     False, "0.0002"),

    # ── Lambda ablation ──────────────────────────────────────────────────────────
    ("et1_lam0p1",          0.1,  "all",     False, "0.0002"),
    ("et1_lam1",            1.0,  "all",     False, "0.0002"),
    ("et1_lam10",          10.0,  "all",     False, "0.0002"),  
    # ── Layer ablation ───────────────────────────────────────────────────────────
    ("et1_shallow",         1.0,  "shallow", False, "0.0002"),
    ("et1_deep",            1.0,  "deep",    False, "0.0002"),
    ("et1_replace_lam10",  10.0,  "all",     True,  "0.0002"),

]


def cleanup_checkpoints(name):

    checkpoint_dir = Path(CHECKPOINTS_DIR) / name
    removed = 0
    for f in checkpoint_dir.glob("*_net_*.pth"):
        m = re.match(r"^(\d+)_net_", f.name)
        if m and int(m.group(1)) not in KEEP_EPOCHS:
            f.unlink()
            removed += 1
    print(f"  Kept epoch checkpoints {sorted(KEEP_EPOCHS)}, removed {removed} intermediate .pth files.")


def evaluate(name):
    """Run evaluate.py on the finished checkpoint and append a row to the shared CSV."""
    cmd = [
        sys.executable, "evaluate.py",
        "--name",       name,
        "--dataroot",   DATASET_DIR,
        "--netG",       "resnet_9blocks",
        "--ngf",        "64",
        "--output_csv", RESULTS_CSV,
    ]
    print(f"  Evaluating {name}...")
    result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: evaluate.py failed for {name}:\n{result.stderr[-800:]}")
    else:
        # Print just the metrics lines (skip verbose model-load output)
        for line in result.stdout.splitlines():
            if any(k in line for k in ("FID", "KID", "LPIPS", "SSIM", "Appended", "Results")):
                print(f"  {line.strip()}")


def train(name, lambda_perceptual, perceptual_layers, no_cycle_l1, lr):
    args = [
        "--dataroot",          DATASET_DIR,
        "--name",              name,
        "--model",             "cycle_gan",
        "--dataset_mode",      "unaligned",
        "--netG",              "resnet_9blocks",
        "--netD",              "basic",
        "--ngf",               "64",
        "--ndf",               "64",
        "--norm",              "instance",
        "--no_dropout",
        "--gan_mode",          "lsgan",
        "--lr",                lr,
        "--beta1",             "0.5",
        "--pool_size",         "50",
        "--n_epochs",          "50",
        "--n_epochs_decay",    "50",
        "--batch_size",        "1",
        "--load_size",         "286",
        "--crop_size",         "256",
        "--lambda_A",          "10.0",
        "--lambda_B",          "10.0",
        "--lambda_identity",   "0.5",
        "--lambda_perceptual", str(lambda_perceptual),
        "--perceptual_layers", perceptual_layers,
        "--save_epoch_freq",   "5",
        "--seed",              str(SEED),
        "--no_html",
    ]
    if no_cycle_l1:
        args.append("--no_cycle_l1")

    print(
        f"\n{'='*70}\n"
        f"Training : {name}\n"
        f"  lambda_p={lambda_perceptual}, layers={perceptual_layers}, "
        f"replace_l1={no_cycle_l1}, lr={lr}, seed={SEED}\n"
        f"{'='*70}"
    )
    train(args)
    cleanup_checkpoints(name)
    evaluate(name)


if __name__ == "__main__":
    for name, lam, layers, replace, lr in CONFIGS:
        train(name, lam, layers, replace, lr)
