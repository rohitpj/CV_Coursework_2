import sys
import re
import csv
import time
import subprocess
from pathlib import Path

from train import train

_REPO = Path(__file__).resolve().parent
DATASET_DIR     = str(_REPO / "datasets" / "apple2orange")
CHECKPOINTS_DIR = str(_REPO / "checkpoints")
RESULTS_CSV     = str(_REPO / "evaluation" / "et2_results.csv")
TIMING_CSV      = str(_REPO / "evaluation" / "et2_timing.csv")

# Same seed as ET1 for consistency across all experiments; report this value.
SEED = 42

# Coarse epoch checkpoints kept for convergence analysis ("does U-Net converge
# faster than ResNet?"). Dense intermediate .pth files are removed to save disk.
# Loss logs and opt.txt are never touched — required submission artefacts.
KEEP_EPOCHS = {20, 40, 60, 80, 100}

# (name, netG, ngf, netD, n_layers_D, lr)
#
# gan_mode=lsgan throughout — matches Task 2 baseline and ET1; not a variable here.
# et2_resnet9 is configured identically to Task 2 defaults so it can serve as
# the clean "original setup" the spec asks about. Document this in the report.
CONFIGS = [
    # ── Baselines ──────────────────────────────────────────────────────────────
    # ResNet-9: matches Task 2 exactly; U-Net-256: generator architecture change.
    # Both use the same discriminator/lr/ngf so the generator is the only variable.
    ("et2_resnet9",      "resnet_9blocks", 64,  "basic",    3, "0.0002"),
    ("et2_unet256",      "unet_256",       64,  "basic",    3, "0.0002"),

    # ── Generator depth axis ───────────────────────────────────────────────────
    # resnet_6blocks: fewer residual blocks = less capacity; compare FID and
    # convergence speed against resnet_9blocks (capacity vs overfitting trade-off).
    ("et2_resnet6",      "resnet_6blocks", 64,  "basic",    3, "0.0002"),

    # ── Discriminator patch-size axis ──────────────────────────────────────────
    # pixel (1×1): per-pixel feedback, very local.
    # n_layers 5: larger receptive field than basic (70×70) — discriminator sees
    # more context per decision; watch for D overpowering G and causing collapse.
    ("et2_disc_pixel",   "resnet_9blocks", 64,  "pixel",    3, "0.0002"),
    ("et2_disc_large",   "resnet_9blocks", 64,  "n_layers", 5, "0.0002"),

    # ── Generator width axis ───────────────────────────────────────────────────
    # ngf controls channel width throughout the generator (capacity vs cost).
    # Report parameter counts (printed by model.setup) and training time alongside FID.
    ("et2_ngf32",        "resnet_9blocks", 32,  "basic",    3, "0.0002"),
    ("et2_ngf128",       "resnet_9blocks", 128, "basic",    3, "0.0002"),

    # ── LR cross: {resnet9, unet256} × {0.0002, 0.0001} ───────────────────────
    # Without the resnet9_lr1 control, you cannot tell whether 0.0001 helps
    # U-Net *specifically* (skip-connection gradient flow) or any architecture.
    # Both runs needed to make the claim defensible in the report.
    ("et2_resnet9_lr1",  "resnet_9blocks", 64,  "basic",    3, "0.0001"),
    ("et2_unet256_lr1",  "unet_256",       64,  "basic",    3, "0.0001"),
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


def log_timing(name, netG, ngf, netD, n_layers_D, lr, elapsed_sec):
    """Append training-time row to TIMING_CSV for the report's cost comparison."""
    csv_path = Path(TIMING_CSV)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["name", "netG", "ngf", "netD", "n_layers_D", "lr",
                              "seed", "elapsed_min"])
        writer.writerow([name, netG, ngf, netD, n_layers_D, lr,
                         SEED, f"{elapsed_sec / 60:.1f}"])
    print(f"  Training time: {elapsed_sec / 60:.1f} min — logged to {csv_path}")


def evaluate(name, netG, ngf):
    """Run evaluate.py on the finished checkpoint; appends a row to RESULTS_CSV."""
    cmd = [
        sys.executable, "evaluate.py",
        "--name",       name,
        "--dataroot",   DATASET_DIR,
        "--netG",       netG,
        "--ngf",        str(ngf),
        "--output_csv", RESULTS_CSV,
    ]
    print(f"  Evaluating {name}...")
    result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: evaluate.py failed for {name}:\n{result.stderr[-800:]}")
    else:
        for line in result.stdout.splitlines():
            if any(k in line for k in ("FID", "KID", "LPIPS", "SSIM", "Appended", "Results")):
                print(f"  {line.strip()}")


def train(name, netG, ngf, netD, n_layers_D, lr):
    args = [
        "--dataroot",        DATASET_DIR,
        "--name",            name,
        "--model",           "cycle_gan",
        "--dataset_mode",    "unaligned",
        "--netG",            netG,
        "--netD",            netD,
        "--n_layers_D",      str(n_layers_D),
        "--ngf",             str(ngf),
        "--ndf",             "64",
        "--norm",            "instance",
        "--no_dropout",
        "--gan_mode",        "lsgan",
        "--lr",              lr,
        "--beta1",           "0.5",
        "--pool_size",       "50",
        "--n_epochs",        "50",
        "--n_epochs_decay",  "50",
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
        f"  netG={netG}, ngf={ngf}, netD={netD}, n_layers_D={n_layers_D}, "
        f"lr={lr}, seed={SEED}\n"
        f"{'='*70}"
    )
    t0 = time.time()
    train(args)
    elapsed = time.time() - t0

    cleanup_checkpoints(name)
    log_timing(name, netG, ngf, netD, n_layers_D, lr, elapsed)
    evaluate(name, netG, ngf)


if __name__ == "__main__":
    for name, netG, ngf, netD, n_layers_D, lr in CONFIGS:
        train(name, netG, ngf, netD, n_layers_D, lr)
