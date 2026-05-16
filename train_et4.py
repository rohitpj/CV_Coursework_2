import sys
import re
import shutil
import subprocess
from pathlib import Path

DATASET_DIR = "./datasets/apple2orange"
CHECKPOINTS_DIR = "./checkpoints"

# apple2orange has ~995 trainA and ~1019 trainB images
# max_dataset_size limits how many are loaded from each directory
FULL = 10000   # larger than the dataset so all images are used
HALF = 497     # ~50% of 995
QUARTER = 248  # ~25% of 995

# (name, max_dataset_size, n_epochs, n_epochs_decay, lr)
#
# The improvement strategy for limited data is to train for longer.
# With fewer images per epoch, the model sees fewer gradient updates, so
# doubling the epochs compensates by giving it the same total number of
# update steps as the full-data run.
CONFIGS = [
    # baseline -- full dataset
    ("et4_full",              FULL,    50,  50,  "0.0002"),

    # reduced data -- same training budget
    ("et4_50pct",             HALF,    50,  50,  "0.0002"),
    ("et4_25pct",             QUARTER, 50,  50,  "0.0002"),

    # reduced data + improvement: train longer to compensate
    ("et4_50pct_more_epochs", HALF,    100, 100, "0.0002"),
    ("et4_25pct_more_epochs", QUARTER, 100, 100, "0.0002"),
]


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


def train(name, max_dataset_size, n_epochs, n_epochs_decay, lr):
    cmd = [
        sys.executable, "train.py",
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
        "--n_epochs",          str(n_epochs),
        "--n_epochs_decay",    str(n_epochs_decay),
        "--batch_size",        "1",
        "--load_size",         "286",
        "--crop_size",         "256",
        "--lambda_A",          "10.0",
        "--lambda_B",          "10.0",
        "--lambda_identity",   "0.5",
        "--max_dataset_size",  str(max_dataset_size),
        "--save_epoch_freq",   "5",
        "--no_html",
    ]

    pct = "100%" if max_dataset_size == FULL else ("50%" if max_dataset_size == HALF else "25%")
    print(f"\nTraining: {name}  (data={pct}, epochs={n_epochs}+{n_epochs_decay}, lr={lr})")
    subprocess.run(cmd, cwd=Path(__file__).parent)
    save_best_model(name)


if __name__ == "__main__":
    for name, max_size, n_ep, n_dec, lr in CONFIGS:
        train(name, max_size, n_ep, n_dec, lr)
