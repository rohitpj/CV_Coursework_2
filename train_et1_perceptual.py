import sys
import subprocess
from pathlib import Path

DATASET_DIR = "./datasets/apple2orange"

# (name, lambda_perceptual, perceptual_layers, no_cycle_l1)
CONFIGS = [
    ("et1_lam0p1",  0.1,  "all",     False),
    ("et1_lam1",    1.0,  "all",     False),
    ("et1_lam10",  10.0,  "all",     False),
    ("et1_shallow", 1.0,  "shallow", False),
    ("et1_deep",    1.0,  "deep",    False),
    ("et1_replace", 1.0,  "all",     True),
]


def train(name, lambda_perceptual, perceptual_layers, no_cycle_l1):
    cmd = [
        sys.executable, "train.py",
        "--dataroot",         DATASET_DIR,
        "--name",             name,
        "--model",            "cycle_gan",
        "--dataset_mode",     "unaligned",
        "--netG",             "resnet_9blocks",
        "--netD",             "basic",
        "--ngf",              "64",
        "--ndf",              "64",
        "--norm",             "instance",
        "--no_dropout",
        "--gan_mode",         "lsgan",
        "--lr",               "0.0002",
        "--beta1",            "0.5",
        "--pool_size",        "50",
        "--n_epochs",         "50",
        "--n_epochs_decay",   "50",
        "--batch_size",       "1",
        "--load_size",        "286",
        "--crop_size",        "256",
        "--lambda_A",         "10.0",
        "--lambda_B",         "10.0",
        "--lambda_identity",  "0.5",
        "--lambda_perceptual", str(lambda_perceptual),
        "--perceptual_layers", perceptual_layers,
        "--save_epoch_freq",  "5",
        "--no_html",
    ]
    if no_cycle_l1:
        cmd.append("--no_cycle_l1")

    print(f"\nTraining: {name}  (lambda_p={lambda_perceptual}, layers={perceptual_layers}, replace_l1={no_cycle_l1})")
    subprocess.run(cmd, cwd=Path(__file__).parent)


if __name__ == "__main__":
    for name, lam, layers, replace in CONFIGS:
        train(name, lam, layers, replace)
