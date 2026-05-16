import sys
import subprocess
from pathlib import Path

DATASET_DIR = "./datasets/apple2orange"

# (name, netG, lr)
CONFIGS = [
    ("et2_resnet9",     "resnet_9blocks", "0.0002"),
    ("et2_unet256",     "unet_256",       "0.0002"),

    # lr tuning for U-Net -- skip connections change gradient flow so a
    # different lr may be needed compared to the ResNet baseline
    ("et2_unet256_lr1", "unet_256",       "0.0001"),
]


def train(name, netG, lr):
    cmd = [
        sys.executable, "train.py",
        "--dataroot",        DATASET_DIR,
        "--name",            name,
        "--model",           "cycle_gan",
        "--dataset_mode",    "unaligned",
        "--netG",            netG,
        "--netD",            "basic",
        "--ngf",             "64",
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
        "--no_html",
    ]

    print(f"\nTraining: {name}  (netG={netG}, lr={lr})")
    subprocess.run(cmd, cwd=Path(__file__).parent)


if __name__ == "__main__":
    for name, netG, lr in CONFIGS:
        train(name, netG, lr)
