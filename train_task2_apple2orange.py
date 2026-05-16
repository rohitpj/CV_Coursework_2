import sys
import subprocess
import zipfile
import requests
from pathlib import Path

DATASET_URL = "http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/apple2orange.zip"
DATASET_DIR = "./datasets/apple2orange"


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


if __name__ == "__main__":
    download_dataset()

    cmd = [
        sys.executable, "train.py",
        "--dataroot",        DATASET_DIR,
        "--name",            "apple2orange_cyclegan_default",
        "--model",           "cycle_gan",
        "--dataset_mode",    "unaligned",
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

    print("Starting training...")
    subprocess.run(cmd)
