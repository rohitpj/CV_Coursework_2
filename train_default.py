import zipfile
import requests
from pathlib import Path

from train import train as run_training
from evaluate import run_evaluation

_REPO = Path(__file__).resolve().parent
DATASET_URL     = "http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/apple2orange.zip"
DATASET_DIR     = str(_REPO / "datasets" / "apple2orange")
CHECKPOINTS_DIR = str(_REPO / "checkpoints")
RESULTS_CSV     = str(_REPO / "evaluation" / "task2_results.csv")

SEED           = 42
N_EPOCHS       = 50
N_EPOCHS_DECAY = 50

# (name, lr)
CONFIGS = [
    ("task2_lr0001",  "0.0001"),
    ("task2_default", "0.0002"),
    ("task2_lr0004",  "0.0004"),
]


def download_dataset():
    if Path(DATASET_DIR).exists():
        print("Dataset already exists, skipping download.")
        return

    zip_path = _REPO / "datasets" / "apple2orange.zip"
    (_REPO / "datasets").mkdir(exist_ok=True)
    print(f"Downloading dataset from {DATASET_URL} ...")
    try:
        r = requests.get(DATASET_URL, stream=True, timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Download failed: {e}\n"
            f"Place the dataset manually at {DATASET_DIR} (trainA/, trainB/, testA/, testB/)."
        ) from e

    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

    print("Verifying and extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            zip_path.unlink()
            raise RuntimeError(f"Downloaded zip is corrupt (first bad file: {bad}). Delete datasets/ and re-run.")
        zf.extractall(_REPO / "datasets")
    zip_path.unlink()
    print("Dataset ready.")


def evaluate(name):
    try:
        run_evaluation(name, "latest", "resnet_9blocks", 64, "instance", DATASET_DIR, output_csv=RESULTS_CSV)
    except Exception as e:
        print(f"WARNING: evaluation failed for {name}: {e}")


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
        "--save_epoch_freq", "20",
        "--seed",            str(SEED),
        "--no_html",
    ]

    print(f"\nTraining: {name} | lr={lr}, epochs={N_EPOCHS}+{N_EPOCHS_DECAY}")
    run_training(args)
    evaluate(name)


if __name__ == "__main__":
    download_dataset()
    for name, lr in CONFIGS:
        train(name, lr)
