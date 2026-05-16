from pathlib import Path

from train import train as run_training
from evaluate import run_evaluation

_REPO = Path(__file__).resolve().parent
DATASET_DIR     = str(_REPO / "datasets" / "apple2orange")
CHECKPOINTS_DIR = str(_REPO / "checkpoints")
RESULTS_CSV     = str(_REPO / "evaluation" / "et4_results.csv")

SEED = 42

FULL    = 10000
HALF    = 497
QUARTER = 248

# (name, max_dataset_size, n_epochs, n_epochs_decay, lr)
CONFIGS = [
    ("et4_full",              FULL,    50,  50,  "0.0002"),
    ("et4_50pct",             HALF,    50,  50,  "0.0002"),
    ("et4_25pct",             QUARTER, 50,  50,  "0.0002"),
    ("et4_50pct_more_epochs", HALF,    100, 100, "0.0002"),
    ("et4_25pct_more_epochs", QUARTER, 100, 100, "0.0002"),
]


def evaluate(name):
    try:
        run_evaluation(name, "latest", "resnet_9blocks", 64, "instance", DATASET_DIR, output_csv=RESULTS_CSV)
    except Exception as e:
        print(f"WARNING: evaluation failed for {name}: {e}")


def train(name, max_dataset_size, n_epochs, n_epochs_decay, lr):
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
        "--n_epochs",          str(n_epochs),
        "--n_epochs_decay",    str(n_epochs_decay),
        "--batch_size",        "1",
        "--load_size",         "286",
        "--crop_size",         "256",
        "--lambda_A",          "10.0",
        "--lambda_B",          "10.0",
        "--lambda_identity",   "0.5",
        "--max_dataset_size",  str(max_dataset_size),
        "--save_epoch_freq",   "20",
        "--seed",              str(SEED),
        "--no_html",
    ]

    print(f"\nTraining: {name} | max_dataset_size={max_dataset_size}, epochs={n_epochs}+{n_epochs_decay}")
    run_training(args)
    evaluate(name)


if __name__ == "__main__":
    for name, max_size, n_ep, n_dec, lr in CONFIGS:
        train(name, max_size, n_ep, n_dec, lr)
