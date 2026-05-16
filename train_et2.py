import csv
import time
from pathlib import Path

from train import train as run_training
from evaluate import run_evaluation

_REPO = Path(__file__).resolve().parent
DATASET_DIR     = str(_REPO / "datasets" / "apple2orange")
CHECKPOINTS_DIR = str(_REPO / "checkpoints")
RESULTS_CSV     = str(_REPO / "evaluation" / "et2_results.csv")
TIMING_CSV      = str(_REPO / "evaluation" / "et2_timing.csv")

SEED = 42

# (name, netG, ngf, netD, n_layers_D, lr)
CONFIGS = [
    ("et2_resnet9",      "resnet_9blocks", 64,  "basic",    3, "0.0002"),
    ("et2_unet256",      "unet_256",       64,  "basic",    3, "0.0002"),
    ("et2_resnet6",      "resnet_6blocks", 64,  "basic",    3, "0.0002"),
    ("et2_disc_pixel",   "resnet_9blocks", 64,  "pixel",    3, "0.0002"),
    ("et2_disc_large",   "resnet_9blocks", 64,  "n_layers", 5, "0.0002"),
    ("et2_ngf32",        "resnet_9blocks", 32,  "basic",    3, "0.0002"),
    ("et2_ngf128",       "resnet_9blocks", 128, "basic",    3, "0.0002"),
    ("et2_resnet9_lr1",  "resnet_9blocks", 64,  "basic",    3, "0.0001"),
    ("et2_unet256_lr1",  "unet_256",       64,  "basic",    3, "0.0001"),
]


def log_timing(name, netG, ngf, netD, n_layers_D, lr, elapsed_sec):
    csv_path = Path(TIMING_CSV)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["name", "netG", "ngf", "netD", "n_layers_D", "lr", "seed", "elapsed_min"])
        writer.writerow([name, netG, ngf, netD, n_layers_D, lr, SEED, f"{elapsed_sec / 60:.1f}"])
    print(f"  Training time: {elapsed_sec / 60:.1f} min")


def evaluate(name, netG, ngf):
    try:
        run_evaluation(name, "latest", netG, ngf, "instance", DATASET_DIR, output_csv=RESULTS_CSV)
    except Exception as e:
        print(f"WARNING: evaluation failed for {name}: {e}")


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
        "--save_epoch_freq", "20",
        "--seed",            str(SEED),
        "--no_html",
    ]

    print(f"\nTraining: {name} | netG={netG}, ngf={ngf}, netD={netD}, n_layers_D={n_layers_D}, lr={lr}")
    t0 = time.time()
    run_training(args)
    elapsed = time.time() - t0

    log_timing(name, netG, ngf, netD, n_layers_D, lr, elapsed)
    evaluate(name, netG, ngf)


if __name__ == "__main__":
    for name, netG, ngf, netD, n_layers_D, lr in CONFIGS:
        train(name, netG, ngf, netD, n_layers_D, lr)
