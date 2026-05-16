from pathlib import Path

from train import train as run_training
from evaluate import run_evaluation

_REPO = Path(__file__).resolve().parent
DATASET_DIR     = str(_REPO / "datasets" / "apple2orange")
CHECKPOINTS_DIR = str(_REPO / "checkpoints")
RESULTS_CSV     = str(_REPO / "evaluation" / "et1_results.csv")

SEED = 42

# (name, lambda_perceptual, perceptual_layers, no_cycle_l1, lr)
CONFIGS = [
    ("et1_baseline",       0.0,  "all",     False, "0.0002"),
    ("et1_lam0p1",         0.1,  "all",     False, "0.0002"),
    ("et1_lam1",           1.0,  "all",     False, "0.0002"),
    ("et1_lam10",         10.0,  "all",     False, "0.0002"),
    ("et1_shallow",        1.0,  "shallow", False, "0.0002"),
    ("et1_deep",           1.0,  "deep",    False, "0.0002"),
    ("et1_replace_lam10", 10.0,  "all",     True,  "0.0002"),
]


def evaluate(name):
    try:
        run_evaluation(name, "latest", "resnet_9blocks", 64, "instance", DATASET_DIR, output_csv=RESULTS_CSV)
    except Exception as e:
        print(f"WARNING: evaluation failed for {name}: {e}")


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
        "--save_epoch_freq",   "20",
        "--seed",              str(SEED),
        "--no_html",
    ]
    if no_cycle_l1:
        args.append("--no_cycle_l1")

    print(f"\nTraining: {name} | lambda_p={lambda_perceptual}, layers={perceptual_layers}, replace_l1={no_cycle_l1}")
    run_training(args)
    evaluate(name)


if __name__ == "__main__":
    for name, lam, layers, replace, lr in CONFIGS:
        train(name, lam, layers, replace, lr)
