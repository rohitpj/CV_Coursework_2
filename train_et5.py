import csv
from pathlib import Path

from train import train as run_training
from evaluate import run_evaluation

_REPO = Path(__file__).resolve().parent
DATASET_DIR     = str(_REPO / "datasets" / "apple2orange")
CHECKPOINTS_DIR = str(_REPO / "checkpoints")
RESULTS_CSV     = str(_REPO / "evaluation" / "et5_results.csv")

SEED        = 42
KEEP_EPOCHS = {20, 40, 60, 80, 100}

FULL    = 10000
QUARTER = 248

# (name, lambda_perceptual, perceptual_layers, no_cycle_l1, max_dataset_size)
SETUP1_NEW = [
    ("et5_perc_25pct",      1.0, "all", False, QUARTER),
    ("et5_perc_25pct_lam3", 3.0, "all", False, QUARTER),
]

# (name, netG, ngf, max_dataset_size)
SETUP2_NEW = [
    ("et5_resnet9_25pct",  "resnet_9blocks", 64, QUARTER),
    ("et5_unet256_25pct",  "unet_256",       64, QUARTER),
]

# Existing checkpoints re-evaluated so all cells use the same evaluate.py version
SETUP1_REEVAL = [
    ("et1_baseline", "resnet_9blocks", 64),
    ("et1_lam1",     "resnet_9blocks", 64),
    ("et4_25pct",    "resnet_9blocks", 64),
]

SETUP2_REEVAL = [
    ("et2_resnet9",  "resnet_9blocks", 64),
    ("et2_unet256",  "unet_256",       64),
]


def cleanup_checkpoints(name):
    checkpoint_dir = Path(CHECKPOINTS_DIR) / name
    if not checkpoint_dir.exists():
        return
    for f in checkpoint_dir.glob("*_net_*.pth"):
        epoch_str = f.name.split("_")[0]
        if epoch_str.isdigit() and int(epoch_str) not in KEEP_EPOCHS:
            f.unlink()


def evaluate(name, netG, ngf):
    checkpoint_dir = Path(CHECKPOINTS_DIR) / name
    if not (checkpoint_dir / "latest_net_G_A.pth").exists():
        print(f"SKIP {name}: checkpoint not found.")
        return
    try:
        run_evaluation(name, "latest", netG, ngf, "instance", DATASET_DIR, output_csv=RESULTS_CSV)
    except Exception as e:
        print(f"WARNING: evaluation failed for {name}: {e}")


def train_setup1(name, lambda_perceptual, perceptual_layers, no_cycle_l1, max_dataset_size):
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
        "--lr",                "0.0002",
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
        "--max_dataset_size",  str(max_dataset_size),
        "--save_epoch_freq",   "5",
        "--seed",              str(SEED),
        "--no_html",
    ]
    if no_cycle_l1:
        args.append("--no_cycle_l1")

    print(f"\nSetup 1 — {name} | lambda_p={lambda_perceptual}, layers={perceptual_layers}, max_data={max_dataset_size}")
    run_training(args)
    cleanup_checkpoints(name)
    evaluate(name, "resnet_9blocks", 64)


def train_setup2(name, netG, ngf, max_dataset_size):
    args = [
        "--dataroot",         DATASET_DIR,
        "--name",             name,
        "--model",            "cycle_gan",
        "--dataset_mode",     "unaligned",
        "--netG",             netG,
        "--netD",             "basic",
        "--ngf",              str(ngf),
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
        "--max_dataset_size", str(max_dataset_size),
        "--save_epoch_freq",  "5",
        "--seed",             str(SEED),
        "--no_html",
    ]

    print(f"\nSetup 2 — {name} | netG={netG}, ngf={ngf}, max_data={max_dataset_size}")
    run_training(args)
    cleanup_checkpoints(name)
    evaluate(name, netG, ngf)


def print_summary():
    if not Path(RESULTS_CSV).exists():
        print("No results CSV found — run training + evaluation first.")
        return

    rows = {}
    with open(RESULTS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["name"]] = row

    def fval(name, col):
        v = rows.get(name, {}).get(col, "")
        try:
            return float(v)
        except ValueError:
            return None

    def fmt(v, spec):
        return format(v, spec) if v is not None else "—"

    print("\nSetup 1: Perceptual loss under data scarcity (FID A->B, lower is better)\n")
    s1_rows = [
        ("Baseline",         "et1_baseline", "et4_25pct"),
        ("Perceptual λ=1.0", "et1_lam1",     "et5_perc_25pct"),
        ("Perceptual λ=3.0", "—",            "et5_perc_25pct_lam3"),
    ]
    print(f"{'Config':<22}  {'Full FID':>10}  {'25% FID':>10}  {'Delta':>8}")
    print("-" * 56)
    for label, full_name, small_name in s1_rows:
        f_full  = fval(full_name, "fid_ab")
        f_small = fval(small_name, "fid_ab")
        delta   = f"{f_small - f_full:+.2f}" if (f_full is not None and f_small is not None) else "—"
        print(f"  {label:<20}  {fmt(f_full, '.2f'):>10}  {fmt(f_small, '.2f'):>10}  {delta:>8}")

    print("\nSetup 2: Architecture ranking stability\n")
    s2_rows = [
        ("ResNet-9",  "et2_resnet9",  "et5_resnet9_25pct"),
        ("U-Net-256", "et2_unet256",  "et5_unet256_25pct"),
    ]
    print(f"{'Backbone':<14}  {'Full FID':>10}  {'Full SSIM':>10}  {'25% FID':>10}  {'25% SSIM':>10}")
    print("-" * 60)
    for label, full_name, small_name in s2_rows:
        print(f"  {label:<12}  "
              f"{fmt(fval(full_name, 'fid_ab'), '.2f'):>10}  "
              f"{fmt(fval(full_name, 'ssim_cyc_a'), '.4f'):>10}  "
              f"{fmt(fval(small_name, 'fid_ab'), '.2f'):>10}  "
              f"{fmt(fval(small_name, 'ssim_cyc_a'), '.4f'):>10}")


if __name__ == "__main__":
    print("\nSetup 1 — new training: perceptual loss at 25% data")
    for cfg in SETUP1_NEW:
        train_setup1(*cfg)

    print("\nSetup 2 — new training: backbones at 25% data")
    for cfg in SETUP2_NEW:
        train_setup2(*cfg)

    print("\nRe-evaluating existing checkpoints")
    for name, netG, ngf in SETUP1_REEVAL + SETUP2_REEVAL:
        evaluate(name, netG, ngf)

    print_summary()
