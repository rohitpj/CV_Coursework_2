"""
ET5 — Combining previous ETs as robustness checks.

Setup 1  (ET1 × ET4 × ET3): Does the perceptual-loss benefit survive data scarcity?
  Grid: {baseline, perceptual} × {full, 25%}
  Existing cells reused:
    et1_baseline  — full data, no perceptual loss
    et1_lam1      — full data, λ_perc=1.0
    et4_25pct     — 25% data,  no perceptual loss
  New cells (this script trains):
    et5_perc_25pct       — 25% data, λ_perc=1.0 (matches best ET1 value)
    et5_perc_25pct_lam3  — 25% data, λ_perc=3.0 (prior should strengthen under
                           data scarcity; test whether optimal λ shifts upward)

Setup 2  (ET2 × ET4 × ET3): Is the ET2 architecture ranking stable across data regimes?
  Grid: {ResNet-9, U-Net-256} × {full, 25%}
  Existing cells reused:
    et2_resnet9   — full data, ResNet-9
    et2_unet256   — full data, U-Net-256
  New cells (this script trains):
    et5_resnet9_25pct  — 25% data, ResNet-9
    et5_unet256_25pct  — 25% data, U-Net-256

Total new training: 4 runs.
All four use the same epoch schedule (50 + 50) as the matching full-data runs so
the data-scarcity effect is isolated and not confounded with training budget.

Seed is fixed at 42 across all runs (matching ET1/ET2/ET4).
"""

import sys
import re
import csv
import subprocess
from pathlib import Path

from train import train

_REPO = Path(__file__).resolve().parent
DATASET_DIR     = str(_REPO / "datasets" / "apple2orange")
CHECKPOINTS_DIR = str(_REPO / "checkpoints")
RESULTS_CSV     = str(_REPO / "evaluation" / "et5_results.csv")

SEED        = 42
KEEP_EPOCHS = {20, 40, 60, 80, 100}

FULL    = 10000   # larger than dataset; loads all images
QUARTER = 248     # ~25% of 995 trainA images (same constant as ET4)

# ── Setup 1: new training cells ──────────────────────────────────────────────
# (name, lambda_perceptual, perceptual_layers, no_cycle_l1, max_dataset_size)
SETUP1_NEW = [
    ("et5_perc_25pct",      1.0, "all", False, QUARTER),
    # If the optimal λ on full data was 1.0, a scarce-data prior may need more
    # weight.  Update the higher value here once ET1 results are in if needed.
    ("et5_perc_25pct_lam3", 3.0, "all", False, QUARTER),
]

# ── Setup 2: new training cells ──────────────────────────────────────────────
# (name, netG, ngf, max_dataset_size)
SETUP2_NEW = [
    ("et5_resnet9_25pct",  "resnet_9blocks", 64, QUARTER),
    ("et5_unet256_25pct",  "unet_256",       64, QUARTER),
]

# ── Existing checkpoints to re-evaluate with the updated metrics pipeline ────
# evaluate.py now computes KID and translation-strength LPIPS that were absent
# when ET1/ET2/ET4 originally ran.  Re-running is necessary so all cells in the
# comparison table were produced by the same version of evaluate.py.
# NOTE: if an existing checkpoint doesn't exist yet (ET1/ET2 not yet trained),
# evaluate() will print a WARNING and skip gracefully — run ET1/ET2 first.
SETUP1_REEVAL = [
    ("et1_baseline", "resnet_9blocks", 64),   # full data, baseline loss
    ("et1_lam1",     "resnet_9blocks", 64),   # full data, λ_perc=1.0
    ("et4_25pct",    "resnet_9blocks", 64),   # 25% data,  baseline loss
]

SETUP2_REEVAL = [
    ("et2_resnet9",  "resnet_9blocks", 64),   # full data, ResNet-9
    ("et2_unet256",  "unet_256",       64),   # full data, U-Net-256
]


# ─────────────────────────────────────────────────────────────────────────────

def cleanup_checkpoints(name):
    checkpoint_dir = Path(CHECKPOINTS_DIR) / name
    if not checkpoint_dir.exists():
        return
    removed = 0
    for f in checkpoint_dir.glob("*_net_*.pth"):
        m = re.match(r"^(\d+)_net_", f.name)
        if m and int(m.group(1)) not in KEEP_EPOCHS:
            f.unlink()
            removed += 1
    print(f"  Kept epoch checkpoints {sorted(KEEP_EPOCHS)}, removed {removed} intermediate .pth files.")


def evaluate(name, netG, ngf):
    """Call evaluate.py and append a row to RESULTS_CSV."""
    checkpoint_dir = Path(CHECKPOINTS_DIR) / name
    if not (checkpoint_dir / "latest_net_G_A.pth").exists():
        print(f"  SKIP {name}: checkpoint not found (train it first).")
        return

    cmd = [
        sys.executable, "evaluate.py",
        "--name",       name,
        "--dataroot",   DATASET_DIR,
        "--netG",       netG,
        "--ngf",        str(ngf),
        "--output_csv", RESULTS_CSV,
    ]
    print(f"  Evaluating {name} (netG={netG}, ngf={ngf})...")
    result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: evaluate.py failed for {name}:\n{result.stderr[-800:]}")
    else:
        for line in result.stdout.splitlines():
            if any(k in line for k in ("FID", "KID", "LPIPS", "SSIM", "Appended", "Results")):
                print(f"  {line.strip()}")


def train_setup1(name, lambda_perceptual, perceptual_layers, no_cycle_l1, max_dataset_size):
    """Train a Setup 1 cell: perceptual-loss variant at reduced data."""
    pct = "25%" if max_dataset_size == QUARTER else "100%"
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

    print(
        f"\n{'='*70}\n"
        f"Setup 1 — {name}\n"
        f"  data={pct}, λ_perc={lambda_perceptual}, layers={perceptual_layers}, "
        f"replace_l1={no_cycle_l1}, seed={SEED}\n"
        f"{'='*70}"
    )
    train(args)
    cleanup_checkpoints(name)
    evaluate(name, "resnet_9blocks", 64)


def train_setup2(name, netG, ngf, max_dataset_size):
    """Train a Setup 2 cell: backbone variant at reduced data."""
    pct = "25%" if max_dataset_size == QUARTER else "100%"
    args = [
        "--dataroot",        DATASET_DIR,
        "--name",            name,
        "--model",           "cycle_gan",
        "--dataset_mode",    "unaligned",
        "--netG",            netG,
        "--netD",            "basic",
        "--ngf",             str(ngf),
        "--ndf",             "64",
        "--norm",            "instance",
        "--no_dropout",
        "--gan_mode",        "lsgan",
        "--lr",              "0.0002",
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
        "--max_dataset_size", str(max_dataset_size),
        "--save_epoch_freq", "5",
        "--seed",            str(SEED),
        "--no_html",
    ]

    print(
        f"\n{'='*70}\n"
        f"Setup 2 — {name}\n"
        f"  data={pct}, netG={netG}, ngf={ngf}, seed={SEED}\n"
        f"{'='*70}"
    )
    train(args)
    cleanup_checkpoints(name)
    evaluate(name, netG, ngf)


def print_summary():
    """Read RESULTS_CSV and print formatted interaction tables for the report.

    Setup 1 table — interaction: does perceptual loss help more at 25% data?
    Setup 2 table — does the architecture ranking hold under data scarcity,
                    and do FID and SSIM agree on which backbone is better?
    """
    if not Path(RESULTS_CSV).exists():
        print("  No results CSV found — run training + evaluation first.")
        return

    rows = {}
    with open(RESULTS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["name"]] = row   # last row wins if a name appears twice

    def get(name, col):
        if name not in rows:
            return "—"
        v = rows[name].get(col, "—")
        return v if v else "—"

    def fval(name, col):
        s = get(name, col)
        try:
            return float(s)
        except ValueError:
            return None

    # ── Setup 1 ──────────────────────────────────────────────────────────────
    # Key interaction metric: FID(A->B).  Lower = better translation quality.
    # Does (FID_perc - FID_baseline) change sign or magnitude as data shrinks?
    print("\n" + "="*70)
    print("ET5 SUMMARY — Setup 1: Perceptual loss under data scarcity")
    print("(Primary metric: FID A->B — lower is better)\n")

    s1_rows = [
        ("Baseline (no perc)",  "et1_baseline",        "et4_25pct"),
        ("Perceptual λ=1.0",    "et1_lam1",            "et5_perc_25pct"),
        ("Perceptual λ=3.0",    "—",                   "et5_perc_25pct_lam3"),
    ]
    header = f"{'Config':<24}  {'Full data (FID↓)':>18}  {'25% data (FID↓)':>18}  {'Δ(25%-full)':>12}"
    print(header)
    print("-" * len(header))
    for label, full_name, small_name in s1_rows:
        f_full  = fval(full_name, "fid_ab")
        f_small = fval(small_name, "fid_ab")
        s_full  = f"{f_full:.2f}" if f_full  is not None else "—"
        s_small = f"{f_small:.2f}" if f_small is not None else "—"
        delta   = f"{f_small - f_full:+.2f}" if (f_full is not None and f_small is not None) else "—"
        print(f"  {label:<22}  {s_full:>18}  {s_small:>18}  {delta:>12}")

    print(
        "\nInterpretation hint: if Δ is smaller for perceptual rows than baseline,\n"
        "the VGG prior cushions the drop — a key ET5 finding.\n"
        "Also check KID and translation LPIPS for the same interaction.\n"
    )

    # ── Setup 2 ──────────────────────────────────────────────────────────────
    # Two consistency checks:
    #   (a) Metric consistency: does FID and SSIM agree on the better backbone?
    #   (b) Data consistency:   does the full-data winner still win at 25%?
    print("="*70)
    print("ET5 SUMMARY — Setup 2: Architecture ranking stability")
    print("(FID↓ = translation quality; SSIM_cyc↑ = cycle fidelity)\n")

    s2_rows = [
        ("ResNet-9",   "et2_resnet9",  "et5_resnet9_25pct"),
        ("U-Net-256",  "et2_unet256",  "et5_unet256_25pct"),
    ]
    col_w = 10
    print(f"{'Backbone':<14}  "
          f"{'Full FID↓':>{col_w}}  {'Full SSIM↑':>{col_w}}  "
          f"{'25% FID↓':>{col_w}}  {'25% SSIM↑':>{col_w}}")
    print("-" * 64)
    for label, full_name, small_name in s2_rows:
        ff  = fval(full_name,  "fid_ab");    sf  = f"{ff:.2f}"  if ff  is not None else "—"
        fs  = fval(full_name,  "ssim_cyc_a");ss_ = f"{fs:.4f}" if fs  is not None else "—"
        qf  = fval(small_name, "fid_ab");    qsf = f"{qf:.2f}"  if qf  is not None else "—"
        qs  = fval(small_name, "ssim_cyc_a");qss = f"{qs:.4f}" if qs  is not None else "—"
        print(f"  {label:<12}  {sf:>{col_w}}  {ss_:>{col_w}}  {qsf:>{col_w}}  {qss:>{col_w}}")

    print(
        "\nMetric consistency: if FID and SSIM rank backbones differently, this\n"
        "demonstrates the ET3 metric-disagreement phenomenon via architecture.\n"
        "Data consistency:   if the ranking flips between full/25% columns,\n"
        "report it as a key ET5 finding (backbone choice is data-regime-dependent).\n"
    )


if __name__ == "__main__":
    # ── Step 1: train the four new cells ─────────────────────────────────────
    print("\n" + "#"*70)
    print("# ET5 SETUP 1 — new training: perceptual loss at 25% data")
    print("#"*70)
    for cfg in SETUP1_NEW:
        train_setup1(*cfg)

    print("\n" + "#"*70)
    print("# ET5 SETUP 2 — new training: backbones at 25% data")
    print("#"*70)
    for cfg in SETUP2_NEW:
        train_setup2(*cfg)

    # ── Step 2: re-evaluate existing checkpoints with updated metrics ─────────
    # This ensures all cells in the comparison table were scored by the same
    # version of evaluate.py (which now includes KID and translation LPIPS).
    print("\n" + "#"*70)
    print("# ET5 RE-EVALUATION — existing checkpoints")
    print("#"*70)
    for name, netG, ngf in SETUP1_REEVAL + SETUP2_REEVAL:
        evaluate(name, netG, ngf)

    # ── Step 3: print interaction tables ─────────────────────────────────────
    print_summary()
