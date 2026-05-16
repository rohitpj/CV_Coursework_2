from pathlib import Path

from evaluate import run_evaluation

_REPO = Path(__file__).resolve().parent
CHECKPOINTS_DIR = str(_REPO / "checkpoints")
DATASET_DIR     = str(_REPO / "datasets" / "apple2orange")
RESULTS_CSV     = str(_REPO / "evaluation" / "all_results.csv")


def parse_opt(opt_path):
    opts = {}
    for line in opt_path.read_text().splitlines():
        if ":" not in line or line.strip().startswith("-"):
            continue
        key, _, rest = line.strip().partition(": ")
        opts[key.strip()] = rest.split("[")[0].strip()
    return opts


if __name__ == "__main__":
    checkpoints = Path(CHECKPOINTS_DIR)
    dirs = sorted(d for d in checkpoints.iterdir() if d.is_dir())

    skipped = []
    for d in dirs:
        if not (d / "latest_net_G_A.pth").exists():
            skipped.append(d.name)
            continue

        opt_file = d / "train_opt.txt"
        if opt_file.exists():
            opts = parse_opt(opt_file)
            netG = opts.get("netG", "resnet_9blocks")
            ngf  = int(opts.get("ngf", "64"))
            norm = opts.get("norm", "instance")
        else:
            netG, ngf, norm = "resnet_9blocks", 64, "instance"

        print(f"\nEvaluating: {d.name} (netG={netG}, ngf={ngf})")
        try:
            run_evaluation(d.name, "latest", netG, ngf, norm, DATASET_DIR, output_csv=RESULTS_CSV)
        except Exception as e:
            print(f"  WARNING: failed for {d.name}: {e}")

    if skipped:
        print(f"\nSkipped (no latest checkpoint): {', '.join(skipped)}")
    print(f"\nResults written to {RESULTS_CSV}")
