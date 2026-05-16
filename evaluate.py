import sys
import csv
import torch
from pathlib import Path
from torchvision.utils import make_grid, save_image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure

sys.path.insert(0, str(Path(__file__).parent.parent))
from data import create_dataset
from models import create_model

# =============================================================================
# Configuration — override via CLI args or change defaults here
# =============================================================================
#EXPERIMENT_NAME  = "apple2orange_cyclegan_default"
EXPERIMENT_NAME  = "task2_lr0004"
DATAROOT         = "./datasets/apple2orange"
EPOCH            = "latest"
CHECKPOINTS_DIR  = "./checkpoints"
OUTPUT_DIR       = "./evaluation/results/"
NUM_SAMPLES      = 10
NETG             = "resnet_9blocks"   # must match the architecture used in training
NETD             = "basic"
NGF              = 64
NDF              = 64
NORM             = "instance"
IMAGE_SIZE       = 256
# FID feature dimension.  2048 is standard but unreliable on small datasets
# (~260 apple2orange test images — far fewer than the 2048-dim covariance needs).
# Lower values (192, 768) give more stable estimates; changing this one constant
# is the small-sample FID sensitivity experiment discussed in the ET3 report.
FID_FEATURE      = 2048
# KID is unbiased and robust on small sets; subset_size must be <= n_test_images.
KID_SUBSET_SIZE  = 50
# Fixed seed so metric computation is reproducible across runs (Inception/AlexNet
# forward passes are deterministic, but this guards any stochastic preprocessing).
EVAL_SEED        = 42
# =============================================================================


def build_opt(name, epoch, netG, ngf, norm, dataroot):
    """Build an inference-mode option namespace from explicit arguments.

    All values are passed in directly rather than read from module globals so
    that the function is safe to call in loops with different per-epoch arguments.
    """
    import argparse
    return argparse.Namespace(
        # Paths
        dataroot           = dataroot,
        name               = name,
        checkpoints_dir    = CHECKPOINTS_DIR,
        results_dir        = OUTPUT_DIR,
        # Model
        model              = "cycle_gan",
        netG               = netG,
        netD               = NETD,
        ngf                = ngf,
        ndf                = NDF,
        norm               = norm,
        no_dropout         = True,
        init_type          = "normal",
        init_gain          = 0.02,
        # Dataset
        dataset_mode       = "unaligned",
        direction          = "AtoB",
        input_nc           = 3,
        output_nc          = 3,
        load_size          = IMAGE_SIZE,
        crop_size          = IMAGE_SIZE,
        preprocess         = "resize_and_crop",
        no_flip            = True,
        serial_batches     = True,
        num_threads        = 0,
        batch_size         = 1,
        max_dataset_size   = float("inf"),
        # Test-specific
        epoch              = epoch,
        load_iter          = 0,
        isTrain            = False,
        phase              = "test",
        num_test           = float("inf"),
        eval               = True,
        # Display / logging
        display_winsize    = 256,
        aspect_ratio       = 1.0,
        verbose            = False,
        suffix             = "",
        use_wandb          = False,
        wandb_project_name = "CycleGAN-and-pix2pix",
        seed               = None,
    )


def to_uint8(t):
    return ((t.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)


def to_01(t):
    return (t.clamp(-1, 1) + 1) / 2


def save_sample_grid(real_A, fake_B, rec_A, real_B, fake_A, rec_B, out_path):
    """Save a 2×3 grid.
    Row 1: real_A | fake_B (A->B translation) | rec_A  (cycle reconstruction)
    Row 2: real_B | fake_A (B->A translation) | rec_B  (cycle reconstruction)
    """
    images = torch.cat([
        to_01(real_A), to_01(fake_B), to_01(rec_A),
        to_01(real_B), to_01(fake_A), to_01(rec_B),
    ], dim=0)
    grid = make_grid(images, nrow=3, padding=4, pad_value=1.0)
    save_image(grid, out_path)


def run_evaluation(name, epoch, netG, ngf, norm, dataroot, output_csv=None):
    """Evaluate one experiment checkpoint and (optionally) append metrics to a CSV.

    Metrics computed:
      Translation quality  — FID and KID (fake domain vs real target domain).
                             FID is standard; KID is unbiased on small datasets.
      Translation strength — LPIPS(fake_B, real_A) / LPIPS(fake_A, real_B).
                             Near-zero means the generator barely transforms the
                             input (under-translation / near-identity mapping).
      Cycle fidelity       — LPIPS and SSIM between rec and real.
                             High fidelity here does NOT imply good translation —
                             the identity mapping has perfect cycle consistency.

    These three groups can disagree.  Document the disagreement cases in the
    ET3 report as evidence that no single metric captures full translation quality.
    """
    torch.manual_seed(EVAL_SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    opt = build_opt(name, epoch, netG, ngf, norm, dataroot)
    opt.device = device

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    n_images = len(dataset)
    print(
        f"\nEvaluating {name} (epoch={epoch}, netG={netG}, ngf={ngf}) "
        f"on {n_images} images..."
    )
    if n_images < 500 and FID_FEATURE == 2048:
        print(
            f"  NOTE: FID(feature=2048) on {n_images} images is unreliable — "
            f"covariance estimation needs many more samples than feature dims. "
            f"Interpret FID with caution; KID is the more trustworthy metric here."
        )

    # -- Translation quality metrics ------------------------------------------
    fid_AtoB = FrechetInceptionDistance(feature=FID_FEATURE, normalize=False).to(device)
    fid_BtoA = FrechetInceptionDistance(feature=FID_FEATURE, normalize=False).to(device)
    # KID is unbiased; its mean and std together describe metric uncertainty.
    kid_AtoB = KernelInceptionDistance(subset_size=KID_SUBSET_SIZE, normalize=False).to(device)
    kid_BtoA = KernelInceptionDistance(subset_size=KID_SUBSET_SIZE, normalize=False).to(device)

    # -- Translation strength (perceptual distance between fake and its source) -
    # Useful to catch under-translation: if fake_B looks like real_A (LPIPS ≈ 0),
    # the generator is effectively doing nothing useful.
    lpips_trans_A = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(device)
    lpips_trans_B = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(device)

    # -- Cycle-reconstruction fidelity ----------------------------------------
    lpips_cyc_A = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(device)
    lpips_cyc_B = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(device)
    ssim_A = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ssim_B = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    samples_dir = Path(OUTPUT_DIR) / f"{name}_epoch{epoch}_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving {NUM_SAMPLES} sample grids to {samples_dir}")

    for i, data in enumerate(dataset):
        model.set_input(data)
        with torch.no_grad():
            model.forward()

        real_A = model.real_A
        real_B = model.real_B
        fake_B = model.fake_B   # G_A(real_A): A -> B
        fake_A = model.fake_A   # G_B(real_B): B -> A
        rec_A  = model.rec_A    # G_B(fake_B): reconstructed A
        rec_B  = model.rec_B    # G_A(fake_A): reconstructed B

        fid_AtoB.update(to_uint8(real_B), real=True)
        fid_AtoB.update(to_uint8(fake_B), real=False)
        fid_BtoA.update(to_uint8(real_A), real=True)
        fid_BtoA.update(to_uint8(fake_A), real=False)

        kid_AtoB.update(to_uint8(real_B), real=True)
        kid_AtoB.update(to_uint8(fake_B), real=False)
        kid_BtoA.update(to_uint8(real_A), real=True)
        kid_BtoA.update(to_uint8(fake_A), real=False)

        lpips_trans_A.update(fake_B.clamp(-1, 1), real_A.clamp(-1, 1))
        lpips_trans_B.update(fake_A.clamp(-1, 1), real_B.clamp(-1, 1))

        lpips_cyc_A.update(rec_A.clamp(-1, 1), real_A.clamp(-1, 1))
        lpips_cyc_B.update(rec_B.clamp(-1, 1), real_B.clamp(-1, 1))
        ssim_A.update(to_01(rec_A), to_01(real_A))
        ssim_B.update(to_01(rec_B), to_01(real_B))

        if i < NUM_SAMPLES:
            save_sample_grid(real_A, fake_B, rec_A, real_B, fake_A, rec_B,
                             samples_dir / f"sample_{i:03d}.png")
            save_image(to_01(real_A), samples_dir / f"sample_{i:03d}_realA.png")
            save_image(to_01(fake_B), samples_dir / f"sample_{i:03d}_fakeB.png")
            save_image(to_01(rec_A),  samples_dir / f"sample_{i:03d}_recA.png")
            save_image(to_01(real_B), samples_dir / f"sample_{i:03d}_realB.png")
            save_image(to_01(fake_A), samples_dir / f"sample_{i:03d}_fakeA.png")
            save_image(to_01(rec_B),  samples_dir / f"sample_{i:03d}_recB.png")


    fid_ab         = fid_AtoB.compute().item()
    fid_ba         = fid_BtoA.compute().item()
    kid_ab_m, kid_ab_s = kid_AtoB.compute()
    kid_ba_m, kid_ba_s = kid_BtoA.compute()
    lpips_tr_a     = lpips_trans_A.compute().item()
    lpips_tr_b     = lpips_trans_B.compute().item()
    lpips_cyc_a    = lpips_cyc_A.compute().item()
    lpips_cyc_b    = lpips_cyc_B.compute().item()
    ssim_cyc_a     = ssim_A.compute().item()
    ssim_cyc_b     = ssim_B.compute().item()

    print(f"\nResults for {name} (epoch={epoch}, netG={netG}):")
    print(f"  -- Translation quality (lower = more realistic) --")
    print(f"  FID  (A->B): {fid_ab:.2f}    FID  (B->A): {fid_ba:.2f}")
    print(f"  KID  (A->B): {kid_ab_m.item():.4f} ± {kid_ab_s.item():.4f}    "
          f"KID  (B->A): {kid_ba_m.item():.4f} ± {kid_ba_s.item():.4f}")
    print(f"  -- Translation strength (near 0 = under-translation / identity collapse) --")
    print(f"  LPIPS trans_A (fake_B vs real_A): {lpips_tr_a:.4f}")
    print(f"  LPIPS trans_B (fake_A vs real_B): {lpips_tr_b:.4f}")
    print(f"  -- Cycle fidelity (lower LPIPS / higher SSIM = better reconstruction) --")
    print(f"  LPIPS cycle_A: {lpips_cyc_a:.4f}    LPIPS cycle_B: {lpips_cyc_b:.4f}")
    print(f"  SSIM  cycle_A: {ssim_cyc_a:.4f}    SSIM  cycle_B: {ssim_cyc_b:.4f}")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_file = Path(OUTPUT_DIR) / f"{name}_epoch{epoch}.txt"
    with open(out_file, "w") as f:
        f.write(f"experiment: {name}\nepoch: {epoch}\nnetG: {netG}\nngf: {ngf}\n\n")
        f.write(f"FID  (A->B):     {fid_ab:.4f}\n")
        f.write(f"FID  (B->A):     {fid_ba:.4f}\n")
        f.write(f"KID  (A->B):     {kid_ab_m.item():.6f} +/- {kid_ab_s.item():.6f}\n")
        f.write(f"KID  (B->A):     {kid_ba_m.item():.6f} +/- {kid_ba_s.item():.6f}\n")
        f.write(f"LPIPS trans_A:   {lpips_tr_a:.4f}\n")
        f.write(f"LPIPS trans_B:   {lpips_tr_b:.4f}\n")
        f.write(f"LPIPS cycle_A:   {lpips_cyc_a:.4f}\n")
        f.write(f"LPIPS cycle_B:   {lpips_cyc_b:.4f}\n")
        f.write(f"SSIM  cycle_A:   {ssim_cyc_a:.4f}\n")
        f.write(f"SSIM  cycle_B:   {ssim_cyc_b:.4f}\n")
    print(f"\nSaved metrics to {out_file}")
    print(f"Saved {NUM_SAMPLES} sample grids to {samples_dir}")

    if output_csv:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "name", "epoch", "netG", "ngf", "fid_feature",
                    "fid_ab", "fid_ba",
                    "kid_ab_mean", "kid_ab_std", "kid_ba_mean", "kid_ba_std",
                    "lpips_trans_a", "lpips_trans_b",
                    "lpips_cyc_a", "lpips_cyc_b",
                    "ssim_cyc_a", "ssim_cyc_b",
                ])
            writer.writerow([
                name, epoch, netG, ngf, FID_FEATURE,
                f"{fid_ab:.4f}", f"{fid_ba:.4f}",
                f"{kid_ab_m.item():.6f}", f"{kid_ab_s.item():.6f}",
                f"{kid_ba_m.item():.6f}", f"{kid_ba_s.item():.6f}",
                f"{lpips_tr_a:.4f}", f"{lpips_tr_b:.4f}",
                f"{lpips_cyc_a:.4f}", f"{lpips_cyc_b:.4f}",
                f"{ssim_cyc_a:.4f}", f"{ssim_cyc_b:.4f}",
            ])
        print(f"Appended row to {csv_path}")


if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser(add_help=False)
    _p.add_argument("--name",       type=str, default=None)
    _p.add_argument("--dataroot",   type=str, default=None)
    _p.add_argument("--netG",       type=str, default=None)
    _p.add_argument("--ngf",        type=int, default=None)
    _p.add_argument("--norm",       type=str, default=None)
    _p.add_argument("--output_csv", type=str, default=None)
    # Comma-separated epoch list for convergence analysis, e.g. "20,40,60,80,100,latest"
    _p.add_argument("--epochs",     type=str, default=None,
                    help="comma-separated epochs to evaluate; defaults to EPOCH constant")
    _cli, _ = _p.parse_known_args()

    _name     = _cli.name     or EXPERIMENT_NAME
    _dataroot = _cli.dataroot or DATAROOT
    _netG     = _cli.netG     or NETG
    _ngf      = _cli.ngf      or NGF
    _norm     = _cli.norm     or NORM
    _epochs   = [e.strip() for e in _cli.epochs.split(",")] if _cli.epochs else [EPOCH]

    for _epoch in _epochs:
        run_evaluation(_name, _epoch, _netG, _ngf, _norm, _dataroot, _cli.output_csv)
