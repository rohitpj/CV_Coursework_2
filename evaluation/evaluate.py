"""
ET3: Quantitative Evaluation - FID, LPIPS, SSIM

Loads a trained CycleGAN checkpoint, runs inference on the test set, and
computes three complementary metrics:

  FID   - Frechet Inception Distance between the distribution of generated
          images and real images. Captures perceptual quality AND diversity.
          Computed per translation direction (A->B and B->A). Lower is better.

  LPIPS - Learned Perceptual Image Patch Similarity, measured on the cycle-
          reconstructed images (real_A vs rec_A, real_B vs rec_B). Tells you
          how faithfully content is preserved after a round-trip translation.
          Lower is better.

  SSIM  - Structural Similarity Index on the same cycle pairs. Cheap to
          compute and captures structural (edge/luminance) preservation.
          Higher is better.

Why these three together:
  FID captures generation quality at the distribution level.
  LPIPS captures perceptual cycle fidelity per image (uses deep features).
  SSIM captures low-level structural cycle fidelity (pixel statistics).
  They often disagree -- that disagreement is analytically interesting.

Usage:
  python evaluation/evaluate.py \\
    --name apple2orange_cyclegan_default \\
    --dataroot ./datasets/apple2orange \\
    --epoch latest

  # compare a specific saved epoch:
  python evaluation/evaluate.py \\
    --name apple2orange_cyclegan_default \\
    --dataroot ./datasets/apple2orange \\
    --epoch 100
"""

import argparse
import sys
from pathlib import Path

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure

# Add the repo root to sys.path so we can reuse its model/data infrastructure.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from options.test_options import TestOptions
from data import create_dataset
from models import create_model


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="ET3: Compute FID, LPIPS, SSIM for a trained CycleGAN model"
    )
    parser.add_argument("--name", type=str, required=True,
                        help="Experiment name (same as --name used during training)")
    parser.add_argument("--dataroot", type=str, required=True,
                        help="Path to dataset root (must contain testA/ and testB/)")
    parser.add_argument("--epoch", type=str, default="latest",
                        help="Checkpoint to load: 'latest' or an epoch number e.g. '100'")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints",
                        help="Root directory where experiment checkpoints are stored")
    parser.add_argument("--num_test", type=int, default=10000,
                        help="Max images to evaluate (default 10000 = effectively all)")
    parser.add_argument("--output_dir", type=str, default="./evaluation/results",
                        help="Directory to write the metric summary text file")
    parser.add_argument("--fid_feature", type=int, default=2048,
                        choices=[64, 192, 768, 2048],
                        help="Inception feature layer for FID (2048 = standard)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Tensor format helpers
# ---------------------------------------------------------------------------

def to_uint8(t):
    """[-1, 1] float tensor (B, C, H, W) -> uint8 tensor in [0, 255] for FID."""
    return ((t.clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0).to(torch.uint8)


def to_float01(t):
    """[-1, 1] float tensor -> [0, 1] float tensor for SSIM."""
    return (t.clamp(-1.0, 1.0) + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Model + dataset loader
# ---------------------------------------------------------------------------

def load_model_and_dataset(args, device):
    """
    Reuse the repo's TestOptions + create_model/create_dataset so we don't
    have to reimplement model loading or the data pipeline.

    We inject a synthetic sys.argv so TestOptions parses the right flags
    without the user having to know about all of them.
    """
    sys.argv = [
        "evaluate.py",
        "--dataroot",       args.dataroot,
        "--name",           args.name,
        "--checkpoints_dir", args.checkpoints_dir,
        "--model",          "cycle_gan",    # need both G_A and G_B
        "--dataset_mode",   "unaligned",
        "--epoch",          args.epoch,
        "--phase",          "test",
        "--no_dropout",                     # match training default
        "--load_size",      "256",          # no random crop during evaluation
        "--crop_size",      "256",
        "--no_flip",                        # deterministic evaluation
        "--serial_batches",                 # iterate in order
        "--num_test",       str(args.num_test),
        "--results_dir",    args.output_dir,
        "--eval",                           # puts BatchNorm/Dropout in eval mode
    ]
    opt = TestOptions().parse()
    opt.device = device
    opt.num_threads = 0    # single-threaded is fine for eval
    opt.batch_size = 1

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    return model, dataset, opt


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, dataset, opt = load_model_and_dataset(args, device)

    n_images = min(len(dataset), args.num_test)
    print(f"\nEvaluating '{args.name}' (epoch={args.epoch}) on {n_images} samples...\n")

    # -- Metric objects --------------------------------------------------

    # FID: needs uint8 images in [0, 255]; accumulates Inception features
    # internally before computing the Frechet distance at the end.
    fid_AtoB = FrechetInceptionDistance(feature=args.fid_feature, normalize=False).to(device)
    fid_BtoA = FrechetInceptionDistance(feature=args.fid_feature, normalize=False).to(device)

    # LPIPS: evaluate cycle consistency in perceptual space.
    # normalize=False because model tensors are already in [-1, 1].
    lpips_A = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(device)
    lpips_B = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(device)

    # SSIM: structural cycle consistency; data_range=1.0 because we convert to [0, 1].
    ssim_A = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ssim_B = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # -- Forward pass over test set --------------------------------------

    for i, data in enumerate(dataset):
        if i >= args.num_test:
            break

        model.set_input(data)
        with torch.no_grad():
            model.forward()  # populates real_A, real_B, fake_B, fake_A, rec_A, rec_B

        real_A = model.real_A   # (1, 3, H, W)  in [-1, 1]
        real_B = model.real_B
        fake_B = model.fake_B   # G_A(real_A) : apple -> orange
        fake_A = model.fake_A   # G_B(real_B) : orange -> apple
        rec_A  = model.rec_A    # G_B(fake_B) : round-trip reconstruction of A
        rec_B  = model.rec_B    # G_A(fake_A) : round-trip reconstruction of B

        # FID -- compare generated distribution against real distribution
        fid_AtoB.update(to_uint8(real_B), real=True)
        fid_AtoB.update(to_uint8(fake_B), real=False)
        fid_BtoA.update(to_uint8(real_A), real=True)
        fid_BtoA.update(to_uint8(fake_A), real=False)

        # LPIPS -- perceptual distance between original and cycle-reconstructed
        lpips_A.update(rec_A.clamp(-1, 1), real_A.clamp(-1, 1))
        lpips_B.update(rec_B.clamp(-1, 1), real_B.clamp(-1, 1))

        # SSIM -- structural distance between original and cycle-reconstructed
        ssim_A.update(to_float01(rec_A), to_float01(real_A))
        ssim_B.update(to_float01(rec_B), to_float01(real_B))

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1:>4}/{n_images}] images processed")

    # -- Compute final scores -------------------------------------------

    fid_ab       = fid_AtoB.compute().item()
    fid_ba       = fid_BtoA.compute().item()
    lpips_cyc_a  = lpips_A.compute().item()
    lpips_cyc_b  = lpips_B.compute().item()
    ssim_cyc_a   = ssim_A.compute().item()
    ssim_cyc_b   = ssim_B.compute().item()

    # -- Print table ----------------------------------------------------

    W = 62
    print("\n" + "=" * W)
    print(f"  Results: {args.name}  |  epoch = {args.epoch}  |  n = {i + 1}")
    print("=" * W)
    print(f"  {'Metric':<34} {'Direction':<10} {'Score':>8}")
    print("-" * W)
    print(f"  {'FID  (lower = better)':<34} {'A -> B':<10} {fid_ab:>8.2f}")
    print(f"  {'FID  (lower = better)':<34} {'B -> A':<10} {fid_ba:>8.2f}")
    print("-" * W)
    print(f"  {'Cycle LPIPS  (lower = better)':<34} {'A->B->A':<10} {lpips_cyc_a:>8.4f}")
    print(f"  {'Cycle LPIPS  (lower = better)':<34} {'B->A->B':<10} {lpips_cyc_b:>8.4f}")
    print("-" * W)
    print(f"  {'Cycle SSIM   (higher = better)':<34} {'A->B->A':<10} {ssim_cyc_a:>8.4f}")
    print(f"  {'Cycle SSIM   (higher = better)':<34} {'B->A->B':<10} {ssim_cyc_b:>8.4f}")
    print("=" * W)

    # -- Save to file ---------------------------------------------------

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.name}_epoch{args.epoch}.txt"

    with open(out_path, "w") as f:
        f.write(f"experiment:      {args.name}\n")
        f.write(f"epoch:           {args.epoch}\n")
        f.write(f"n_images:        {i + 1}\n")
        f.write(f"fid_feature:     {args.fid_feature}\n\n")
        f.write(f"FID (A->B):      {fid_ab:.4f}\n")
        f.write(f"FID (B->A):      {fid_ba:.4f}\n")
        f.write(f"LPIPS cycle_A:   {lpips_cyc_a:.4f}\n")
        f.write(f"LPIPS cycle_B:   {lpips_cyc_b:.4f}\n")
        f.write(f"SSIM cycle_A:    {ssim_cyc_a:.4f}\n")
        f.write(f"SSIM cycle_B:    {ssim_cyc_b:.4f}\n")

    print(f"\nResults saved to: {out_path}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
