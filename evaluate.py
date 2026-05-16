import sys
from pathlib import Path
import torch
from torchvision.utils import make_grid, save_image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure

sys.path.insert(0, str(Path(__file__).parent.parent))
from data import create_dataset
from models import create_model

# =============================================================================
# Configuration — change these to evaluate a different experiment
# =============================================================================
EXPERIMENT_NAME  = "et2_resnet9"
DATAROOT         = "./datasets/apple2orange"
EPOCH            = "latest"
CHECKPOINTS_DIR  = "./checkpoints"
OUTPUT_DIR       = "./evaluation/results/"
NUM_SAMPLES      = 10        # how many sample comparison grids to save
NETG             = "resnet_9blocks"         # must match the architecture used in training
                                      # e.g. "resnet_9blocks", "unet_256", "unet_128"
NETD             = "basic"            # "basic" (PatchGAN 70x70), "pixel", "n_layers"
NGF              = 64                 # number of generator filters
NDF              = 64                 # number of discriminator filters
NORM             = "instance"         # "instance" or "batch"
IMAGE_SIZE       = 256               # must match the size used in training
# =============================================================================


def build_opt():
    import argparse
    opt = argparse.Namespace(
        # Paths
        dataroot         = DATAROOT,
        name             = EXPERIMENT_NAME,
        checkpoints_dir  = CHECKPOINTS_DIR,
        results_dir      = OUTPUT_DIR,
        # Model
        model            = "cycle_gan",
        netG             = NETG,
        netD             = NETD,
        ngf              = NGF,
        ndf              = NDF,
        norm             = NORM,
        no_dropout       = True,
        init_type        = "normal",
        init_gain        = 0.02,
        # Dataset
        dataset_mode     = "unaligned",
        direction        = "AtoB",
        input_nc         = 3,
        output_nc        = 3,
        load_size        = IMAGE_SIZE,
        crop_size        = IMAGE_SIZE,
        preprocess       = "resize_and_crop",
        no_flip          = True,
        serial_batches   = True,
        num_threads      = 0,
        batch_size       = 1,
        max_dataset_size = float("inf"),
        # Test-specific
        epoch            = EPOCH,
        load_iter        = 0,
        isTrain          = False,
        phase            = "test",
        num_test         = float("inf"),
        eval             = True,
        # Display / logging
        display_winsize  = 256,
        aspect_ratio     = 1.0,
        verbose          = False,
        suffix           = "",
        use_wandb        = False,
        wandb_project_name = "CycleGAN-and-pix2pix",
    )
    return opt


def to_uint8(t):
    return ((t.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)


def to_01(t):
    return (t.clamp(-1, 1) + 1) / 2


def save_sample_grid(real_A, fake_B, rec_A, real_B, fake_A, rec_B, out_path):
    """Save a 2x3 grid showing both translation directions and reconstructions.
    Row 1: real_A | fake_B (A->B) | rec_A  (cycle A->B->A)
    Row 2: real_B | fake_A (B->A) | rec_B  (cycle B->A->B)
    """
    images = torch.cat([
        to_01(real_A), to_01(fake_B), to_01(rec_A),
        to_01(real_B), to_01(fake_A), to_01(rec_B),
    ], dim=0)
    grid = make_grid(images, nrow=3, padding=4, pad_value=1.0)
    save_image(grid, out_path)


def run_evaluation():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    opt = build_opt()
    opt.device = device

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    fid_AtoB = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    fid_BtoA = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    lpips_A  = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(device)
    lpips_B  = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(device)
    ssim_A   = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ssim_B   = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # Prepare samples directory
    samples_dir = Path(OUTPUT_DIR) / f"{EXPERIMENT_NAME}_epoch{EPOCH}_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEvaluating {EXPERIMENT_NAME} (epoch={EPOCH}) on {len(dataset)} images...")
    print(f"Saving {NUM_SAMPLES} sample comparison grids to {samples_dir}")

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

        lpips_A.update(rec_A.clamp(-1, 1), real_A.clamp(-1, 1))
        lpips_B.update(rec_B.clamp(-1, 1), real_B.clamp(-1, 1))

        ssim_A.update(to_01(rec_A), to_01(real_A))
        ssim_B.update(to_01(rec_B), to_01(real_B))

        # Save the first NUM_SAMPLES comparison grids
        if i < NUM_SAMPLES:
            grid_path = samples_dir / f"sample_{i:03d}.png"
            save_sample_grid(real_A, fake_B, rec_A, real_B, fake_A, rec_B, grid_path)
            # Also save individual images for use in the report
            save_image(to_01(real_A), samples_dir / f"sample_{i:03d}_realA.png")
            save_image(to_01(fake_B), samples_dir / f"sample_{i:03d}_fakeB.png")
            save_image(to_01(rec_A),  samples_dir / f"sample_{i:03d}_recA.png")
            save_image(to_01(real_B), samples_dir / f"sample_{i:03d}_realB.png")
            save_image(to_01(fake_A), samples_dir / f"sample_{i:03d}_fakeA.png")
            save_image(to_01(rec_B),  samples_dir / f"sample_{i:03d}_recB.png")

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(dataset)} images done")

    fid_ab      = fid_AtoB.compute().item()
    fid_ba      = fid_BtoA.compute().item()
    lpips_cyc_a = lpips_A.compute().item()
    lpips_cyc_b = lpips_B.compute().item()
    ssim_cyc_a  = ssim_A.compute().item()
    ssim_cyc_b  = ssim_B.compute().item()

    print(f"\nResults for {EXPERIMENT_NAME} (epoch={EPOCH}):")
    print(f"  FID (A->B):      {fid_ab:.2f}   (lower is better)")
    print(f"  FID (B->A):      {fid_ba:.2f}   (lower is better)")
    print(f"  LPIPS cycle_A:   {lpips_cyc_a:.4f}  (lower is better)")
    print(f"  LPIPS cycle_B:   {lpips_cyc_b:.4f}  (lower is better)")
    print(f"  SSIM cycle_A:    {ssim_cyc_a:.4f}  (higher is better)")
    print(f"  SSIM cycle_B:    {ssim_cyc_b:.4f}  (higher is better)")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_file = Path(OUTPUT_DIR) / f"{EXPERIMENT_NAME}_epoch{EPOCH}.txt"
    with open(out_file, "w") as f:
        f.write(f"experiment: {EXPERIMENT_NAME}\n")
        f.write(f"epoch:      {EPOCH}\n")
        f.write(f"netG:       {NETG}\n\n")
        f.write(f"FID (A->B):     {fid_ab:.4f}\n")
        f.write(f"FID (B->A):     {fid_ba:.4f}\n")
        f.write(f"LPIPS cycle_A:  {lpips_cyc_a:.4f}\n")
        f.write(f"LPIPS cycle_B:  {lpips_cyc_b:.4f}\n")
        f.write(f"SSIM cycle_A:   {ssim_cyc_a:.4f}\n")
        f.write(f"SSIM cycle_B:   {ssim_cyc_b:.4f}\n")

    print(f"\nSaved metrics to {out_file}")
    print(f"Saved {NUM_SAMPLES} sample grids to {samples_dir}")


if __name__ == "__main__":
    run_evaluation()