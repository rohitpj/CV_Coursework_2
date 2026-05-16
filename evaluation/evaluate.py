import sys
from pathlib import Path
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure

sys.path.insert(0, str(Path(__file__).parent.parent))
from options.test_options import TestOptions
from data import create_dataset
from models import create_model

# Change these to evaluate a different experiment
EXPERIMENT_NAME = "apple2orange_cyclegan_default"
DATAROOT = "./datasets/apple2orange"
EPOCH = "latest"
CHECKPOINTS_DIR = "./checkpoints"
OUTPUT_DIR = "./evaluation/results"


def to_uint8(t):
    return ((t.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)


def to_01(t):
    return (t.clamp(-1, 1) + 1) / 2


def run_evaluation():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Inject args so TestOptions can parse them (reuses the repo's model loading)
    sys.argv = [
        "evaluate.py",
        "--dataroot",        DATAROOT,
        "--name",            EXPERIMENT_NAME,
        "--checkpoints_dir", CHECKPOINTS_DIR,
        "--model",           "cycle_gan",
        "--dataset_mode",    "unaligned",
        "--epoch",           EPOCH,
        "--phase",           "test",
        "--no_dropout",
        "--load_size",       "256",
        "--crop_size",       "256",
        "--no_flip",
        "--serial_batches",
        "--results_dir",     OUTPUT_DIR,
        "--eval",
    ]

    opt = TestOptions().parse()
    opt.device = device
    opt.num_threads = 0
    opt.batch_size = 1

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    fid_AtoB = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    fid_BtoA = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    lpips_A = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(device)
    lpips_B = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(device)
    ssim_A = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ssim_B = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    print(f"\nEvaluating {EXPERIMENT_NAME} (epoch={EPOCH}) on {len(dataset)} images...")

    for i, data in enumerate(dataset):
        model.set_input(data)
        with torch.no_grad():
            model.forward()

        real_A = model.real_A
        real_B = model.real_B
        fake_B = model.fake_B   # G_A(real_A): apple -> orange
        fake_A = model.fake_A   # G_B(real_B): orange -> apple
        rec_A = model.rec_A     # G_B(fake_B): reconstructed apple
        rec_B = model.rec_B     # G_A(fake_A): reconstructed orange

        fid_AtoB.update(to_uint8(real_B), real=True)
        fid_AtoB.update(to_uint8(fake_B), real=False)
        fid_BtoA.update(to_uint8(real_A), real=True)
        fid_BtoA.update(to_uint8(fake_A), real=False)

        lpips_A.update(rec_A.clamp(-1, 1), real_A.clamp(-1, 1))
        lpips_B.update(rec_B.clamp(-1, 1), real_B.clamp(-1, 1))

        ssim_A.update(to_01(rec_A), to_01(real_A))
        ssim_B.update(to_01(rec_B), to_01(real_B))

        if (i + 1) % 50 == 0:
            print(f"  {i + 1} images done")

    fid_ab = fid_AtoB.compute().item()
    fid_ba = fid_BtoA.compute().item()
    lpips_cyc_a = lpips_A.compute().item()
    lpips_cyc_b = lpips_B.compute().item()
    ssim_cyc_a = ssim_A.compute().item()
    ssim_cyc_b = ssim_B.compute().item()

    print(f"\nResults for {EXPERIMENT_NAME} (epoch={EPOCH}):")
    print(f"  FID (A->B):      {fid_ab:.2f}   (lower is better)")
    print(f"  FID (B->A):      {fid_ba:.2f}   (lower is better)")
    print(f"  LPIPS cycle_A:   {lpips_cyc_a:.4f}  (lower is better)")
    print(f"  LPIPS cycle_B:   {lpips_cyc_b:.4f}  (lower is better)")
    print(f"  SSIM cycle_A:    {ssim_cyc_a:.4f}  (higher is better)")
    print(f"  SSIM cycle_B:    {ssim_cyc_b:.4f}  (higher is better)")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_file = f"{OUTPUT_DIR}/{EXPERIMENT_NAME}_epoch{EPOCH}.txt"
    with open(out_file, "w") as f:
        f.write(f"experiment: {EXPERIMENT_NAME}\n")
        f.write(f"epoch: {EPOCH}\n\n")
        f.write(f"FID (A->B):     {fid_ab:.4f}\n")
        f.write(f"FID (B->A):     {fid_ba:.4f}\n")
        f.write(f"LPIPS cycle_A:  {lpips_cyc_a:.4f}\n")
        f.write(f"LPIPS cycle_B:  {lpips_cyc_b:.4f}\n")
        f.write(f"SSIM cycle_A:   {ssim_cyc_a:.4f}\n")
        f.write(f"SSIM cycle_B:   {ssim_cyc_b:.4f}\n")
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    run_evaluation()
