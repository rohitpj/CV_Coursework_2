"""
system_info.py
Collects all hardware and software information required for the reproducibility
section of the coursework report. Run once before submission and paste the output
into Section 3 (Experimental Setup) of your report.

Usage:
    python system_info.py
    python system_info.py > system_info.txt   # save to file
"""

import sys
import os
import platform
import subprocess
from pathlib import Path
from datetime import datetime


# ── helpers ──────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def row(label, value):
    print(f"  {label:<35} {value}")


def run(cmd):
    """Run a shell command and return stdout, or None on failure."""
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


# ── 1. operating system ───────────────────────────────────────────────────────

section("1. Operating System")
row("OS",            platform.system())
row("OS Release",    platform.release())
row("OS Version",    platform.version())
row("Architecture",  platform.machine())
row("Hostname",      platform.node())
row("Python",        sys.version.replace("\n", " "))
row("Python path",   sys.executable)
row("Date/time",     datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# ── 2. cpu ────────────────────────────────────────────────────────────────────

section("2. CPU")
cpu_name = (
    run("lscpu | grep 'Model name' | cut -d: -f2")
    or run("sysctl -n machdep.cpu.brand_string")
    or platform.processor()
    or "Unknown"
)
row("CPU model",     cpu_name.strip())

cpu_cores_physical = run("lscpu | grep '^Core(s) per socket' | awk '{print $NF}'") or "?"
cpu_cores_logical  = run("nproc") or str(os.cpu_count())
row("Physical cores", cpu_cores_physical)
row("Logical cores",  cpu_cores_logical)

mem_kb = run("grep MemTotal /proc/meminfo | awk '{print $2}'")
if mem_kb:
    row("Total RAM",  f"{int(mem_kb) / 1024 / 1024:.1f} GB")
else:
    row("Total RAM",  run("sysctl -n hw.memsize") or "Unknown")


# ── 3. gpu ────────────────────────────────────────────────────────────────────

section("3. GPU (nvidia-smi)")
smi = run("nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap "
          "--format=csv,noheader")
if smi:
    for gpu_idx, line in enumerate(smi.splitlines()):
        parts = [p.strip() for p in line.split(",")]
        row(f"GPU {gpu_idx} name",         parts[0] if len(parts) > 0 else "?")
        row(f"GPU {gpu_idx} driver",       parts[1] if len(parts) > 1 else "?")
        row(f"GPU {gpu_idx} VRAM",         parts[2] if len(parts) > 2 else "?")
        row(f"GPU {gpu_idx} compute cap.", parts[3] if len(parts) > 3 else "?")
else:
    row("nvidia-smi", "Not available (no NVIDIA GPU or driver not installed)")


# ── 4. cuda / cudnn ───────────────────────────────────────────────────────────

section("4. CUDA / cuDNN")
try:
    import torch
    row("CUDA available",   str(torch.cuda.is_available()))
    row("CUDA version",     torch.version.cuda or "N/A")
    row("cuDNN version",    str(torch.backends.cudnn.version()) if torch.cuda.is_available() else "N/A")
    row("cuDNN enabled",    str(torch.backends.cudnn.enabled))
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            row(f"Device {i} name",      props.name)
            row(f"Device {i} VRAM",      f"{props.total_memory / 1024**3:.1f} GB")
            row(f"Device {i} SM count",  str(props.multi_processor_count))
except ImportError:
    row("torch", "Not installed")


# ── 5. python packages ────────────────────────────────────────────────────────

section("5. Key Python Package Versions")
packages = [
    "torch",
    "torchvision",
    "numpy",
    "scipy",
    "pandas",
    "cv2",          # opencv-python
    "PIL",          # Pillow
    "skimage",      # scikit-image
    "matplotlib",
    "seaborn",
    "tensorboard",
    "tqdm",
    "sklearn",      # scikit-learn
    "torchmetrics",
]

for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "unknown")
        row(pkg, ver)
    except ImportError:
        row(pkg, "NOT INSTALLED")


# ── 6. full pip freeze (for complete reproducibility) ─────────────────────────

section("6. Full Environment (pip freeze)")
freeze = run("pip freeze")
if freeze:
    for line in freeze.splitlines():
        print(f"  {line}")
else:
    print("  pip freeze failed — try running manually.")


# ── 7. conda environment (if applicable) ──────────────────────────────────────

section("7. Conda Environment (if applicable)")
conda_info = run("conda info --json")
if conda_info:
    import json
    try:
        info = json.loads(conda_info)
        row("Conda version",      info.get("conda_version", "?"))
        row("Active environment", info.get("active_prefix_name", "?"))
        row("Active prefix",      info.get("active_prefix", "?"))
    except Exception:
        print("  Could not parse conda info.")
else:
    print("  Conda not detected.")


# ── 8. repository info ────────────────────────────────────────────────────────

section("8. Repository / Code Info")
git_hash   = run("git rev-parse HEAD")
git_branch = run("git rev-parse --abbrev-ref HEAD")
git_remote = run("git remote get-url origin")
row("Git commit hash",  git_hash   or "Not a git repo")
row("Git branch",       git_branch or "?")
row("Git remote",       git_remote or "?")
row("Working dir",      str(Path.cwd()))


# ── footer ────────────────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print("  Done. Copy the output above into Section 3 of your report.")
print(f"{'=' * 60}\n")