#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — HPC Environment Setup for SegmentedBSK / FasterSimulation
# =============================================================================
# [AI WARNING] DO NOT MODIFY THIS SCRIPT OR THE VENV CREATION LOGIC UNLESS
# EXPLICITLY ASKED. THE EXPORTS AND PYTHON ISOLATION ARE CRITICAL FOR
# HPC STABILITY.
# =============================================================================
#
# Creates a Python virtual environment (if it doesn't exist) and installs
# all required Python packages. Run this ONCE before submitting any SLURM jobs.
#
# Usage:
# setup_env.sh — Full HPC Environment Rebuild
# =============================================================================
set -euo pipefail   # exit on error, undefined var, pipe failure

# Force the script to ignore the ~/.local/ folder to prevent version conflicts
export PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ─── Configuration ────────────────────────────────────────────────────────────
ENV_DIR="$(cd "$(dirname "$0")" && pwd)/bsk_env"   # venv lives next to this script
PYTHON_MIN_MAJOR=3
PYTHON_MIN_MINOR=9

# Logging colors (if supported)
log()  { echo "[setup_env] $*"; }
ok()   { echo "[setup_env] ✓  $*"; }
warn() { echo "[setup_env] ⚠  $*"; }
fail() { echo "[setup_env] ✗  $*" >&2; exit 1; }

# ─── 1. Find Python ──────────────────────────────────────────────────────────
PYTHON="python3" # Using the module-loaded version from discovery

# ─── 2. Create virtual environment ────────────────────────────────────────────
if [ -d "$ENV_DIR" ]; then
    log "Removing existing environment at $ENV_DIR..."
    rm -rf "$ENV_DIR"
fi

log "Creating clean virtual environment at $ENV_DIR ..."
"$PYTHON" -m venv --system-site-packages "$ENV_DIR"
ok "Environment created."

VENV_PYTHON="$ENV_DIR/bin/python3"
VENV_PIP="$ENV_DIR/bin/pip"

# ─── 4. Upgrade pip & setuptools ─────────────────────────────────────────────
log "Upgrading pip and setuptools..."
"$VENV_PIP" install --quiet --upgrade pip setuptools wheel
ok "pip upgraded."

# ─── 5. Install required packages ─────────────────────────────────────────────
#
# Package → reason
# ─────────────────────────────────────────────────────────────────────────────
# numpy        — all numerical array operations
# matplotlib   — all plotting (Agg backend for headless HPC)
# h5py         — reading/writing HDF5 results files
# tqdm         — progress bars in main.py and mirror_plotting.py
# pyyaml       — YAML config parsing in utilities.py
# scipy        — LQR solver and signal processing in mirror_controller.py
# poppy        — physical optics / wavefront propagation (segmented_optics.py, mirror_plotting.py)
# astropy      — units (poppy dependency, used directly in segmented_optics.py)
# Pillow       — GIF/image creation in mirror_plotting.py (PIL.Image)
# ─────────────────────────────────────────────────────────────────────────────

PACKAGES=(
    "numpy"
    "matplotlib"
    "h5py"
    "tqdm"
    "pyyaml"
    "scipy"
    "poppy"
    "astropy"
    "Pillow"
    "pytest"           # required by Basilisk's unitTestSupport.py (imported at module load)
    "bsk"              # required for additional utilities
    "ipykernel"        # required for Jupyter Notebook kernel support
)

log "Installing Python packages..."
for pkg in "${PACKAGES[@]}"; do
    # Check if already installed to avoid slow redundant installs
    pkg_name="${pkg%%[>=<!]*}"   # strip any version specifier for the check
    if "$VENV_PYTHON" -c "import importlib.util; exit(0 if importlib.util.find_spec('${pkg_name,,}'.replace('-','_')) else 1)" &>/dev/null 2>&1; then
        ok "$pkg_name already installed — skipping."
    else
        log "Installing $pkg ..."
        "$VENV_PIP" install --quiet "$pkg"
        ok "$pkg installed."
    fi
done

# ─── 6. Verify all imports work ───────────────────────────────────────────────
log "Verifying imports inside venv..."

VERIFY_SCRIPT=$(cat <<'PYEOF'
import sys
failures = []
checks = [
    ("numpy",         "numpy"),
    ("matplotlib",    "matplotlib"),
    ("h5py",          "h5py"),
    ("tqdm",          "tqdm"),
    ("yaml (pyyaml)", "yaml"),
    ("scipy",         "scipy"),
    ("poppy",         "poppy"),
    ("astropy",       "astropy"),
    ("PIL (Pillow)",  "PIL"),
    ("pytest",        "pytest"),
]
for label, mod in checks:
    try:
        __import__(mod)
        print(f"  [OK]  {label}")
    except ImportError as e:
        print(f"  [FAIL] {label}: {e}")
        failures.append(label)

# Basilisk (system-level, may or may not be visible via --system-site-packages)
try:
    from Basilisk.architecture import messaging
    print("  [OK]  Basilisk.architecture.messaging")
except ImportError as e:
    print(f"  [WARN] Basilisk not importable from venv: {e}")
    print("         Make sure PYTHONPATH includes the Basilisk dist3 folder.")

sys.exit(len(failures))
PYEOF
)

"$VENV_PYTHON" -c "$VERIFY_SCRIPT"
EXIT_CODE=$?

echo ""
if [ "$EXIT_CODE" -eq 0 ]; then
    ok "All packages verified successfully!"
else
    warn "$EXIT_CODE package(s) failed to import. Review the output above."
fi

# ─── 7. Register Jupyter Kernel ──────────────────────────────────────────────
log "Registering Jupyter kernel..."
"$VENV_PYTHON" -m ipykernel install --user --name=bsk_env --display-name="Python (BSK Env)"
ok "Kernel 'Python (BSK Env)' registered."

# ─── 8. Print usage summary ───────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Environment ready: $ENV_DIR"
echo ""
echo "  To activate manually:"
echo "    source $ENV_DIR/bin/activate"
echo ""
echo "  Python path for SLURM scripts:"
echo "    $VENV_PYTHON"
echo ""
echo "  Update slurm_array.sh PYTHON variable to:"
echo "    PYTHON=$VENV_PYTHON"
echo ""
echo "  If Basilisk is not visible, add to SLURM job header:"
echo "    export PYTHONPATH=\$HOME/basilisk/dist3:\$PYTHONPATH"
echo "============================================================"
