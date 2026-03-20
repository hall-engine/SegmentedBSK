#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — HPC Environment Setup for SegmentedBSK / FasterSimulation
# =============================================================================
#
# Creates a Python virtual environment (if it doesn't exist) and installs
# all required Python packages. Run this ONCE before submitting any SLURM jobs.
#
# Usage:
#   bash setup_env.sh
#
# After running, activate the environment with:
#   source ./bsk_env/bin/activate
#
# Then update slurm_array.sh PYTHON variable to point to:
#   ./bsk_env/bin/python3
# =============================================================================

set -euo pipefail   # exit on error, undefined var, pipe failure

# ─── Configuration ────────────────────────────────────────────────────────────
ENV_DIR="$(cd "$(dirname "$0")" && pwd)/bsk_env"   # venv lives next to this script
PYTHON_MIN_MAJOR=3
PYTHON_MIN_MINOR=9    # Basilisk requires Python ≥ 3.9

# ─── Helper functions ─────────────────────────────────────────────────────────
log()  { echo "[setup_env] $*"; }
ok()   { echo "[setup_env] ✓  $*"; }
warn() { echo "[setup_env] ⚠  $*"; }
fail() { echo "[setup_env] ✗  $*" >&2; exit 1; }

# ─── 1. Find a suitable Python interpreter ────────────────────────────────────
log "Looking for a Python interpreter (≥${PYTHON_MIN_MAJOR}.${PYTHON_MIN_MINOR})..."

PYTHON=""
for candidate in python3 python python3.12 python3.11 python3.10 python3.9; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        major="${ver%%.*}"
        minor="${ver##*.}"
        if [ "$major" -ge "$PYTHON_MIN_MAJOR" ] && [ "$minor" -ge "$PYTHON_MIN_MINOR" ]; then
            PYTHON="$candidate"
            ok "Using $PYTHON (version $ver)"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    # On many HPC clusters Python is behind a module system — try common names
    warn "No suitable Python found in PATH. Trying HPC module system..."
    for mod in python/3.11 python/3.10 python3/3.11 python3/3.10; do
        if module load "$mod" &>/dev/null 2>&1; then
            PYTHON="python3"
            ok "Loaded module $mod"
            break
        fi
    done
fi

if [ -z "$PYTHON" ]; then
    fail "Could not find Python ≥ ${PYTHON_MIN_MAJOR}.${PYTHON_MIN_MINOR}. " \
         "Load the appropriate module manually (e.g. 'module load python/3.11') " \
         "and re-run this script."
fi

# ─── 2. Check if Basilisk is importable (must be pre-installed on the HPC) ───
log "Checking for Basilisk..."
if "$PYTHON" -c "from Basilisk.architecture import messaging" &>/dev/null 2>&1; then
    ok "Basilisk already importable from the system Python."
    BASILISK_PYTHON="$PYTHON"   # we can use system python with its Basilisk
else
    # Basilisk may live in a specific conda/venv — check common HPC locations
    warn "Basilisk not found in '$PYTHON'. Searching common HPC locations..."
    BASILISK_PYTHON=""
    for bsk_candidate in \
        "$HOME/basilisk/dist3/bsk_env/bin/python3" \
        "$HOME/dev/basilisk_env/bin/python3" \
        "$HOME/basilisk_env/bin/python3" \
        "/opt/basilisk/bin/python3"; do
        if [ -x "$bsk_candidate" ] && "$bsk_candidate" -c "from Basilisk.architecture import messaging" &>/dev/null 2>&1; then
            BASILISK_PYTHON="$bsk_candidate"
            ok "Found Basilisk-capable Python: $BASILISK_PYTHON"
            break
        fi
    done

    if [ -z "$BASILISK_PYTHON" ]; then
        warn "============================================================"
        warn "Basilisk not found automatically."
        warn "Basilisk must be built/installed separately on the HPC."
        warn "See: https://hanspeterschaub.info/basilisk/Install/installBasilisk.html"
        warn "After installing, add its site-packages to PYTHONPATH:"
        warn "  export PYTHONPATH=\$HOME/basilisk/dist3:\$PYTHONPATH"
        warn "Then re-run this script."
        warn "============================================================"
        # We continue anyway — other packages can still be installed
        BASILISK_PYTHON="$PYTHON"
    else
        PYTHON="$BASILISK_PYTHON"
    fi
fi

# ─── 3. Create virtual environment ────────────────────────────────────────────
if [ -d "$ENV_DIR" ]; then
    ok "Virtual environment already exists at $ENV_DIR — skipping creation."
else
    log "Creating virtual environment at $ENV_DIR ..."
    # --system-site-packages lets the venv see any system-level Basilisk install
    "$PYTHON" -m venv --system-site-packages "$ENV_DIR"
    ok "Virtual environment created."
fi

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

# ─── 7. Print usage summary ───────────────────────────────────────────────────
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
