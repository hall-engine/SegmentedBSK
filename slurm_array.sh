#!/bin/bash
# =============================================================================
# slurm_array.sh — Multi-node HPC array job for the formation simulation
# =============================================================================
# Each array task runs ONE call to main.run() with a unique parameter combo.
# SLURM distributes tasks across whatever nodes are available.
#
# Usage:
#   1. Edit PARAM_I, PARAM_RAAN, PARAM_OMEGA below to match your sweep.
#   2. sbatch slurm_array.sh
#
# The --array index selects one (i, raan, omega) combo from the flat grid
# built at the bottom of this script (same logic as monte_carlo.py).
#
# Results land in the same results/ directory as a local run.
# =============================================================================

#SBATCH --job-name=optical_reef_mc
#SBATCH --output=logs/mc_%A_%a.out     # %A = job id, %a = array index
#SBATCH --error=logs/mc_%A_%a.err
#SBATCH --time=02:00:00                # wall-clock limit per task
#SBATCH --ntasks=1                     # 1 process per task (sim is single-proc)
#SBATCH --cpus-per-task=2              # BSK + mirror plotting
#SBATCH --mem=4G                       # per task

# ─── Parameter grid ───────────────────────────────────────────────────────────
# Edit these arrays.  Every combo gets one array task.
PARAM_I=(0)
PARAM_RAAN=(45)
PARAM_OMEGA=(0 45 90 135 180 225 270 315)

# ─── Build flat index → (i, raan, omega) mapping ─────────────────────────────
combos=()
for i in "${PARAM_I[@]}"; do
  for r in "${PARAM_RAAN[@]}"; do
    for o in "${PARAM_OMEGA[@]}"; do
      combos+=("${i},${r},${o}")
    done
  done
done

# Total number of tasks (set --array accordingly, 0-based)
N_COMBOS=${#combos[@]}
echo "Total combos: $N_COMBOS  |  This task index: $SLURM_ARRAY_TASK_ID"

# Guard against out-of-range index
if [ "$SLURM_ARRAY_TASK_ID" -ge "$N_COMBOS" ]; then
  echo "Array index out of range — exiting."
  exit 1
fi

# Parse the combo for this task
IFS=',' read -r I_DEG RAAN_DEG OMEGA_DEG <<< "${combos[$SLURM_ARRAY_TASK_ID]}"
echo "Running: i=$I_DEG  raan=$RAAN_DEG  omega=$OMEGA_DEG"

# ─── Run ──────────────────────────────────────────────────────────────────────
mkdir -p logs

PYTHON="$(dirname "$0")/bsk_env/bin/python3"   # created by setup_env.sh
# If the venv doesn't exist yet, run:  bash setup_env.sh
SIM_DIR="$(dirname "$0")"

"$PYTHON" - <<EOF
import sys, matplotlib
matplotlib.use("Agg")
sys.path.insert(0, "$SIM_DIR")
from config import SimConfig
import main as simulation

cfg = SimConfig()
simulation.run(
    cfg,
    base_i_deg     = $I_DEG,
    base_raan_deg  = $RAAN_DEG,
    base_omega_deg = $OMEGA_DEG,
    time_step_sec  = 0.5,
    read_every     = 100,
    show_plots     = True,
    mirror_plotting= True,
)
EOF

# Submit with:
#   sbatch --array=0-$((N_COMBOS-1)) slurm_array.sh
# e.g. for 8 combos:
#   sbatch --array=0-7 slurm_array.sh
