"""
monte_carlo.py — Semi-major axis sweep for the formation simulation
===============================================================
Runs the same configuration multiple times, each with a different
semi-major axis (a_geo), to quantify variability across orbital altitudes.

Run with:
    python monte_carlo.py

Edit A_GEO_VALUES below to control which semi-major axes are swept.
Edit N_WORKERS to control parallelism.

Notes
-----
- Uses `multiprocessing` with the `spawn` context (macOS default).
  Each worker is a completely independent Python interpreter — no shared
  Basilisk state, no thread-safety issues.
- Workers call main.run(**kwargs) exactly as main.py's __main__ block does.
- If a worker crashes it prints the traceback and the sweep continues.
- Results go to the same results_base as a serial run (from SimConfig).
- On HPC: set --cpus-per-task in your SLURM script; N_WORKERS auto-reads
  SLURM_CPUS_PER_TASK.  For multi-node HPC use RUNSIM.job array mode.
"""

import multiprocessing
import os
import traceback
import numpy as np

import matplotlib
matplotlib.use("Agg")   # no display in worker processes


import itertools

# ─── PARAMETER GRID ────────────────────────────────────────────────────────────
# The grid calculates every unique combination of these variables.
# Total Runs = len(RING_VALUES) * len(FOCAL_LENGTH_VALUES)

RING_VALUES         = [1, 5, 10]
FOCAL_LENGTH_VALUES = np.linspace(2000, 25000, 100)

# These kwargs are passed to main.run() for EVERY simulation (fixed settings).
FIXED_KWARGS = dict(
    time_step_sec    = 0.5,     # sim timestep [s]
    read_every       = 100,     # mirror plotting frame interval
    show_plots       = False,   # save all plots after each sim
    save_data        = True,    # keep h5 and config saved
    mirror_plotting  = False,   # run mirror animation (slow — keep False for sweeps)
    disable_progress = True,    # suppress tqdm in workers
)

# Number of sims to run simultaneously.
# On HPC: reads SLURM_CPUS_PER_TASK.  Locally: all cores minus 2.
_slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
N_WORKERS = int(_slurm_cpus) if _slurm_cpus else max(1, os.cpu_count() - 2)

# =============================================================================


def _worker(kwargs: dict) -> str:
    """
    Runs ONE simulation in a separate worker process.

    Returns a short status string ("✓ ..." or "✗ ...") for the summary.
    """
    import matplotlib.pyplot as plt
    from config import SimConfig
    import main as simulation

    rings = kwargs.get("rings")
    focal = kwargs.get("target_focal_length")
    
    # Tag logic for printouts
    tag  = f"rings={rings}_focal={focal:.0f}"
    print(f"[START] {tag}", flush=True)
    try:
        cfg = SimConfig()
        simulation.run(cfg, **kwargs)
        plt.close("all")
        print(f"[DONE]  {tag}", flush=True)
        return f"✓  {tag}"
    except Exception:
        print(f"\n[monte_carlo] WORKER FAILED — {tag}\n{traceback.format_exc()}",
              flush=True)
        return f"✗  {tag}  (FAILED — see stderr)"


def build_param_grid() -> list[dict]:
    """Return one kwarg dict per parameter set."""
    grid = []
    # Generate every massive combinatorics tuple via python's itertools!
    for rings, focal in itertools.product(RING_VALUES, FOCAL_LENGTH_VALUES):
        kw = dict(FIXED_KWARGS)
        kw["rings"] = int(rings)
        kw["target_focal_length"] = float(focal)
        grid.append(kw)
    return grid


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    grid      = build_param_grid()
    n_sims    = len(grid)
    
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--array-id", type=int, default=None, 
                        help="Inject a specific grid index directly (used by SLURM Mode B)")
    args = parser.parse_args()

    if args.array_id is not None:
        # MODE B: Array Task Run (Single Core)
        idx = args.array_id
        if idx < 0 or idx >= n_sims:
            print(f"Error: SLURM Array ID {idx} does not map to a grid index (0 to {n_sims-1})")
            sys.exit(1)
        print(f"=== SLURM Array Mode Injected: Executing grid index {idx}/{n_sims-1} ===")
        res = _worker(grid[idx])
        print(res)
    else:
        # MODE A: Full Multiprocessing Parallel Sweep
        n_workers = min(N_WORKERS, n_sims)

        print("=" * 60)
        print(f"  2D Parameter Sweep: {n_sims} sims, {n_workers} parallel workers")
        print(f"  rings: {len(RING_VALUES)} steps")
        print(f"  focal: {len(FOCAL_LENGTH_VALUES)} steps")
        print("=" * 60)

        with multiprocessing.Pool(processes=n_workers) as pool:
            results = pool.map(_worker, grid)

        print("\n" + "=" * 60)
        print("  Results summary:")
        for r in results:
            print(f"    {r}")
        print("=" * 60)
