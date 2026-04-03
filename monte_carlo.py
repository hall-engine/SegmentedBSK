"""
monte_carlo.py — Orbital parameter sweep for the formation simulation
======================================================================
Runs the same configuration multiple times, each with different
orbital parameters (inclination, RAAN, argument of periapsis) to
characterise performance across different orbit geometries.

Run with:
    python monte_carlo.py

Edit N_WORKERS to control parallelism.

Notes
-----
- Uses `multiprocessing` with the `spawn` context (macOS default).
  Each worker is a completely independent Python interpreter — no shared
  Basilisk state, no thread-safety issues.
- Workers call main.run(**kwargs) exactly as main.py's __main__ block does.
- If a worker crashes it prints the traceback and the sweep continues.
- Results go to results/controller_sweep/<param_tag>/ (one sub-folder per combo).
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

# Sweep over orbital parameters
BASE_I_DEG_VALUES     = [0.0]           # Base inclination [deg]
BASE_RAAN_DEG_VALUES  = [0.0,45.0, 90.0, 180.0, 270, 360]          # RAAN [deg]
BASE_OMEGA_DEG_VALUES = [0.0]           # Argument of periapsis [deg]


# These kwargs are passed to main.run() for EVERY simulation (fixed settings).
FIXED_KWARGS = dict(
    read_every       = 100,     # mirror plotting frame interval
    show_plots       = True,   # save all plots after each sim
    save_data        = True,    # keep h5 and config saved
    mirror_plotting  = True,   # run mirror animation (slow — keep False for sweeps)
    disable_progress = False,    # suppress tqdm in workers
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

    i_deg    = kwargs.get("base_i_deg")
    raan_deg = kwargs.get("base_raan_deg")
    omega_deg = kwargs.get("base_omega_deg")

    tag = f"i={i_deg}_raan={raan_deg}_omega={omega_deg}"
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
    for i_deg, raan_deg, omega_deg in itertools.product(BASE_I_DEG_VALUES, BASE_RAAN_DEG_VALUES, BASE_OMEGA_DEG_VALUES):
        kw = dict(FIXED_KWARGS)
        kw["base_i_deg"]     = float(i_deg)
        kw["base_raan_deg"]  = float(raan_deg)
        kw["base_omega_deg"] = float(omega_deg)
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
        print(f"  Orbital Parameter Sweep: {n_sims} sims, {n_workers} parallel workers")
        print(f"  base_i_deg:     {BASE_I_DEG_VALUES}")
        print(f"  base_raan_deg:  {BASE_RAAN_DEG_VALUES}")
        print(f"  base_omega_deg: {BASE_OMEGA_DEG_VALUES}")
        print("=" * 60)

        with multiprocessing.Pool(processes=n_workers) as pool:
            results = pool.map(_worker, grid)

        print("\n" + "=" * 60)
        print("  Results summary:")
        for r in results:
            print(f"    {r}")
        print("=" * 60)
