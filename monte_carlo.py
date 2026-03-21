"""
monte_carlo.py — Random-seed sweep for the formation simulation
===============================================================
Runs the same orbital configuration (from config.py defaults) N times, each
with a different random_seed, to quantify stochastic variability from CSS
noise, initial mirror actuation, etc.

Run with:
    python monte_carlo.py

Edit RANDOM_SEED_VALUES below to control which seeds are swept.
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

import matplotlib
matplotlib.use("Agg")   # no display in worker processes


# ─── SEED LIST ────────────────────────────────────────────────────────────────
# Each value becomes one independent simulation run using the orbital defaults
# in config.py.  Add/remove seeds as needed.

RANDOM_SEED_VALUES = list(range(1, 30))   # seeds 1 … 20  (20 replicates)

# These kwargs are passed to main.run() for EVERY simulation (fixed settings).
FIXED_KWARGS = dict(
    time_step_sec    = 0.5,    # sim timestep [s]
    read_every       = 100,    # mirror plotting frame interval
    show_plots       = True,   # save all plots after each sim
    mirror_plotting  = False,  # run mirror animation (slow — keep False for sweeps)
    disable_progress = True,   # suppress tqdm in workers
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

    seed = kwargs.get("random_seed")
    tag  = f"seed={seed}"
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
    """Return one kwarg dict per seed."""
    grid = []
    for seed in RANDOM_SEED_VALUES:
        kw = dict(FIXED_KWARGS)
        kw["random_seed"] = seed
        grid.append(kw)
    return grid


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    grid      = build_param_grid()
    n_sims    = len(grid)
    n_workers = min(N_WORKERS, n_sims)

    print("=" * 60)
    print(f"  Seed sweep: {n_sims} sims, {n_workers} parallel workers")
    print(f"  Seeds: {RANDOM_SEED_VALUES}")
    print("=" * 60)

    with multiprocessing.Pool(processes=n_workers) as pool:
        results = pool.map(_worker, grid)

    print("\n" + "=" * 60)
    print("  Results summary:")
    for r in results:
        print(f"    {r}")
    print("=" * 60)
