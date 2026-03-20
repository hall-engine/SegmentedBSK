"""
monte_carlo.py — Parallel Monte Carlo runner for the formation simulation
=========================================================================
Replaces the serial triple-for-loop at the bottom of main.py.

Run with:
    python monte_carlo.py

Edit the PARAMETER GRID section below to define your sweep.
Edit N_WORKERS to control parallelism (one BSK sim per worker process).

Notes
-----
- Uses `multiprocessing` with the `spawn` context (macOS default).
  Each worker is a completely independent Python interpreter — no shared
  Basilisk state, no thread-safety issues.
- Workers call main.run(**kwargs) exactly as main.py's __main__ block does.
  Mirror plotting, plot saving, etc. are all handled inside run().
- If a worker crashes it prints the traceback and the sweep continues.
- Results go to the same results_base as a serial run (from SimConfig).
- On HPC: set --cpus-per-task in your SLURM script; N_WORKERS auto-reads
  SLURM_CPUS_PER_TASK.  For multi-node HPC use slurm_array.sh instead.
"""

import itertools        # standard library: makes all combinations of lists (like nested for-loops)
import multiprocessing  # standard library: runs Python functions in separate processes simultaneously
import os               # standard library: access to environment variables and CPU count
import traceback        # standard library: captures the full error message if a worker crashes

import matplotlib
matplotlib.use("Agg")   # tell matplotlib NOT to open any windows — required when running in background processes
                        # (each worker process has no display, so "Agg" writes to file only)


# ─── PARAMETER GRID ───────────────────────────────────────────────────────────
# Define the values you want to sweep.
# Every combination of (i, raan, omega) becomes one separate simulation run.
# e.g. 1 i × 1 raan × 8 omega = 8 total simulations.

BASE_I_DEG_VALUES     = [0, 45, 90, 135, 180]  # inclination values to try [deg]
BASE_RAAN_DEG_VALUES  = [0]                       # RAAN values to try [deg]
BASE_OMEGA_DEG_VALUES = [0, 45, 90, 135, 180]     # argument of perigee values to try [deg]
A_GEO_VALUES          = [40e6, 50e6, 60e6]        # semi-major axis values to try [m]
E_GEO_VALUES          = [0.2, 0.4, 0.6, 0.8]     # eccentricity values to try

# These kwargs are passed to main.run() for EVERY simulation (the fixed settings).
FIXED_KWARGS = dict(
    time_step_sec    = 0.5,    # sim timestep [s]
    read_every       = 100,    # mirror plotting frame interval
    show_plots       = True,   # save all plots after each sim
    mirror_plot      = False,  # run mirror animation after each sim
    disable_progress = True,   # suppress tqdm in workers — bars scroll chaos in parallel
)

# Number of sims to run simultaneously.
# Think of this like how many parallel lanes on a highway.
# On HPC: reads SLURM_CPUS_PER_TASK so we don't over-allocate.
# Locally: uses all cores minus 2 so your machine stays usable.
# Each sim process uses ~1-2 GB of RAM, so don't set this too high.
_slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")            # will be None if not on SLURM
N_WORKERS = int(_slurm_cpus) if _slurm_cpus else max(1, os.cpu_count() - 2)

# =============================================================================


def _worker(kwargs: dict) -> str:
    """
    This function runs ONE simulation. It gets called inside a worker process.

    KEY CONCEPT: each call to _worker() happens in a completely separate Python
    interpreter — not just a separate thread, but a whole new process with its
    own memory.  That means Basilisk's C++ global state is never shared between
    sims, so they can't interfere with each other.

    Returns a short status string ("✓ ..." or "✗ ...") for the summary printout.
    """

    # Imports are done INSIDE the worker, not at the top of the file.
    # This is important: when a new process is spawned (created), it starts
    # fresh and re-imports everything it needs.  Doing it here ensures each
    # worker gets its own clean copy of main, config, matplotlib, etc.
    import matplotlib.pyplot as plt
    from config import SimConfig
    import main as simulation   # the actual simulation — we just call simulation.run()

    # Build a short label for this combo, used in logging and the final summary
    tag = (f"i={kwargs.get('base_i_deg')}° "
           f"raan={kwargs.get('base_raan_deg')}° "
           f"omega={kwargs.get('base_omega_deg')}° "
           f"a={kwargs.get('a_geo', 0)/1e6:.0f}Mm "
           f"e={kwargs.get('e_geo')}")
    print(f"[START] {tag}", flush=True)
    try:
        cfg = SimConfig()               # fresh config object for this sim
        simulation.run(cfg, **kwargs)   # run the sim + plots + mirror animation
        plt.close("all")               # release matplotlib memory before this process exits
        print(f"[DONE]  {tag}", flush=True)
        return f"✓  {tag}"             # report success back to the main process
    except Exception:
        # If anything goes wrong, print the full error but DON'T crash the whole sweep.
        # The Pool will collect this "✗" result and the other sims continue normally.
        print(f"\n[monte_carlo] WORKER FAILED — {tag}\n{traceback.format_exc()}",
              flush=True)              # flush=True forces the error to print immediately
        return f"✗  {tag}  (FAILED — see stderr)"


def build_param_grid() -> list[dict]:
    """
    Expand the parameter lists into a flat list of kwarg dicts.

    itertools.product() is the equivalent of nested for-loops:
        for i in BASE_I_DEG_VALUES:
            for r in BASE_RAAN_DEG_VALUES:
                for o in BASE_OMEGA_DEG_VALUES:
                    ...
    but returns all combinations at once as a list of tuples.
    """
    grid = []
    for i_deg, raan_deg, omega_deg, a_geo, e_geo in itertools.product(
            BASE_I_DEG_VALUES, BASE_RAAN_DEG_VALUES, BASE_OMEGA_DEG_VALUES, A_GEO_VALUES, E_GEO_VALUES):
        kw = dict(FIXED_KWARGS)         # copy the fixed kwargs so we don't mutate the original
        kw["base_i_deg"]     = i_deg    # add this combo's variable parameters
        kw["base_raan_deg"]  = raan_deg
        kw["base_omega_deg"] = omega_deg
        kw["a_geo"]          = a_geo
        kw["e_geo"]          = e_geo
        grid.append(kw)                 # one dict per sim
    return grid


if __name__ == "__main__":
    # This block only runs when you execute `python monte_carlo.py` directly.
    # It does NOT run if another script imports this file.

    # "spawn" means: when creating a worker process, start a brand-new Python
    # interpreter from scratch.  The alternative is "fork" (copy the current
    # process), which can cause crashes with C++ extensions like Basilisk.
    # macOS defaults to spawn already, but we set it explicitly to be safe.
    multiprocessing.set_start_method("spawn", force=True)

    grid      = build_param_grid()          # list of kwarg dicts, one per sim
    n_sims    = len(grid)
    n_workers = min(N_WORKERS, n_sims)      # no point having more workers than sims

    print("=" * 70)
    print(f"  Monte Carlo sweep: {n_sims} sims, {n_workers} parallel workers")
    print("=" * 70)
    for kw in grid:
        print(f"  · i={kw['base_i_deg']:>5}°  raan={kw['base_raan_deg']:>5}°"
              f"  omega={kw['base_omega_deg']:>5}°  a={kw['a_geo']/1e6:.0f} Mm  e={kw['e_geo']:.1f}")
    print("=" * 70)

    # Pool manages a group of worker processes.
    # Think of it as a job queue: you give it a list of tasks (grid),
    # and it keeps N_WORKERS processes busy until all tasks are done.
    # "with" ensures the pool is properly shut down even if something crashes.
    with multiprocessing.Pool(processes=n_workers) as pool:

        # pool.map() is like: [_worker(g) for g in grid]
        # but runs up to n_workers calls simultaneously in separate processes.
        # It blocks here until ALL sims finish, then returns the list of results.
        results = pool.map(_worker, grid)

    # Print a final summary of which sims succeeded and which failed
    print("\n" + "=" * 70)
    print("  Results summary:")
    for r in results:
        print(f"    {r}")
    print("=" * 70)
