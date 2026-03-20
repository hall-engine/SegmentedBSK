import os
import json
import dataclasses
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import shutil
import argparse
import yaml
from types import SimpleNamespace
import time
import random   

def sim_tag(cfg) -> str:
    """
    Single source of truth for the sim parameter string.
    Used as both the results sub-folder name and the plot filename suffix.
    e.g. 'sim_a45000000.0_e0.8_i-8_raan70_omega20'
    """
    return (f"sim_a{cfg.a_geo:.2e}_e{cfg.e_geo}"
            f"_i{cfg.base_i_deg}_raan{cfg.base_raan_deg}_omega{cfg.base_omega_deg}")


def create_sim_dir(cfg, results_base="./results"):
    """
    Creates a new sub-folder inside results_base named after the sim parameters.
    Returns the absolute path to the newly created directory.
    """
    folder_name = sim_tag(cfg)
    target_path = os.path.join(results_base, folder_name)
    
    # Ensure the results base exists
    if not os.path.exists(results_base):
        os.makedirs(results_base)
        
    # Create the simulation-specific directory
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        print(f"Created results: .../{os.path.relpath(target_path)}")
    else:
        print(f"Results dir already exists: .../{os.path.relpath(target_path)}")
        
    return target_path

def save_sim_config(cfg, target_path: str):
    """
    Saves the SimConfig parameters to a JSON file in the target directory.
    """
    config_dict = dataclasses.asdict(cfg)
    file_path = os.path.join(target_path, "sim_config.json")
    with open(file_path, "w") as f:
        json.dump(config_dict, f, indent=4)
    print(f"Saved sim_config.json to .../{os.path.relpath(target_path)}")


def save_states_h5(states: dict, target_path: str, filename: str = "mirror_states.h5",
                   time_arr: np.ndarray = None,
                   cfg=None,
                   rel_pos_arr: np.ndarray = None,
                   rel_pos_B_arr: np.ndarray = None):
    """
    Save mirror segment state histories to an HDF5 file.

    File layout
    -----------
    mirror_states.h5
    ├── time                          (T,)    simulation time [s]
    ├── rel_pos                       (T, 3)  detector pos − aperture pos [m]
    ├── rel_pos_B                     (T, 3)  detector pos − aperture pos in Body frame [m]
    ├── config/                       group — SimConfig fields as attrs / datasets
    └── segment_<key>/
        ├── position                  (T, 6)  [x, y, z, θx, θy, θz]
        ├── mirror_actuation          (T, 6)  [tip, tilt, piston, ṫip, ṫilt, ṗiston]
        ├── desired_mirror_actuation  (T, 6)  same layout, commanded values
        └── point_on_det_plane        (T, 2)  [dX, dY]

    Parameters
    ----------
    states      : dict mapping segment key → State object
    target_path : directory to write the file into
    filename    : HDF5 filename (default: "mirror_states.h5")
    time_arr    : 1-D array of simulation timestamps [s]
    cfg         : SimConfig dataclass instance (optional)
    rel_pos_arr : (T, 3) array of r_detector − r_aperture in ECI [m] (optional)
    rel_pos_B_arr : (T, 3) array of r_detector − r_aperture in Body frame [m] (optional)
    """
    import dataclasses

    file_path = os.path.join(target_path, filename)
    with h5py.File(file_path, "w") as f:
        f.attrs["n_segments"] = len(states)

        # ── Time ──────────────────────────────────────────────────────────────
        if time_arr is not None:
            f.create_dataset("time", data=np.array(time_arr), compression="gzip")

        # ── Relative position (r_det - r_app) ─────────────────────────────────
        if rel_pos_arr is not None:
            f.create_dataset("rel_pos", data=np.array(rel_pos_arr), compression="gzip")
        if rel_pos_B_arr is not None:
            f.create_dataset("rel_pos_B", data=np.array(rel_pos_B_arr), compression="gzip")

        # ── Config ────────────────────────────────────────────────────────────
        if cfg is not None:
            cfg_grp = f.create_group("config")
            for field in dataclasses.fields(cfg):
                val = getattr(cfg, field.name)
                if isinstance(val, (int, float, bool, str)):
                    cfg_grp.attrs[field.name] = val
                elif isinstance(val, (list, np.ndarray)):
                    cfg_grp.create_dataset(field.name, data=np.array(val))
                # skip non-serialisable types (e.g. numpy matrix)

        # ── Segment histories ─────────────────────────────────────────────────
        for key, state in states.items():
            grp = f.create_group(f"segment_{key}")
            grp.attrs["segment_number"] = int(state.number)

            grp.create_dataset("position",
                               data=np.array(state.hist_position),
                               compression="gzip")
            grp.create_dataset("mirror_actuation",
                               data=np.array(state.hist_mirror_actuation),
                               compression="gzip")
            grp.create_dataset("desired_mirror_actuation",
                               data=np.array(state.hist_desired_mirror_actuation),
                               compression="gzip")
            grp.create_dataset("point_on_det_plane",
                               data=np.array(state.hist_point_on_det_plane),
                               compression="gzip")
    print(f"Saved mirror_states.h5 to .../{os.path.relpath(target_path)}")


def set_dark_transparent(ax=None, grid_alpha=0.1):
    """
    Makes plot background transparent and all text/grid/axes/colorbars white.
    Works with both plt and ax-based plotting.
    """
    if ax is None:
        ax = plt.gca()
    fig = ax.get_figure()

    # Transparent background
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    
    # --- 3D axes special handling ---
    if isinstance(ax, Axes3D):
        # Turn OFF the pane completely → avoids black fallback
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
    
        # No pane edges (keeps background fully invisible)
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
    
        # White axis lines
        ax.xaxis.line.set_color('white')
        ax.yaxis.line.set_color('white')
        ax.zaxis.line.set_color('white')
    
        # White ticks
        ax.xaxis.set_tick_params(color='white')
        ax.yaxis.set_tick_params(color='white')
        ax.zaxis.set_tick_params(color='white')
    
        # White tick labels
        for t in ax.xaxis.get_ticklabels():
            t.set_color('white')
        for t in ax.yaxis.get_ticklabels():
            t.set_color('white')
        for t in ax.zaxis.get_ticklabels():
            t.set_color('white')


    # White text and ticks
    ax.title.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(colors='white')

    # White spines
    for spine in ax.spines.values():
        spine.set_color('white')
        
    # white SUPtitle if present.     
    if fig._suptitle is not None:
        fig._suptitle.set_color('white')

    # White legend (if present)
    legend = ax.get_legend()
    if legend:
        plt.setp(legend.get_texts(), color='white')
        legend.get_frame().set_alpha(0)

    # Only recolor grid if grid is already on
    if ax._gridOn:
        ax.grid(True, color='white', alpha=grid_alpha)
        

    # Make saved figure transparent
    plt.rcParams['savefig.transparent'] = True

    # --- Handle colorbars if present ---
    # Covers both imshow-based and manually created colorbars
    for im in ax.get_images():
        if hasattr(im, 'colorbar') and im.colorbar:
            cbar = im.colorbar
            _set_white_colorbar(cbar)

    # Also catch colorbars manually attached via fig.colorbar()
    for cbar in fig.axes:
        # Heuristic: colorbars are thin axes
        if cbar not in fig.axes or cbar == ax:
            continue
        pos = cbar.get_position()
        if pos.width < 0.1 or pos.height < 0.1:
            try:
                cbar.yaxis.label.set_color('white')
                cbar.tick_params(colors='white')
                plt.setp(cbar.get_yticklabels(), color='white')
                for spine in cbar.spines.values():
                    spine.set_color('white')
            except Exception:
                pass



def _set_white_colorbar(cbar):
    """Helper for setting all colorbar elements white."""
    cbar.ax.yaxis.label.set_color('white')           # label text
    cbar.ax.yaxis.set_tick_params(color='white')     # tick lines
    plt.setp(cbar.ax.get_yticklabels(), color='white')  # tick labels
    cbar.outline.set_edgecolor('white')
    cbar.ax.set_facecolor('none')     
