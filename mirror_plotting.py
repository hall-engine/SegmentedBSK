import matplotlib.pyplot as plt
import h5py
import numpy as np
import os
import re
import sys
import shutil
import utilities as u
import matplotlib.patches as patches
import poppy
import astropy.units as units
from tqdm import tqdm
from PIL import Image
from State import State

def run(cfg, read_every, opd_vmax=0.005, debug=False):
    # Make sure the fastersimulation package is importable
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, HERE)

    # ── Path to results ────────────────────────────────────────────────────────────
    # folder_name = 'testing_faster/sim_15.03_300'
    # path = os.path.join(HERE, "results", folder_name, "mirror_states.h5")
    # graph_path = os.path.join(HERE, "results", folder_name)

    path = os.path.join(cfg.out_dir, "mirror_states.h5")
    graph_path = cfg.out_dir

    # ── Load from HDF5 ─────────────────────────────────────────────────────────────
    states   = []
    time     = None   # (T,)   simulation time [s]
    rel_pos  = None   # (T, 3) r_detector − r_aperture in ECI [m]
    rel_pos_B = None  # (T, 3) r_detector − r_aperture in aperture Body frame [m]
    cfg_dict = {}     # flat dict of all SimConfig fields

    with h5py.File(path, "r") as f:
        # ── Time ──────────────────────────────────────────────────────────────
        # Prefer mirror_time (engaged phase) if available, as mirror datasets 
        # are co-indexed with it. Fall back to global "time" otherwise.
        if "mirror_time" in f:
            time = f["mirror_time"][:]
            #print(f"Using mirror_time: {time.shape} ({time[0]:.1f}s → {time[-1]:.1f}s)")
        elif "time" in f:
            time = f["time"][:]
            #print(f"Using global time: {time.shape}")
        else:
            print("No time array in file.")

        # ── Relative position ──────────────────────────────────────────────────
        if "rel_pos_B" in f and time is not None:
            # rel_pos_B is now saved as engaged-phase-only, co-indexed with mirror_time
            rel_pos_B = f["rel_pos_B"][:]
            if len(rel_pos_B) != len(time):
                # If they still don't match, we might be reading full-sim time against engaged-phase data.
                # Attempt to slice rel_pos_B if it was accidentally saved as full-sim.
                if len(rel_pos_B) > len(time):
                    # Guessing alignment... this is risky but better than crashing
                    rel_pos_B = rel_pos_B[-len(time):]
                else:
                    print(f"[WARNING] rel_pos_B length {len(rel_pos_B)} != time length {len(time)}.")
        
        if rel_pos_B is None and "rel_pos" in f and time is not None:
            # Fallback to ECI rel_pos if body frame is missing
            raw_target = f["rel_pos"]
            if len(raw_target) == len(time):
                rel_pos = raw_target[:]
            else:
                # Need to slice ECI to match mirror time span
                # This assumes 'time' is a sub-segment of the simulation
                dt = time[1] - time[0] if len(time) > 1 else 0.1
                start_idx = int(np.round(time[0] / dt))
                end_idx = start_idx + len(time)
                rel_pos = raw_target[start_idx:end_idx]

        # ── Config ────────────────────────────────────────────────────────────
        if "config" in f:
            cfg_grp = f["config"]
            # Scalar fields stored as attrs
            for k, v in cfg_grp.attrs.items():
                cfg_dict[k] = v
            # Array fields stored as datasets
            for k in cfg_grp:
                cfg_dict[k] = cfg_grp[k][:]
            print(f">> config loaded")

        # ── Segment states ────────────────────────────────────────────────────
        for seg_name in sorted(f.keys()):
            if not isinstance(f[seg_name], h5py.Group):
                continue                          # skip time, rel_pos datasets
            if seg_name == "config":
                continue
            grp = f[seg_name]
            seg_num = int(grp.attrs["segment_number"])

            if grp["position"].shape[0] == 0:
                if debug:
                    print(f"  [{seg_name}]  empty — skipping")
                continue
            # Position is constant (1, 6) — use the single row
            pos_data = grp["position"][:]
            s = State(
                number=seg_num,
                position=pos_data[-1].tolist(),
                mirror_actuation=grp["mirror_actuation"][-1].tolist(),
                desired_mirror_actuation=grp["desired_mirror_actuation"][-1].tolist(),
            )
            s.hist_position                 = pos_data  # (1, 6) constant
            s.hist_mirror_actuation         = grp["mirror_actuation"][:]
            s.hist_desired_mirror_actuation = grp["desired_mirror_actuation"][:]
            s.hist_point_on_det_plane       = grp["point_on_det_plane"][:]

            states.append(s)
            if debug:
                print(f"  [{seg_name}]  T={s.hist_mirror_actuation.shape[0]} ticks")

    print(f">> pulled {len(states)} State objects")

    # Guard: nothing to plot if mirror data was never written
    if time is None or len(time) == 0:
        print(">> mirror_plotting: HDF5 has no time data — skipping all mirror plots.")
        return

    # Extract required parameters for create_frames
    rings = cfg.rings
    flattoflat = cfg.flat_to_flat
    gap = cfg.gap
    det_size = cfg.det_size
    wideview = cfg.wideview
    
    # create HEX shape
    hexdm = poppy.dms.HexSegmentedDeformableMirror(rings=rings,
                                                   flattoflat=flattoflat*units.m, 
                                                   gap=gap*units.cm, 
                                                   include_factor_of_two=True)

    # -- PLOTS --
    # ==================================================================================================================
    if time is not None and len(time) > 0:
        suffix = "_" + u.sim_tag(cfg) if hasattr(u, 'sim_tag') else ""
        plots_mirror_actuation(states, plot_state_list=[0, 1, 3, 6], graph_path=graph_path, time=time, cfg=cfg, suffix=suffix)
        # Use body-frame relative position (x_B, y_B = focal plane coords); fall back to ECI if unavailable
        focal_plane_pos = rel_pos_B if rel_pos_B is not None else rel_pos
        create_frames(hexdm, states, time, graph_path, focal_plane_pos, det_size, wideview, opd_vmax=opd_vmax, read_every=read_every, if_opd=False, suffix=suffix)
    else:
        print(">> Skipping mirror plots and frames (no data).")
    # ==================================================================================================================


def plots_mirror_actuation(states, plot_state_list, graph_path, time=None, cfg=None, suffix=""):
    """
    One figure per (actuation / residuals), with two side-by-side columns:
      LEFT  — calibration period, rad / m  scale
      RIGHT — observation period, µrad / µm scale  (zoomed in)
    Width of each panel is proportional to its duration.
    """
    dt       = time[1] - time[0] if len(time) > 1 else 0.1
    cal_sec  = getattr(cfg, 'cal_window_sec', 900.0) if cfg else 900.0
    n_cal    = min(int(round(cal_sec / dt)), len(time) - 1)
    t_cal    = time[:n_cal]
    t_obs    = time[n_cal:]
    has_obs  = len(t_obs) > 1
    row_labels = ["Tip", "Tilt", "Piston"]
    for residual, fname, title in [
        (False, f"mirror_actuation{suffix}.png",  "Mirror Actuation"),
        (True,  f"mirror_residuals{suffix}.png",  "Mirror Residuals"),
    ]:
        ncols        = 2 if has_obs else 1
        width_ratios = [1, 1] if has_obs else [1]
        fig, axes = plt.subplots(
            3, ncols, figsize=(14 if has_obs else 10, 7),
            gridspec_kw={'width_ratios': width_ratios},
            sharey='none'
        )
        fig.suptitle(title, fontsize=13)
        # normalise axes to always be (3, ncols)
        if ncols == 1:
            axes = axes[:, np.newaxis]
        for row in range(3):
            ax_cal = axes[row, 0]
            cal_unit    = "rad" if row < 2 else "m"
            obs_unit    = "µrad" if row < 2 else "µm"
            obs_scale   = 1e6
            # --- calibration column ---
            for s in plot_state_list:
                des = states[s].hist_desired_mirror_actuation[:n_cal, row]
                act = states[s].hist_mirror_actuation[:n_cal, row]
                lbl = states[s].number
                if residual:
                    ax_cal.plot(t_cal, act - des, label=str(lbl))
                else:
                    ax_cal.plot(t_cal, des, ls='--', label=f"des {lbl}")
                    ax_cal.plot(t_cal, act,         label=str(lbl))
            ax_cal.set_ylabel(f"{row_labels[row]} [{cal_unit}]")
            if row == 0:
                ax_cal.set_title("Calibration", fontsize=10)
            if row == 2:
                ax_cal.set_xlabel("Time [s]")
            ax_cal.grid(); u.set_dark_transparent(ax=ax_cal)
            # --- observation column ---
            if has_obs:
                ax_obs = axes[row, 1]
                for s in plot_state_list:
                    des = states[s].hist_desired_mirror_actuation[n_cal:, row]
                    act = states[s].hist_mirror_actuation[n_cal:, row]
                    lbl = states[s].number
                    if residual:
                        ax_obs.plot(t_obs, (act - des) * obs_scale, label=str(lbl))
                    else:
                        ax_obs.plot(t_obs, des * obs_scale, ls='--', label=f"des {lbl}")
                        ax_obs.plot(t_obs, act * obs_scale,         label=str(lbl))
                ax_obs.set_ylabel(f"{row_labels[row]} [{obs_unit}]")
                if row == 0:
                    ax_obs.set_title("Fine Observation", fontsize=10)
                if row == 2:
                    ax_obs.set_xlabel("Time [s]")
                ax_obs.grid(); u.set_dark_transparent(ax=ax_obs)
        plt.tight_layout()
        plt.savefig(f"{graph_path}/{fname}")
        plt.close()
        print(f">> {fname} saved")


def _rectange_for_detectorax(ax, pos, size, color):
    # pos contains [x, y], size is the square length
    rect = patches.Rectangle((pos[0] - size/2, pos[1] - size/2), size, size, 
                             linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(rect)


def create_frames(hexdm, states, time, graph_path, position, det_size, wideview, read_every, opd_vmax=0.005, if_opd=False, suffix=""):
    """
    Plots the frame
    LHS has the HEXDM shape colour
    RHS has where they point
    """
    # CREATE GIF_TEMP FOLDER
    gif_temp_dir = os.path.join(graph_path, 'gif_temp')
    if os.path.exists(gif_temp_dir):
        # CLEAR AND WIPE PREVIOUS RUNS
        shutil.rmtree(gif_temp_dir)
    os.makedirs(gif_temp_dir)
    # wideview limits
    Xlimits = [-wideview, wideview]
    Ylimits = [-wideview, wideview]
    # detector limits
    detXlimits = [-det_size/2 , det_size/2]
    detYlimits = [-det_size/2 , det_size/2]
    # for all the values: i = integer index, time_frame = float time value [s]
    indices = range(0, len(time), read_every)
    for i in tqdm(indices, desc='processing frames: ', total=len(indices)):
        time_frame = time[i]
        # Check if the index is cleanly divisible by read_every
        if if_opd: 
            fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(24,6))  # Adjust height for better aspect ratio
        else:   
            fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(19,6))  # Adjust height for better aspect ratio
        # actuate mirror surface
        for s_idx, s_key in enumerate(states):
            state_obj = states[s_key] if isinstance(states, dict) else s_key
            [tip, tilt, piston] = state_obj.hist_mirror_actuation[i, :3]
            s_label = getattr(state_obj, 'number', s_idx)
            hexdm.set_actuator(s_label, piston, -tip, -tilt)
        # HEXDM
        hexdm.display(what='opd', colorbar_orientation="vertical", opd_vmax=opd_vmax, ax=ax[0])
        
        # Define a colormap for segments
        cmap = plt.get_cmap('tab10')
        # DETECTOR
        for s_idx, s_key in enumerate(states):
            # Handle states whether it is a list or dict
            state_obj = states[s_key] if isinstance(states, dict) else s_key
            # Check attribute name
            pt = state_obj.hist_point_on_det_plane[i]
            s_label = getattr(state_obj, 'number', s_idx)
            color = cmap(s_idx % 10)
            
            # wideview
            ax[1].scatter(pt[0], pt[1], marker='.', s=5, color=color)
            
            # detector plane
            ax[2].scatter(pt[0]-position[i, 0], 
                        pt[1]-position[i, 1], 
                        s=100, 
                        marker='+',
                        color=color,
                        alpha=0.8,
                        label=f'seg {s_label}')
                        
        # set axis properties once
        ax[1].set_ylim(Ylimits)
        ax[1].set_xlim(Xlimits)
        ax[1].set_title("broad focal plane")
        ax[1].grid(False)
        ax[1].axhline(0, color='white', linestyle=':', linewidth=2, alpha=0.5)
        ax[1].axvline(0, color='white', linestyle=':', linewidth=2, alpha=0.5)
        
        ax[2].set_ylim(detYlimits)
        ax[2].set_xlim(detXlimits)
        ax[2].set_title("detector plane")
        if len(states) < 10:
            ax[2].legend(loc='upper right', fontsize='xx-small')
            
        # OPD JUST IN CASE
        if if_opd:
            # We don't have detector.current_opd easily available without the object, 
            # so we plot the OPD from hexdm directly or skip if not passed
            im = hexdm.display(what='opd', colorbar_orientation="vertical", opd_vmax=2.0, ax=ax[3])
            ax[3].set_title('OPD [m]')
            
        # optical axis centre crosshair
        ax[1].scatter(position[i, 0], position[i, 1], marker='+', s=50, color='white')
        # detector boundary
        _rectange_for_detectorax(ax[1], position[i, 0:2], det_size*25*(1-position[i, 2]), color='k')
        plt.suptitle(f"time step {time_frame:5.2f} s")
        plt.tight_layout()
        plotname = f'frame_{time_frame:.4f}.png'
        white_path = os.path.join(gif_temp_dir, f'white_{plotname}')
        black_path = os.path.join(gif_temp_dir, f'black_{plotname}')
        # save white
        plt.savefig(white_path)
        for a in ax:
            u.set_dark_transparent(ax=a)
        _rectange_for_detectorax(ax[1], position[i, 0:2], det_size*25, color='white')
        # save black
        plt.savefig(black_path)
        plt.close('all')
    
    # Creates two GIFs (white + black) from PNG frames inside gif_temp_dir.
    # After creation, deletes all frame PNGs.

    print(f'>> creating GIFs from frames...')
    white_name=f"white_animation{suffix}.gif"
    black_name=f"black_animation{suffix}.gif"
    duration=20
    pattern_white = re.compile(r"white_frame_([\d.]+)\.png")
    pattern_black = re.compile(r"black_frame_([\d.]+)\.png")

    frames_white = []
    frames_black = []

    for filename in os.listdir(gif_temp_dir):
        m_w = pattern_white.match(filename)
        m_b = pattern_black.match(filename)
        if m_w:
            frames_white.append((float(m_w.group(1)), filename))
        elif m_b:
            frames_black.append((float(m_b.group(1)), filename))

    frames_white.sort(key=lambda x: float(x[0]))
    frames_black.sort(key=lambda x: float(x[0]))

    def save_gif(frames, output_name):
        if not frames:
            print(f">> no frames found for {output_name}")
            return
        imgs = []
        for _, fname in frames:
            path = os.path.join(gif_temp_dir, fname)
            with Image.open(path) as im:
                imgs.append(im.convert("RGBA").copy())
        paletted = [im.convert("P", palette=Image.ADAPTIVE) for im in imgs]
        output_path = os.path.join(graph_path, output_name)
        paletted[0].save(
            output_path,
            save_all=True,
            append_images=paletted[1:],
            duration=duration,
            loop=0,
            disposal=2,
            transparency=0
        )
        print(">> GIF created.")
        del imgs, paletted

    save_gif(frames_white, white_name)
    save_gif(frames_black, black_name)

    # Clean up frame PNGs
    to_delete = [f for _, f in frames_white] + [f for _, f in frames_black]
    for f in to_delete:
        os.remove(os.path.join(gif_temp_dir, f))
    print(f">> deleted {len(to_delete)} frame files.")


if __name__ == "__main__":
    from config import SimConfig
    cfg = SimConfig()
    # run(cfg)
    # or specify out_dir here if manually re-plotting:
    # cfg.out_dir = "./results/testing_faster/sim_15.03_300.0"
    run(cfg)
