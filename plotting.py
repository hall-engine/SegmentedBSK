"""
plotting.py — All Simulation Plots
=====================================
All plot functions for the formation simulation. Call ``run_all()`` to
generate the full suite from a completed simulation run.

Outputs are saved to ./results/ (created automatically).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib as mpl




# ─── Directory helper ─────────────────────────────────────────────────────────
def _mkresults(out_dir="./results"):
    os.makedirs(out_dir, exist_ok=True)


# ─── Frame helpers ────────────────────────────────────────────────────────────

def _plane_fixed_frame(log1, log2, time):
    """Compute relative motion in a non-rotating orbital-plane frame."""
    r1, v1 = log1.r_BN_N, log1.v_BN_N
    r2      = log2.r_BN_N

    h0    = np.cross(r1[0], v1[0])
    z_hat = h0 / np.linalg.norm(h0)
    ix    = np.array([1., 0., 0.])
    x_hat = ix - np.dot(ix, z_hat) * z_hat
    if np.linalg.norm(x_hat) < 1e-8:
        ix    = np.array([0., 1., 0.])
        x_hat = ix - np.dot(ix, z_hat) * z_hat
    x_hat /= np.linalg.norm(x_hat)
    y_hat  = np.cross(z_hat, x_hat)

    C   = np.vstack((x_hat, y_hat, z_hat)).T
    # Vectorized relative position and velocity (much faster than per-step calls)
    r1_all = log1.r_BN_N
    v1_all = log1.v_BN_N
    r2_all = log2.r_BN_N
    v2_all = log2.v_BN_N
    
    rel = (r2_all - r1_all) @ C
    vel = (v2_all - v1_all) @ C
    return rel, vel, time


def _star_frame_dcm(star_vector=None, frame_dcm=None):
    """
    Return the 3×3 DCM whose rows are (x_s, y_s, z_s) of the aperture frame.

    Preferred: pass ``frame_dcm`` (3×3 ndarray, rows = x,y,z basis vectors).
    This is the frozen ECI frame computed at t=0 from:
        z = star_vector   (orbit normal)
        y = v_perigee     (aperture velocity at t=0)
        x = y × z         (radial outward at perigee)

    Fallback: pass ``star_vector`` only → z-axis is correct but x,y are
    chosen arbitrarily (original behaviour, breaks when RAAN/ω ≠ 0).
    """
    if frame_dcm is not None:
        return np.asarray(frame_dcm)
    # Fallback (kept for backwards compatibility with standalone calls)
    z_s = np.array(star_vector) / np.linalg.norm(star_vector)
    ref = np.array([0., 0., 1.])
    if abs(np.dot(ref, z_s)) > 0.99:
        ref = np.array([1., 0., 0.])
    x_s = np.cross(ref, z_s); x_s /= np.linalg.norm(x_s)
    y_s = np.cross(z_s, x_s)
    return np.vstack((x_s, y_s, z_s))


# --- Asthetic Helper ---------------------------------------------------------

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
    """Internal helper to recolor colorbar elements to white."""
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(colors='white')
    plt.setp(cbar.ax.get_yticklabels(), color='white')
    for spine in cbar.ax.spines.values():
        spine.set_color('white')


# =============================================================================
# INDIVIDUAL PLOT FUNCTIONS
# =============================================================================

def plot_relative_motion(log1, log2, time, out_dir="./results", suffix="", frame_dcm=None):
    """
    Relative motion in the frozen aperture frame (4 figures).

    Parameters
    ----------
    frame_dcm : (3,3) ndarray or None
        Aperture frame DCM (rows = x,y,z basis vectors in ECI).  When supplied
        the plots use the same perifocal frame as all other simulation outputs.
        Falls back to _plane_fixed_frame (arbitrary x orientation) when None.
    """
    if frame_dcm is not None:
        dcm   = np.asarray(frame_dcm)
        r1_all, v1_all = log1.r_BN_N, log1.v_BN_N
        r2_all, v2_all = log2.r_BN_N, log2.v_BN_N
        rel = (r2_all - r1_all) @ dcm.T   # project ECI relative pos into aperture frame
        vel = (v2_all - v1_all) @ dcm.T
        t   = time
    else:
        rel, vel, t = _plane_fixed_frame(log1, log2, time)
    x, y, z = rel[:, 0] / 1e3, rel[:, 1] / 1e3, rel[:, 2] / 1e3

    fig = plt.figure(figsize=(10, 6))
    ax  = fig.add_subplot(projection="3d")
    ax.plot(x, y, z, label="Relative Trajectory")
    ax.set_xlabel("x [km]  (perigee dir.)"); ax.set_ylabel("y [km]  (v_perigee dir.)"); ax.set_zlabel("z [km]  (orbit normal)")
    ax.set_title("3D Relative Motion (Aperture Frame — frozen perifocal)")
    mr = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2
    mx, my, mz = (x.max()+x.min())/2, (y.max()+y.min())/2, (z.max()+z.min())/2
    ax.set_xlim(mx-mr, mx+mr); ax.set_ylim(my-mr, my+mr); ax.set_zlim(mz-mr, mz+mr)
    if hasattr(ax, "set_box_aspect"): ax.set_box_aspect((1, 1, 1))
    ax.legend(); 
    set_dark_transparent(ax); 
    plt.savefig(os.path.join(out_dir, f"relative_position_3d{suffix}.png"))

    plt.figure(figsize=(10, 6))
    plt.plot(t/3600, x, label="x (perigee dir.)"); plt.plot(t/3600, y, label="y (v-perigee dir.)")
    plt.plot(t/3600, z, label="z (orbit normal)")
    plt.xlabel("Time [hours]"); plt.ylabel("Rel. Position [km]")
    plt.legend(); plt.grid(); plt.title("Relative Position (Aperture Frame — frozen perifocal)")
    set_dark_transparent(); plt.savefig(os.path.join(out_dir, f"relative_position{suffix}.png"))

    # Plot 3: In-Plane (Heat Map)
    fig, ax = plt.subplots(figsize=(10, 6))
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(t.min()/3600, t.max()/3600))
    lc.set_array(t / 3600.0)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    ax.autoscale()
    ax.set_xlabel("x [km]  (perigee dir.)"); ax.set_ylabel("y [km]  (v-perigee dir.)")
    ax.set_title("In-Plane Relative Motion — x/y (Colored by Time)")
    ax.axis("equal"); ax.grid()
    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label("Time [hours]")
    set_dark_transparent(ax); plt.savefig(os.path.join(out_dir, f"relative_inplane{suffix}.png"))

    plt.figure(figsize=(10, 6))
    plt.plot(t/3600, vel[:, 0], label="vx (perigee dir.)")
    plt.plot(t/3600, vel[:, 1], label="vy (v-perigee dir.)")
    plt.plot(t/3600, vel[:, 2], label="vz (orbit normal)")
    plt.xlabel("Time [hours]"); plt.ylabel("Rel. Velocity [m/s]")
    plt.legend(); plt.grid(); plt.title("Relative Velocity (Aperture Frame — frozen perifocal)")
    set_dark_transparent(); plt.savefig(os.path.join(out_dir, f"relative_velocity{suffix}.png"))


def plot_orbital_trajectories(r1_n, r2_n, engaged, time, out_dir="./results", aperture_frame_dcm=None, suffix=""):
    """
    3D orbital trajectories in the aperture frame (x=radial, y=along-track, z=orbit-normal).
    Two-panel figure:
      Left  — both orbits projected into aperture frame (shows orbital scale)
      Right  — detector position RELATIVE to aperture in aperture frame (km-scale separation)
    """
    engaged = np.asarray(engaged)

    # Project into aperture frame if DCM provided, else use ECI
    if aperture_frame_dcm is not None:
        dcm = np.array(aperture_frame_dcm)          # (3,3)
        r1  = (dcm @ r1_n.T).T / 1e3               # km, aperture frame
        r2  = (dcm @ r2_n.T).T / 1e3
        xlabel, ylabel, zlabel = "x [km]  (Radial)", "y [km]  (Along-track)", "z [km]  (Orbit-normal)"
        frame_label = "Aperture Frame"
    else:
        r1, r2 = r1_n / 1e3, r2_n / 1e3
        xlabel, ylabel, zlabel = "X [km]", "Y [km]", "Z [km]"
        frame_label = "ECI"

    rel = r2 - r1     # detector relative to aperture [km]

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(111, projection="3d")
    ax1.plot(r1[:, 0], r1[:, 1], r1[:, 2], lw=2, color="orange",
             label="Aperture", alpha=0.9)
    ax1.plot(r2[:, 0], r2[:, 1], r2[:, 2], lw=1, color="cyan",
             label="Detector", alpha=0.7, linestyle="--")
    if np.any(engaged):
        diff   = np.diff(engaged.astype(int), prepend=0, append=0)
        starts = np.where(diff ==  1)[0]
        ends   = np.where(diff == -1)[0]
        for i, (s, e) in enumerate(zip(starts, ends)):
            ax1.plot(r2[s:e, 0], r2[s:e, 1], r2[s:e, 2],
                     color="red", lw=4, label="Control ON" if i == 0 else "")
    ax1.set_xlabel(xlabel); ax1.set_ylabel(ylabel); ax1.set_zlabel(zlabel)
    ax1.set_title(f"3D Orbital Trajectories ({frame_label})")
    # Equal x-y scale so the true ellipse shape is visible; z scales freely
    all_r  = np.vstack([r1, r2])
    xy_half = max(np.ptp(all_r[:, 0]), np.ptp(all_r[:, 1])) / 2
    cx = (all_r[:, 0].max() + all_r[:, 0].min()) / 2
    cy = (all_r[:, 1].max() + all_r[:, 1].min()) / 2
    ax1.set_xlim(cx - xy_half, cx + xy_half)
    ax1.set_ylim(cy - xy_half, cy + xy_half)
    z_half = np.ptp(all_r[:, 2]) / 2
    z_frac = max(0.1, z_half / xy_half) if xy_half > 0 else 1.0
    if hasattr(ax1, "set_box_aspect"): ax1.set_box_aspect((1, 1, z_frac))
    ax1.legend(loc="upper right", fontsize=9)
    set_dark_transparent(ax1)
    plt.savefig(os.path.join(out_dir, f"orbit_3d{suffix}.png"), dpi=120)


    # ── 2-D projections: separate file ────────────────────────────────────────
    fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
    pairs = [(0, 1, xlabel, ylabel, "xy"),
             (2, 0, zlabel, xlabel, "zx"),
             (1, 2, ylabel, zlabel, "yz")]
    for ax, (ci, cj, lx, ly, tag) in zip(axes, pairs):
        ax.plot(r1[:, ci], r1[:, cj], lw=1.5, color="orange", label="Aperture", alpha=0.7)
        ax.plot(r2[:, ci], r2[:, cj], lw=1,   color="cyan",   label="Detector", alpha=0.7, ls="--")
        if np.any(engaged):
            for i, (s, e) in enumerate(zip(starts, ends)):
                ax.plot(r2[s:e, ci], r2[s:e, cj], color="red",
                        lw=3, label="Control ON" if i == 0 else "")
        ax.set_xlabel(lx); ax.set_ylabel(ly)
        if tag == "xy":
            ax.set_aspect('equal')   # true ellipse shape on the main orbital plane
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        set_dark_transparent(ax)
    fig2.suptitle(f"Orbital Projections ({frame_label})", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"orbit_2d{suffix}.png"), dpi=120)



def plot_delta_v(dv, time, out_dir="./results", suffix=""):
    """Cumulative delta-V over full simulation."""
    plt.figure(figsize=(10, 6))
    plt.plot(time / 3600.0, dv)
    plt.xlabel("Time [hours]"); plt.ylabel("Cumulative ΔV [m/s]")
    plt.title("Detector Control Effort (Delta-V)"); plt.grid()
    set_dark_transparent() 
    plt.savefig(os.path.join(out_dir, f"delta_v{suffix}.png"))


def plot_detailed_thrust(dv_xyz, total_dv, time, out_dir="./results", suffix=""):
    """Per-axis and total delta-V in the engagement window."""
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    dv_s    = dv_xyz - dv_xyz[0]
    t       = time / 60.0

    ax[0].plot(t, dv_s[:, 0], "r--", label="ΔV_x (Radial)",      alpha=0.9)
    ax[0].plot(t, dv_s[:, 1], "g--", label="ΔV_y (Along-track)", alpha=0.9)
    ax[0].plot(t, dv_s[:, 2], "b--", label="ΔV_z (Focal)",       alpha=0.9)
    ax[0].set_ylabel("ΔV in Window [m/s]"); ax[0].legend(); ax[0].grid()

    ax[1].plot(t, total_dv, color="orange", linewidth=2, label="Total ΔV")
    ax[1].set_ylabel("Cumulative ΔV [m/s]"); ax[1].set_xlabel("Time [min]")
    ax[1].legend(); ax[1].grid()

    fig.suptitle("Detector Thrust Profile (Engagement Window)")
    for a in ax: set_dark_transparent(a)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"detailed_thrust{suffix}.png"))


def plot_delta_v_observation_window(dv_xyz, total_dv, time, phase, out_dir="./results", suffix=""):
    """
    Cumulative absolute ΔV per aperture-frame axis during the observation window.
    All three curves are monotonically increasing (fuel spent, not net displacement).
    """
    phase_arr = np.array(phase)
    obs_mask = (phase_arr == "Fine Observation")
    if not np.any(obs_mask):
        print("  Skipping observation Delta-V plot (no Observation phase data).")
        return

    dv_xyz_obs   = dv_xyz[obs_mask]
    total_dv_obs = total_dv[obs_mask]
    time_obs     = time[obs_mask]

    t_min = (time_obs - time_obs[0]) / 60.0
    dv_s  = dv_xyz_obs - dv_xyz_obs[0]   # cumulative abs ΔV within the window

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    ax[0].plot(t_min, dv_s[:, 0] * 1e3, "r-", lw=1.5, label="ΔV_x |Radial|")
    ax[0].plot(t_min, dv_s[:, 1] * 1e3, "g-", lw=1.5, label="ΔV_y |Along-track|")
    ax[0].plot(t_min, dv_s[:, 2] * 1e3, "b-", lw=1.5, label="ΔV_z |Focal|")
    ax[0].set_ylabel("Cumulative |ΔV| [mm/s]")
    ax[0].set_title("Per-Axis Cumulative Absolute ΔV (Observation Window)")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3); ax[0].set_ylim(bottom=0)

    ax[1].plot(t_min, total_dv_obs - total_dv_obs[0], color="orange", lw=2, label="Total |ΔV|")
    ax[1].set_ylabel("Cumulative ΔV [m/s]"); ax[1].set_xlabel("Time [min]")
    ax[1].set_title("Total Cumulative ΔV (Observation Window)")
    ax[1].legend(); ax[1].grid(alpha=0.3); ax[1].set_ylim(bottom=0)

    fig.suptitle("Detector ΔV Budget — Observation Window")
    set_dark_transparent(ax[0]); set_dark_transparent(ax[1])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"observation_dv{suffix}.png"))
    print(f"  Observation Delta-V plot saved to .../{os.path.relpath(out_dir)}.")



def plot_rw_speeds(rw_speeds, time, out_dir="./results", suffix=""):
    """Reaction wheel speeds in the engagement window."""
    # Basilisk rwSpeedOutMsg always has MAX_EFF_CNT=36 slots; only the
    # active wheels are nonzero — skip the empty (zero-padded) columns.
    active_cols = np.where(np.any(rw_speeds != 0, axis=0))[0]
    if len(active_cols) == 0:
        active_cols = np.arange(min(3, rw_speeds.shape[1]))  # fallback: first 3
    plt.figure(figsize=(10, 6))
    for k, i in enumerate(active_cols):
        plt.plot(time / 60.0, rw_speeds[:, i], label=f"RW {k+1}")
    plt.xlabel("Time [min]"); plt.ylabel("Speed [RPM]")
    plt.title("Reaction Wheel Speeds (Engagement Window)")
    plt.legend(); plt.grid()
    set_dark_transparent(); plt.savefig(os.path.join(out_dir, f"rw_speeds{suffix}.png"))


def plot_formation_error(pos_err_vec, time, star_vector, out_dir="./results", suffix="",
                         frame_dcm=None, phase=None):
    """Formation error in the aperture frame (pre-projected in main.py)."""
    from matplotlib.patches import Patch
    err_s = np.asarray(pos_err_vec)
    t     = time / 3600.0

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(t, err_s[:, 0], "r--", label="Lateral X", alpha=0.8)
    ax[0].plot(t, err_s[:, 1], "g--", label="Lateral Y", alpha=0.8)
    ax[0].set_ylabel("Error [m]"); ax[0].grid()

    ax[1].plot(t, err_s[:, 2], "b-", label="Focal Length Error")
    ax[1].set_xlabel("Time [hours]"); ax[1].set_ylabel("Error [m]")
    ax[1].grid()

    # Phase-shaded background regions
    phase_colors = {"Calibration": "orange", "Pre-Observation": "yellow", "Fine Observation": "red"}
    if phase is not None:
        ph = np.asarray(phase)
        for a in ax:
            for ph_name, color in phase_colors.items():
                mask = ph == ph_name
                if not np.any(mask): continue
                idx    = np.where(mask)[0]
                breaks = np.where(np.diff(idx) > 1)[0] + 1
                for run in np.split(idx, breaks):
                    a.axvspan(t[run[0]], t[run[-1]], color=color, alpha=0.10, lw=0)

    # Rebuild legends with phase patches
    legend_patches = [Patch(color=c, alpha=0.4, label=n) for n, c in phase_colors.items()
                      if phase is not None and np.any(np.asarray(phase) == n)]
    for a, extra in zip(ax, [legend_patches, legend_patches]):
        handles, labels = a.get_legend_handles_labels()
        a.legend(handles=handles + extra, fontsize=8)

    fig.suptitle("Formation Error (Aperture Frame)")
    for a in ax: set_dark_transparent(a)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"formation_error{suffix}.png"))


def plot_observation_precision(pos_err_vec, time, phase, star_vector, out_dir="./results", frame_dcm=None, suffix=""):
    """
    Observation window precision plot — two-column layout:
      LEFT  : full context window (Calibration → Pre-Obs → Fine Obs), coarse scale
      RIGHT : Fine Observation only, independently auto-scaled for precision

    pos_err_vec is pre-projected into the aperture frame by main.py.
    """
    phase = np.array(phase)

    obs_mask     = (phase == "Fine Observation")
    pre_obs_mask = (phase == "Pre-Observation")
    cal_mask     = (phase == "Calibration")

    if not np.any(obs_mask):
        print("No observation window — skipping precision plot.")
        return

    obs_idx = np.where(obs_mask)[0]

    # Left column: from start of Calibration through end of Fine Obs
    if np.any(cal_mask):
        start_left = np.where(cal_mask)[0][0]
    elif np.any(pre_obs_mask):
        start_left = np.where(pre_obs_mask)[0][0]
    else:
        start_left = max(0, obs_idx[0] - max(1, len(obs_idx) // 5))
    sl_left  = slice(start_left, obs_idx[-1] + 1)

    # Right column: Fine Observation only
    sl_right = slice(obs_idx[0], obs_idx[-1] + 1)

    err_l = np.asarray(pos_err_vec[sl_left]);  t_l = time[sl_left];  ph_l = phase[sl_left]
    err_r = np.asarray(pos_err_vec[sl_right]); t_r = time[sl_right]
    mag_l = np.linalg.norm(err_l, axis=1)
    mag_r = np.linalg.norm(err_r, axis=1)

    def _pick_scale(max_e):
        if max_e < 0.001:   return 1e6, "μm"
        elif max_e < 0.1:   return 1e3, "mm"
        elif max_e < 100.0: return 1.0, "m"
        else:               return 1e-3, "km"

    scale_l, unit_l = _pick_scale(np.max(np.abs(err_l)) if len(err_l) else 1.0)
    scale_r, unit_r = _pick_scale(np.max(np.abs(err_r)) if len(err_r) else 1.0)

    t_obs0  = time[obs_idx[0]]
    t_l_rel = t_l - t_obs0
    t_r_rel = t_r - t_obs0

    fig, axes = plt.subplots(3, 2, figsize=(18, 10),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    fig.suptitle("Observation Window — Precision Formation Error", fontsize=14)

    phase_shades = {
        "Calibration":     ("tomato",     0.20),
        "Pre-Observation": ("gold",       0.30),
        "Fine Observation":("darkorange", 0.25),
    }
    for row in range(3):
        for ph_name, (color, alpha) in phase_shades.items():
            msk = (ph_l == ph_name)
            if np.any(msk):
                axes[row, 0].axvspan(t_l_rel[msk][0], t_l_rel[msk][-1],
                                     alpha=alpha, color=color, label=ph_name)
        axes[row, 1].axvspan(t_r_rel[0], t_r_rel[-1], alpha=0.15, color="darkorange")

    # Row 0: Lateral X/Y
    for col, (t_rel, err, sc, unit) in enumerate([
            (t_l_rel, err_l, scale_l, unit_l),
            (t_r_rel, err_r, scale_r, unit_r)]):
        axes[0, col].plot(t_rel, err[:, 0]*sc, "r-", lw=1.2, label="Lateral X")
        axes[0, col].plot(t_rel, err[:, 1]*sc, "g-", lw=1.2, label="Lateral Y")
        axes[0, col].axhline(0, color="w", lw=0.5, ls="--")
        axes[0, col].set_ylabel(f"Lateral [{unit}]")
        axes[0, col].legend(fontsize=9); axes[0, col].grid(alpha=0.3)
    axes[0, 0].set_title("Context: Cal → Pre-Obs → Fine Obs", fontsize=10)
    axes[0, 1].set_title("Fine Observation (zoomed scale)", fontsize=10)

    # Row 1: Focal Z
    for col, (t_rel, err, sc, unit) in enumerate([
            (t_l_rel, err_l, scale_l, unit_l),
            (t_r_rel, err_r, scale_r, unit_r)]):
        axes[1, col].plot(t_rel, err[:, 2]*sc, "b-", lw=1.5, label="Focal Error")
        axes[1, col].axhline(0, color="w", lw=0.5, ls="--")
        axes[1, col].set_ylabel(f"Focal (Z) [{unit}]")
        axes[1, col].legend(fontsize=9); axes[1, col].grid(alpha=0.3)
    axes[1, 0].set_title("Optical Axis (Defocus)", fontsize=10)
    axes[1, 1].set_title("Optical Axis (Defocus) — Fine Obs", fontsize=10)

    # Row 2: Magnitude
    for col, (t_rel, mag, sc, unit) in enumerate([
            (t_l_rel, mag_l, scale_l, unit_l),
            (t_r_rel, mag_r, scale_r, unit_r)]):
        axes[2, col].plot(t_rel, mag*sc, color="white", lw=1.5, label="|Error|")
        axes[2, col].set_ylabel(f"|Error| [{unit}]")
        axes[2, col].set_xlabel("Time from Obs Start [s]")
        axes[2, col].legend(fontsize=9); axes[2, col].grid(alpha=0.3)
    axes[2, 0].set_title("Total Error Magnitude", fontsize=10)
    axes[2, 1].set_title("Total Error Magnitude — Fine Obs", fontsize=10)

    # Stats box: fine obs numbers only
    stats = (f"Fine Obs Stats  [{unit_r}]\n"
             f"  |e| mean : {np.mean(mag_r)*scale_r:.4f}\n"
             f"  |e| max  : {np.max(mag_r)*scale_r:.4f}\n"
             f"  |e| std  : {np.std(mag_r)*scale_r:.4f}\n"
             f"  Lat RMS  : {np.sqrt(np.mean(err_r[:,0]**2+err_r[:,1]**2))*scale_r:.4f}\n"
             f"  Z mean   : {np.mean(err_r[:,2])*scale_r:+.4f}")
    fig.text(0.78, 0.01, stats, color="white", fontsize=8, fontfamily="monospace",
             va="bottom", bbox=dict(boxstyle="round", facecolor="#111111", alpha=0.6))

    plt.tight_layout(rect=[0, 0.07, 1, 0.96])
    for a in axes.flat:
        set_dark_transparent(a)

    data_path = os.path.join(out_dir, "observation_precision_data.npz")
    np.savez(data_path, time_rel_sec=t_r_rel, error_aperture_m=err_r,
             phase=phase[sl_right])

    plt.savefig(os.path.join(out_dir, f"observation_precision{suffix}.png"), dpi=150)
    print(f"  Observation precision plot and data (.npz) saved to "
          f".../{os.path.relpath(out_dir)} (unit: {unit_r}).")



def plot_sun_tracker_error(css_sun_app, css_sun_det,  # noqa: E128
                            true_sun_app, true_sun_det, time, out_dir="./results", suffix=""):
    """
    CSS/WLS sun tracker deviation vs true SPICE sun direction.

    Panel 1 : Total angular error [millidegrees] — both spacecraft
    Panel 2 : Per-component direction cosine error — Aperture
    Panel 3 : Per-component direction cosine error — Detector
    """
    def _ang_err_deg(est, truth):
        dots = np.clip(np.einsum("ij,ij->i", est, truth), -1.0, 1.0)
        return np.degrees(np.arccos(dots))

    err_app  = _ang_err_deg(css_sun_app, true_sun_app)
    err_det  = _ang_err_deg(css_sun_det, true_sun_det)
    diff_app = css_sun_app - true_sun_app
    diff_det = css_sun_det - true_sun_det
    t_min    = time / 60.0

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    fig.suptitle("Sun Tracker (CSS/WLS) Deviations from True Sun Direction",
                 fontsize=14, fontweight="bold")

    # Panel 1 — angular magnitude
    axes[0].plot(t_min, err_app*1e3, color="royalblue",   lw=1.0, alpha=0.85, label="Aperture")
    axes[0].plot(t_min, err_det*1e3, color="darkorange",  lw=1.0, alpha=0.85, label="Detector")
    axes[0].set_ylabel("Angular Error [mdeg]")
    axes[0].set_title("Total Angular Deviation from True Sun Direction")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    for err, label, col in [(err_app, "Aperture", "royalblue"),
                             (err_det, "Detector", "darkorange")]:
        mu_m  = np.mean(err) * 1e3
        sig_m = np.std(err)  * 1e3
        axes[0].axhline(mu_m, color=col, ls="--", lw=0.8, alpha=0.5)
        axes[0].text(t_min[-1]*0.98, mu_m+sig_m*0.5,
                     f"{label}: μ={mu_m:.2f} σ={sig_m:.2f} mdeg",
                     color=col, fontsize=7.5, ha="right", va="bottom")

    # Panels 2 & 3 — per-component
    for i, (diff, label) in enumerate([(diff_app, "Aperture"), (diff_det, "Detector")], start=1):
        axes[i].plot(t_min, diff[:, 0]*1e3, "r-", lw=0.9, alpha=0.85, label="ΔX (body)")
        axes[i].plot(t_min, diff[:, 1]*1e3, "g-", lw=0.9, alpha=0.85, label="ΔY (body)")
        axes[i].plot(t_min, diff[:, 2]*1e3, "b-", lw=0.9, alpha=0.85, label="ΔZ (body)")
        axes[i].axhline(0, color="k", lw=0.5)
        axes[i].set_ylabel("Cosine Diff. [×10⁻³]")
        axes[i].set_title(f"{label} — Per-Component Error (WLS − True)")
        axes[i].legend(fontsize=9); axes[i].grid(alpha=0.3)
    axes[2].set_xlabel("Time [minutes]")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for a in axes: set_dark_transparent(a)
    plt.savefig(os.path.join(out_dir, f"sun_tracker_error{suffix}.png"), dpi=150)
    print(f"  Sun tracker error saved to .../{os.path.relpath(out_dir)}: App μ={np.mean(err_app)*1e3:.2f} mdeg | "
          f"Det μ={np.mean(err_det)*1e3:.2f} mdeg")


def plot_system_overview(r1_n, r2_n, sun_pos_n, moon_pos_n, time, out_dir="./results", suffix=""):
    """
    Top-down (ECI X-Y) view of the Earth-Moon-Sun system.
    Shows the formation trajectory relative to Earth (origin), Moon, and Sun.
    """
    plt.figure(figsize=(12, 12))
    
    # 1. Earth (origin)
    plt.plot(0, 0, 'go', markersize=10, label="Earth")
    
    # 2. Spacecraft Trajectories (Scale up so they are visible if needed, 
    # but here we'll just plot them and let the Moon/Sun set the scale.)
    # Actually, SC at 4.2e7, Moon at 3.8e8. SC will be visible.
    plt.plot(r1_n[:, 0], r1_n[:, 1], 'b-', label="Aperture", alpha=0.9)
    plt.plot(r2_n[:, 0], r2_n[:, 1], 'r--', label="Detector", alpha=0.7)
    
    # 3. Moon Trajectory
    if np.any(np.linalg.norm(moon_pos_n, axis=1) > 1e6):
        plt.plot(moon_pos_n[:, 0], moon_pos_n[:, 1], 'white', label="Moon", linewidth=2)
        plt.plot(moon_pos_n[-1, 0], moon_pos_n[-1, 1], 'white', markersize=6)
        
    # 4. Sun Direction
    # Sun is at ~1.5e11. We'll plot an arrow pointing toward it if off-scale,
    # or just the Sun point if we want to show it. 
    # Let's show the Sun point but note it's very far.
    if np.any(np.linalg.norm(sun_pos_n, axis=1) > 1e6):
        sun_last = sun_pos_n[-1]
        # We might want to limit the axis so Moon is visible.
        # Let's set the limit to 1.2 * Moon distance.
        moon_dist = np.linalg.norm(moon_pos_n[-1])
        limit = 1.2 * moon_dist
        
        # Plot SRP as a direction arrow from origin (pointing away from Sun)
        srp_dir = -sun_last / np.linalg.norm(sun_last)
        plt.arrow(0, 0, srp_dir[0]*limit*0.2, srp_dir[1]*limit*0.1, 
                  color='orange', head_width=limit*0.005, label="Solar Pressure Direction")
        
        # Actually plot the Sun point way out there if we want to zoom out,
        # but let's stick to Moon-scale.
        plt.xlim(-limit, limit)
        plt.ylim(-limit, limit)

    plt.title("System Overview (Top-Down ECI X-Y)")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal'); set_dark_transparent()
    plt.savefig(os.path.join(out_dir, f"system_overview{suffix}.png"), dpi=150)
    print(f"  System overview plot saved to .../{os.path.relpath(out_dir)}.")


def plot_srp_forces(srp_app, srp_det, time, out_dir="./results", suffix=""):
    """
    Plot SRP force magnitudes and per-axis components for both SC.
    """
    t_hr = time / 3600.0
    f_app_mag = np.linalg.norm(srp_app, axis=1)
    f_det_mag = np.linalg.norm(srp_det, axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Solar Radiation Pressure (SRP) Forces", fontsize=14, fontweight="bold")

    # Aperture (Chief)
    axes[0].plot(t_hr, f_app_mag * 1e3, 'k-', lw=1.5, label="|Force| Total")
    axes[0].plot(t_hr, srp_app[:, 0] * 1e3, 'r--', lw=1.0, label="Fx (inertial)", alpha=0.7)
    axes[0].plot(t_hr, srp_app[:, 1] * 1e3, 'g--', lw=1.0, label="Fy (inertial)", alpha=0.7)
    axes[0].plot(t_hr, srp_app[:, 2] * 1e3, 'b--', lw=1.0, label="Fz (inertial)", alpha=0.7)
    axes[0].set_ylabel("Force [mN]")
    axes[0].set_title("Aperture (Circular Plate Model)")
    axes[0].legend(fontsize=9, loc="upper right")
    axes[0].grid(alpha=0.3)

    # Detector (Deputy)
    axes[1].plot(t_hr, f_det_mag * 1e3, 'k-', lw=1.5, label="|Force| Total")
    axes[1].plot(t_hr, srp_det[:, 0] * 1e3, 'r--', lw=1.0, label="Fx (inertial)", alpha=0.7)
    axes[1].plot(t_hr, srp_det[:, 1] * 1e3, 'g--', lw=1.0, label="Fy (inertial)", alpha=0.7)
    axes[1].plot(t_hr, srp_det[:, 2] * 1e3, 'b--', lw=1.0, label="Fz (inertial)", alpha=0.7)
    axes[1].set_ylabel("Force [mN]")
    axes[1].set_xlabel("Time [hours]")
    axes[1].set_title("Detector (Cannonball Model)")
    axes[1].legend(fontsize=9, loc="upper right")
    axes[1].grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for a in axes: set_dark_transparent(a)
    plt.savefig(os.path.join(out_dir, f"srp_forces{suffix}.png"), dpi=150)
    print(f"  SRP force plot saved to .../{os.path.relpath(out_dir)}.")


# =============================================================================
# ORBITAL GEOMETRY DIAGRAM
# =============================================================================

def _perifocal_basis(i_rad, Omega_rad, omega_rad):
    """Return perifocal unit vectors P (perigee), Q (transverse), W (normal) in ECI."""
    ci, si = np.cos(i_rad), np.sin(i_rad)
    cO, sO = np.cos(Omega_rad), np.sin(Omega_rad)
    co, so = np.cos(omega_rad), np.sin(omega_rad)
    P = np.array([cO*co - sO*so*ci,  sO*co + cO*so*ci,  so*si])
    Q = np.array([-cO*so - sO*co*ci, -sO*so + cO*co*ci, co*si])
    W = np.cross(P, Q)
    return P / np.linalg.norm(P), Q / np.linalg.norm(Q), W / np.linalg.norm(W)


def _orbit_xyz(a, e, P, Q, n=400):
    """Return (3, n) array of orbit positions in ECI [km]."""
    f = np.linspace(0, 2*np.pi, n)
    r = a * (1 - e**2) / (1 + e*np.cos(f))
    return (r * np.cos(f))[:, None] * P + (r * np.sin(f))[:, None] * Q


def _arc3d(ax, v_from, v_to, radius, center=None, n=60, **kw):
    """Draw a circular arc from unit v_from to unit v_to at given radius."""
    if center is None:
        center = np.zeros(3)
    v0 = v_from / np.linalg.norm(v_from)
    v1 = v_to   / np.linalg.norm(v_to)
    normal = np.cross(v0, v1)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-10:
        return
    normal /= norm_len
    angle = np.arccos(np.clip(np.dot(v0, v1), -1, 1))
    t = np.linspace(0, 1, n)
    pts = []
    for ti in t:
        a = angle * ti
        vr = v0*np.cos(a) + np.cross(normal, v0)*np.sin(a) + normal*np.dot(normal, v0)*(1-np.cos(a))
        pts.append(center + radius * vr)
    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], **kw)


def plot_orbital_geometry(cfg, out_dir="./results", suffix=""):
    """
    Rich 3D diagram of the formation orbital geometry.
    Shows:
      - Aperture & detector ellipses in ECI
      - Earth sphere
      - Equatorial plane reference disk
      - Eccentricity vector arrow (perigee direction)
      - RAAN Arc (Ω) in equatorial plane
      - Inclination arc (i) at ascending node
      - Argument of periapsis arc (ω) in orbit plane
    """
    a  = cfg.a / 1e3          # km
    e  = cfg.eccentricity
    i  = np.radians(cfg.base_i_deg)
    Ω  = np.radians(cfg.base_raan_deg)
    ω  = np.radians(cfg.base_omega_deg)
    Δi = np.radians(cfg.det_delta_i_deg)
    ΔΩ = np.radians(getattr(cfg, '_det_delta_O_deg', 0.0))
    Δω = np.radians(getattr(cfg, '_det_delta_omega_deg', 0.0))
    R_E = 6378.1  # Earth radius km

    # Aperture perifocal frame
    P_a, Q_a, W_a = _perifocal_basis(i, Ω, ω)
    # Detector perifocal frame
    P_d, Q_d, W_d = _perifocal_basis(i + Δi, Ω + ΔΩ, ω + Δω)

    # Orbit paths
    xyz_a  = _orbit_xyz(a, e, P_a, Q_a)   # (N,3)
    xyz_d  = _orbit_xyz(a, e, P_d, Q_d)

    # Scale for arrows and arcs
    r_p  = a * (1 - e)   # perigee distance km
    arc_r = r_p * 0.45
    arr_s = r_p * 0.55

    fig = plt.figure(figsize=(13, 10))
    ax  = fig.add_subplot(111, projection='3d')

    # ── Earth sphere ────────────────────────────────────────────────────────
    u_e, v_e = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
    xs = R_E * np.cos(u_e) * np.sin(v_e)
    ys = R_E * np.sin(u_e) * np.sin(v_e)
    zs = R_E * np.cos(v_e)
    ax.plot_surface(xs, ys, zs, color='deepskyblue', alpha=0.25, linewidth=0)

    # ── Equatorial plane disk ───────────────────────────────────────────────
    eq_r   = a * 1.05
    th_eq  = np.linspace(0, 2*np.pi, 200)
    ax.plot(eq_r*np.cos(th_eq), eq_r*np.sin(th_eq), np.zeros(200),
            color='white', lw=0.5, alpha=0.3, linestyle=':')

    # ── Orbits ──────────────────────────────────────────────────────────────
    ax.plot(xyz_a[:, 0], xyz_a[:, 1], xyz_a[:, 2],
            color='orange', lw=2.0, label='Aperture orbit')
    ax.plot(xyz_d[:, 0], xyz_d[:, 1], xyz_d[:, 2],
            color='cyan',   lw=1.2, label='Detector orbit', linestyle='--')

    # ── Vernal equinox (reference direction) ───────────────────────────────
    vernal = np.array([1., 0., 0.])
    ax.quiver(0, 0, 0, arr_s*vernal[0], arr_s*vernal[1], arr_s*vernal[2],
              color='yellow', arrow_length_ratio=0.12, linewidth=1.2)
    ax.text(arr_s*1.12, 0, 0, '♈ (γ)', color='yellow', fontsize=8)

    # ── Ascending node direction ────────────────────────────────────────────
    node_dir = np.array([np.cos(Ω), np.sin(Ω), 0.])
    ax.quiver(0, 0, 0, arc_r*node_dir[0], arc_r*node_dir[1], 0,
              color='lime', arrow_length_ratio=0.14, linewidth=1.0, linestyle='--')
    ax.text(arc_r*1.1*node_dir[0], arc_r*1.1*node_dir[1], 0,
            'AN', color='lime', fontsize=8)

    # ── RAAN arc (Ω): equatorial plane, from vernal equinox to ascending node ──
    _arc3d(ax, vernal, node_dir, arc_r * 0.75,
           color='yellow', lw=1.5, linestyle='-')
    mid_Ω = Ω / 2
    ax.text(arc_r*0.75*np.cos(mid_Ω)*1.15,
            arc_r*0.75*np.sin(mid_Ω)*1.15, 0,
            f'Ω={cfg.base_raan_deg:.0f}°', color='yellow', fontsize=8)

    # ── Inclination arc (i): at ascending node, from equatorial to orbit normal ──
    equat_up = np.array([0., 0., 1.])   # z-axis (equatorial north)
    _arc3d(ax, equat_up, W_a, arc_r * 0.7, center=np.zeros(3),
           color='magenta', lw=1.5, linestyle='-')
    mid_i = i / 2
    # place label along the arc midpoint direction
    arc_mid_i = np.cos(mid_i)*equat_up + np.sin(mid_i)*(W_a - np.dot(W_a, equat_up)*equat_up)
    arc_mid_i /= np.linalg.norm(arc_mid_i)
    ax.text(*(arc_r*0.7*arc_mid_i*1.25),
            f'i={cfg.base_i_deg:.0f}°', color='magenta', fontsize=8)

    # ── Argument of periapsis arc (ω): in orbit plane, from ascending node to perigee ──
    # ascending node direction projected into orbit plane:
    node_in_plane = node_dir - np.dot(node_dir, W_a) * W_a
    node_in_plane /= np.linalg.norm(node_in_plane)
    _arc3d(ax, node_in_plane, P_a, arc_r * 0.55, color='tomato', lw=1.5)
    mid_ω = ω / 2
    # midpoint direction in orbit plane
    arc_mid_ω = P_a if abs(ω) < 1e-6 else (np.cos(mid_ω)*node_in_plane +
                np.sin(mid_ω)*np.cross(W_a, node_in_plane) / max(1e-8, np.linalg.norm(np.cross(W_a, node_in_plane))))
    arc_mid_ω /= np.linalg.norm(arc_mid_ω)
    ax.text(*(arc_r*0.55*arc_mid_ω*1.3),
            f'ω={cfg.base_omega_deg:.0f}°', color='tomato', fontsize=8)

    # ── Eccentricity vector (points toward perigee) ─────────────────────────
    r_perigee_a = r_p * P_a
    ax.quiver(0, 0, 0, r_perigee_a[0], r_perigee_a[1], r_perigee_a[2],
              color='orange', arrow_length_ratio=0.08, linewidth=2.0)
    ax.text(*(r_perigee_a * 1.08), 'e⃗', color='orange', fontsize=10, fontweight='bold')

    # Perigee point on aperture orbit
    ax.scatter(*r_perigee_a, color='orange', s=60, zorder=5)

    # ── Orbit normal arrow ──────────────────────────────────────────────────
    ax.quiver(0, 0, 0, W_a[0]*arr_s, W_a[1]*arr_s, W_a[2]*arr_s,
              color='white', arrow_length_ratio=0.10, linewidth=1.2, alpha=0.7)
    ax.text(*(W_a*arr_s*1.1), 'ĥ / ★', color='white', fontsize=8)

    # ── Axes and cosmetics ──────────────────────────────────────────────────
    lim = a * 1.1
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim*0.6, lim*0.6)
    ax.set_xlabel('X [km]', color='white')
    ax.set_ylabel('Y [km]', color='white')
    ax.set_zlabel('Z [km]', color='white')
    ax.set_title(
        f'Formation Orbital Geometry  (a={a:.0f} km, e={e}, '
        f'i={cfg.base_i_deg:.1f}°, Ω={cfg.base_raan_deg:.1f}°, ω={cfg.base_omega_deg:.1f}°\n'
        f'Δi={cfg.det_delta_i_deg*1000:.3f} mdeg, ΔΩ={getattr(cfg, "_det_delta_O_deg", 0)*1000:.3f} mdeg)',
        fontsize=9.5, color='white'
    )
    ax.legend(loc='upper right', fontsize=9)
    if hasattr(ax, 'set_box_aspect'):
        ax.set_box_aspect((1, 1, 0.5))
    set_dark_transparent(ax)
    fpath = os.path.join(out_dir, f'orbital_geometry{suffix}.png')
    plt.savefig(fpath, dpi=150)
    print(f'  Orbital geometry plot saved to .../{os.path.relpath(out_dir)}.')



def run_all(log1, log2, time, extra_data: dict = None, out_dir="./results"):
    """
    Generate the full plot suite from a completed simulation run.

    Parameters
    ----------
    log1, log2  : BSK state recorders for Aperture and Detector
    time        : time array [s]
    extra_data  : dict from main.py containing trajectory, phase, CSS data, etc.
    out_dir     : path to save plot results
    """
    from utilities import sim_tag
    _mkresults(out_dir)

    # Derive the sim tag from cfg if available (e.g. "_sim_a42164200.0_e0.8_i-8_raan70_omega20")
    suffix = ""
    if extra_data and "cfg" in extra_data:
        suffix = "_" + sim_tag(extra_data["cfg"])

    # Always: relative motion in the frozen aperture frame
    plot_relative_motion(log1, log2, time, out_dir=out_dir, suffix=suffix,
                         frame_dcm=extra_data.get("aperture_frame_dcm") if extra_data else None)

    if not extra_data:
        print(f"Plots saved to .../{os.path.relpath(out_dir)}")
        return

    plot_orbital_trajectories(extra_data["r1_n"], extra_data["r2_n"],
                               extra_data["engaged"], time,
                               aperture_frame_dcm=extra_data.get("aperture_frame_dcm"),
                               out_dir=out_dir, suffix=suffix)
    plot_delta_v(extra_data["dv"], time, out_dir=out_dir, suffix=suffix)

    if "pos_err" in extra_data:
        _fdcm = extra_data.get("aperture_frame_dcm", None)
        plot_formation_error(extra_data["pos_err"], time,
                             extra_data.get("star_vector", [0,0,1]),
                             frame_dcm=_fdcm,
                             phase=extra_data.get("phase"),
                             out_dir=out_dir, suffix=suffix)

    if "pos_err" in extra_data and "phase" in extra_data:
        _fdcm = extra_data.get("aperture_frame_dcm", None)
        plot_observation_precision(extra_data["pos_err"], time,
                                   extra_data["phase"],
                                   extra_data.get("star_vector", [0,0,1]),
                                   frame_dcm=_fdcm, out_dir=out_dir, suffix=suffix)

    # Engagement-window zoomed plots
    if "dv_xyz" in extra_data and any(extra_data["engaged"]):
        engaged = np.asarray(extra_data["engaged"])
        idxs    = np.where(engaged)[0]
        buf     = 120   # 120-step buffer (~12 s at dt=0.1 s)
        sl      = slice(max(0, idxs[0]-buf), min(len(engaged)-1, idxs[-1]+buf))
        plot_detailed_thrust(extra_data["dv_xyz"][sl], extra_data["dv"][sl], time[sl], out_dir=out_dir, suffix=suffix)
        if "rw_speeds" in extra_data:
            plot_rw_speeds(extra_data["rw_speeds"][sl], time[sl], out_dir=out_dir, suffix=suffix)

    # Sun tracker deviation
    keys = ("css_sun_app", "css_sun_det", "true_sun_app", "true_sun_det")
    if all(k in extra_data for k in keys):
        # Only plot if we have non-zero estimates
        app_est = extra_data["css_sun_app"]
        if np.any(np.linalg.norm(app_est, axis=1) > 0.1):
            plot_sun_tracker_error(extra_data["css_sun_app"], extra_data["css_sun_det"],
                                   extra_data["true_sun_app"], extra_data["true_sun_det"],
                                   time, out_dir=out_dir, suffix=suffix)
        else:
            print("  Sun tracker plot skipped (CSS data disabled).")

    # System overview (Planetary scale)
    if "sun_pos_n" in extra_data and "moon_pos_n" in extra_data:
        plot_system_overview(extra_data["r1_n"], extra_data["r2_n"],
                             extra_data["sun_pos_n"], extra_data["moon_pos_n"],
                             time, out_dir=out_dir, suffix=suffix)

    # SRP forces
    if "srp_app_vec" in extra_data and "srp_det_vec" in extra_data:
        plot_srp_forces(extra_data["srp_app_vec"], extra_data["srp_det_vec"], time, out_dir=out_dir, suffix=suffix)

    # Observation Delta-V (Filtered)
    if "phase" in extra_data:
        plot_delta_v_observation_window(extra_data["dv_xyz"], extra_data["dv"], time, extra_data["phase"], out_dir=out_dir, suffix=suffix)

    # Orbital geometry diagram
    if "cfg" in extra_data:
        plot_orbital_geometry(extra_data["cfg"], out_dir=out_dir, suffix=suffix)

    print(f"All plots saved to .../{os.path.relpath(out_dir)}")
