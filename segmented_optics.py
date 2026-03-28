# -*- coding: utf-8 -*-

import poppy
import numpy as np
import astropy.units as units
import mirror_controller
from Basilisk.utilities import RigidBodyKinematics
from Basilisk.architecture import sysModel
from State import State
from control import MissionController


def initialise_aperture_segments(cfg):
    """
    Intialises the segmtned aperture system in POPPY.
    """
    # define configuration values
    rings = cfg.rings                           # number of rings
    gap = cfg.gap * units.cm                    # gap between segments [cm]
    f = cfg.focal_length * units.m              # focal length [m]
    flattoflat = cfg.flat_to_flat * units.m     # flat to flat distance [m]
    # initialise the HEXDM aperture (that is deformable)
    hexdm = poppy.dms.HexSegmentedDeformableMirror(rings=rings, flattoflat=flattoflat, gap=gap, include_factor_of_two=True)
    print('>> HEXDM shape successfully initialised')
    states = {}
    filler6 = [1e-9] * 6
    
    # run loop around segment list and initialise them
    for s in range(len(hexdm.segmentlist)):
        states[s] = State(number = s,
                            position = [hexdm._aper_center(s)[1], hexdm._aper_center(s)[0], 0, 1e-9, 1e-9, 1e-9],
                            mirror_actuation = filler6.copy(),
                            desired_mirror_actuation = filler6.copy())
    print(f'>> {len(states)} states sucessfully initialised')
    # apply geometry to segments
    states = apply_shape_to_pistons(states, hexdm, cfg, curve='parabolic', randomise=cfg.initial_random_actuation)
    # initiate the controller
    controllers = initialise_controller(cfg)
    # pull aperture diameter and strip astropy units
    cfg.app_diameter = float(hexdm.pupil_diam.to_value(units.m))
    # return
    return hexdm, states, controllers, cfg


def apply_shape_to_pistons(states, hexdm, cfg, curve, randomise=1e-3):
    """
    Applies a curvature shape based on curve argument.
    Randomise set as 1e-6 as default. Set to None for it to NOT randomise. 
    """
    # parabolic
    if curve == 'parabolic':
        def _calc(x, y, f):
            return (x**2 + y**2) / (4 * f)
    # spherical
    if curve == 'spherical':
        def _calc(x, y, f):
            return f - np.sqrt(f**2 - (x**2 + y**2))
    # loop
    for s in states:
        cenx, ceny = states[s].position[:2]
        z = _calc(cenx, ceny, cfg.focal_length)
        states[s].mirror_actuation[2] = z
        states[s].desired_mirror_actuation[2] = z
    print(f'>> {curve} curve applied to current and desired piston positions')
    # if I want to randomise for a better graphic
    if randomise is not None:
        for s in states:
            np.random.seed(s) # unique seed for each segment
            states[s].mirror_actuation[0] = (2*np.random.rand() - 1) * randomise
            states[s].mirror_actuation[1] = (2*np.random.rand() - 1) * randomise
        print(f'>> mirror actuation randomised by {randomise} factor')
    # return
    return states


def initialise_controller(cfg):
    """
    Initialises the controllers and stack into a list.
    """
    controllers = []
    segLQR = mirror_controller.segmentLQR(cfg)
    controllers.append(segLQR)
    return controllers


def actuate_mirrors(hexdm, states, cfg):
    """
    Performs mirror actuation on the HEXDM aperture object using 
    DESIRED CHARACTERISTICS RATHER THAN ACTUAL.
    THIS IS TO FIND IDEAL WFE FOR THAT SITUATION!

    Parameters
    ----------
    states
    hexdm
    
    Returns
    -------
    hexdm.
    """
    # runs for all states in system.
    debug = False
    for s in states: 
        [tip, tilt, piston, omega_tip, omega_tilt, piston_velocity] = states[s].mirror_actuation    
        hexdm.set_actuator(s, piston, -tip, -tilt)
        if debug: print(states[s].mirror_actuation)
        if debug: print(f'{s} {piston} {tip} {tilt}')
    return hexdm


def gauge_where_pointing(hexdm, states, app_state, det_state, cfg, aperture_frame_dcm=None):
    """
    Determines where each mirror points and computes desired actuation.
    <-z-- det ----- f ---------- |segment.
    ^ y pos and > x pos

    The detector position is expressed in the APERTURE FRAME (frozen perifocal
    frame: z = orbit-normal / star-vector, y = velocity at perigee, x = y×z).
    This frame is attitude-independent: det_z ≈ +focal_length and det_x/det_y ≈ 0
    whenever the formation is correct, regardless of attitude transients.

    If aperture_frame_dcm is None (backwards-compat), falls back to body frame.

    AO / WFS noise model (Option A + C fine metrology)
    ---------------------------------------------------
    When cfg.enable_ao_metrology_noise is True, Gaussian noise representative of
    a photon/read-noise-limited space Wavefront Sensor is injected into the
    desired_mirror_actuation BEFORE it is handed to the LQR controller:

        desired_piston += N(0, cfg.mirror_wfs_piston_noise_m)
        desired_tip    += N(0, cfg.mirror_wfs_tiptilt_noise_rad)
        desired_tilt   += N(0, cfg.mirror_wfs_tiptilt_noise_rad)

    This is a completely separate noise source from the coarse inter-spacecraft
    metrology (metrology_resolution_m).  The WFS measures optical path differences
    at the pupil plane — not spacecraft separation.  Default values correspond to a
    LUVOIR-class space telescope (10 nm piston, 0.1 µrad tip/tilt).
    Actuator quantization is applied inside lqr_control_full() in mirror_controller.py.
    """
    debug = False
    debugseg = 1
    # Extract state vectors
    r_app_N = np.array(app_state.r_CN_N)
    r_det_N = np.array(det_state.r_CN_N)
    # Relative position in aperture frame (attitude-independent)
    r_rel_N = r_det_N - r_app_N
    if aperture_frame_dcm is not None:
        r_rel_B = np.array(aperture_frame_dcm) @ r_rel_N
    else:
        # fallback: instantaneous body frame (attitude-dependent — may be noisy)
        sigma_BN = np.array(app_state.sigma_BN)
        dcm_BN   = RigidBodyKinematics.MRP2C(sigma_BN)
        r_rel_B  = dcm_BN @ r_rel_N
    det_x, det_y, det_z = r_rel_B[0], r_rel_B[1], r_rel_B[2]
    focal_z = cfg.focal_length
    pointlist = np.zeros((0,2))
    for s in states:
        seg_x, seg_y, seg_z = states[s].position[0:3]
        tip, tilt, piston = states[s].mirror_actuation[0:3]
        # Z-separation from segment surface to detector (aperture frame)
        # At nominal alignment: det_z ≈ +focal_length, seg_z ≈ piston ≈ 0
        dZ = det_z - seg_z - piston
        # guard against degenerate (detector behind mirror or too close)
        if abs(dZ) < 1e-3:
            dZ = cfg.focal_length
        # determine where the segments are currently pointing (tip=x, tilt=y)
        pointX = dZ * np.tan(2*tip) + seg_x
        pointY = dZ * np.tan(2*tilt) + seg_y
        # determine where it SHOULD point
        dY = det_y - seg_y
        dX = det_x - seg_x
        # angles
        desired_tip  = ( 0.5 * np.arctan2(dX, dZ) )
        desired_tilt = ( 0.5 * np.arctan2(dY, dZ) )
        # Dynamic piston: adapts parabolic shape based on instantaneous Z-distance
        cenx, ceny = states[s].position[0], states[s].position[1]
        desired_piston = (cenx**2 + ceny**2) / (4.0 * dZ)
        # put back in
        states[s].desired_mirror_actuation[:2] = desired_tip, desired_tilt
        states[s].desired_mirror_actuation[2]  = desired_piston
        states[s].point_on_det_plane = np.array([pointX, pointY])
        # debug
        if (debug==True) & (s==debugseg):
            print(f'segment {s}')
            print(f'tip {tip} tilt {tilt} piston {piston}')
            print(f'segx {seg_x} segy {seg_y} segz {seg_z}')
            print(f'pointx {pointX} pointy {pointY}')
            print(f'detx {det_x} dety {det_y}')
            print(f'desired tip {desired_tip} desired tilt {desired_tilt}')
            print(30*'-')

    # ── WFS / AO fine metrology noise injection ────────────────────────────────
    # Applied AFTER the geometry loop so it is independent per segment per frame.
    # This represents a single WFS readout (photon noise + read noise), drawn fresh
    # every mirror_control_dt — consistent with a photon-noise-limited sensor model.
    if getattr(cfg, 'enable_ao_metrology_noise', False):
        sigma_piston   = getattr(cfg, 'mirror_wfs_piston_noise_m',    10e-9)
        sigma_tiptilt  = getattr(cfg, 'mirror_wfs_tiptilt_noise_rad', 0.1e-6)
        for s in states:
            states[s].desired_mirror_actuation[0] += np.random.normal(0.0, sigma_tiptilt)  # tip
            states[s].desired_mirror_actuation[1] += np.random.normal(0.0, sigma_tiptilt)  # tilt
            states[s].desired_mirror_actuation[2] += np.random.normal(0.0, sigma_piston)   # piston

    return states