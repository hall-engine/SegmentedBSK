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
    Determines where each mirror points.
    <-z-- det ----- f ---------- |segment.
    ^ y pos and > x pos

    The detector position is expressed in the APERTURE FRAME (frozen perifocal
    frame: z = orbit-normal / star-vector, y = velocity at perigee, x = y×z).
    This frame is attitude-independent: det_z ≈ +focal_length and det_x/det_y ≈ 0
    whenever the formation is correct, regardless of attitude transients.

    If aperture_frame_dcm is None (backwards-compat), falls back to body frame.
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
        # Fixed-design piston: parabolic shape for the DESIGN focal length (cfg.focal_length)
        cenx, ceny = states[s].position[0], states[s].position[1]
        desired_piston = (cenx**2 + ceny**2) / (4.0 * cfg.focal_length)
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
    
    return states