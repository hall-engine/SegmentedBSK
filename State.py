# -*- coding: utf-8 -*-
import numpy as np

class State():
    def __init__(self, number, position, mirror_actuation, desired_mirror_actuation):
        """
        Parameters
        ----------
        number : int
            The number of the state.
        position : list
            The position of the state.
        mirror_actuation : list
            The mirror actuation of the state.
        desired_mirror_actuation : list
            The desired mirror actuation of the state.
        """
        self.number = number
        self.position = position
        self.mirror_actuation = mirror_actuation
        self.desired_mirror_actuation = desired_mirror_actuation
        self.point_on_det_plane = np.zeros((1,2))
        # create histories array
        self.hist_position = np.empty((0, 6))                       # [x, y, z, theta_z, theta_y, theta_z]
        self.hist_mirror_actuation = np.empty((0, 6))               # [tip, tilt, piston] (phi_x, phi_y, v_z)
        self.hist_desired_mirror_actuation = np.empty((0, 6))       # [tip, tilt, piston, tip_dot, tilt_dot, piston_dot]
        self.hist_point_on_det_plane = np.empty((0,2))              # [dX and dY]
        
    def store_histories(self):
        """
        When called, stores the empty history arrays. 
        Returns
        -------
        None.
        """
        # Append the current state to the history arrays
        self.hist_position = np.append(self.hist_position, [self.position], axis=0)
        self.hist_mirror_actuation = np.append(self.hist_mirror_actuation, [self.mirror_actuation], axis=0)
        self.hist_desired_mirror_actuation = np.append(self.hist_desired_mirror_actuation, [self.desired_mirror_actuation], axis=0)
        self.hist_point_on_det_plane = np.append(self.hist_point_on_det_plane,  [self.point_on_det_plane], axis=0)
