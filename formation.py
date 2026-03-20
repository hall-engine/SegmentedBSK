"""
formation.py — Formation Manager
==================================
Configures the inter-spacecraft relationship:
  - Chief (Aperture): inertial star-pointing attitude
  - Deputy (Detector): LOS pointing at Chief
  - Translation control is handled externally in control.py via ExtForceTorque
"""

from spacecraft import SpacecraftManager


class FormationManager:
    """Configures attitude guidance for a two-spacecraft formation."""

    def __init__(self, sim, task_name: str):
        self.sim = sim
        self.task_name = task_name

    def setup_star_alignment(self,
                              chief_sc, chief_mgr: SpacecraftManager,
                              deputy_sc, deputy_mgr: SpacecraftManager,
                              star_vector, focal_length: float):
        """
        Star-alignment formation mode:
          1. Chief points its body +Z at `star_vector` (inertial).
          2. Deputy points its body +Z toward the chief (LOS).
          3. Translation control (maintaining focal length) is external.

        Parameters
        ----------
        chief_sc   : BSK Spacecraft object (Aperture) — retained for API consistency
        chief_mgr  : SpacecraftManager for chief
        deputy_sc  : BSK Spacecraft object (Detector) — retained for API consistency
        deputy_mgr : SpacecraftManager for deputy
        star_vector: [3] inertial unit vector toward target star
        focal_length: desired separation [m]

        Returns
        -------
        star_vector, focal_length  (echoed for convenience)
        """
        # 1. Chief purely points at the star vector using inertial tracking error & PD MRP Feedback
        chief_mgr.add_inertial_pointing_control(star_vector)

        # 2. Deputy LOS pointing — reuse the chief's existing SimpleNav transOutMsg.
        #    No second SimpleNav is needed; chief_mgr.nav was set up by add_simple_nav().
        deputy_mgr.add_los_pointing_control(chief_mgr.nav.transOutMsg)

        return star_vector, focal_length
