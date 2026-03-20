import numpy as np
from Basilisk.utilities import macros
from config import SimConfig
from gravity import setup_srp

def test_srp_plate():
    cfg = SimConfig(enable_srp=True, enable_third_body=True, app_srp_model="plate", app_diameter=30.0)
    
    # Fake sun message as setup_srp only checks if it's not None
    sun_msg = "dummy"
    
    srp_app_fn, _ = setup_srp(None, "task", cfg, None, None, sun_msg)
    
    # MRP Rotation logic (NB @ Body_Z = Inertial_Dir)
    # sigma = e_hat * tan(phi/4)
    # Body +Z to Inertial +X: rotate by 90deg about Y: sigma = [0, tan(pi/8), 0]
    
    r_sc = np.array([0, 0, 0])
    r_sun = np.array([1.5e11, 0, 0]) # Sun is along +X in inertial
    mass = cfg.app_mass
    
    print("-" * 40)
    print("SRP Plate Model Verification")
    print("-" * 40)
    
    # Case 1: Normal incidence (Plate points at Sun)
    # sigma_BN = [0, tan(pi/8), 0] for 90 deg rotate about Y
    sigma_BN = [0, np.tan(np.pi/8), 0] 
    f_normal = srp_app_fn(r_sc, r_sun, sigma_BN, mass)
    print(f"Normal incidence (Plate normal = ? inertial, Sun = +X):")
    # I can't easily see n_hat_n without re-implementing math or printing inside gravity.py
    # Let me just move the logic into the test for a moment to verify the math
    def check_math(sigma_BN):
        s2 = np.dot(sigma_BN, sigma_BN)
        sigma_BN = np.array(sigma_BN)
        s_cross = np.array([[0, -sigma_BN[2], sigma_BN[1]],
                           [sigma_BN[2], 0, -sigma_BN[0]],
                           [-sigma_BN[1], sigma_BN[0], 0]])
        BN = np.eye(3) + 8*s_cross @ s_cross / (1+s2)**2 + 4*(1-s2)*s_cross / (1+s2)**2
        NB = BN.T
        n_hat_n = NB @ np.array([0, 0, 1])
        s_hat_n = np.array([1, 0, 0])
        cos_theta = np.dot(s_hat_n, n_hat_n)
        return n_hat_n, cos_theta

    n1, c1 = check_math(sigma_BN)
    print(f"  sigma: {sigma_BN}")
    print(f"  n_hat: {n1}")
    print(f"  cos_theta: {c1}")
    print(f"  Force: {f_normal} N")
    
    # Case 3: Backside (Angle > 90 deg)
    sigma_back = [0, -np.tan(np.pi/8), 0] # Body +Z points at -X inertial, Sun at +X
    n2, c2 = check_math(sigma_back)
    f_back = srp_app_fn(r_sc, r_sun, sigma_back, mass)
    print(f"Backside (Plate normal = -X inertial, Sun = +X):")
    print(f"  sigma: {sigma_back}")
    print(f"  n_hat: {n2}")
    print(f"  cos_theta: {c2}")
    print(f"  Force: {f_back} N")
    print(f"  Mag:   {np.linalg.norm(f_back):.6e}")

    # Cannonball comparison
    cfg.app_srp_model = "cannonball"
    srp_app_fn_cb, _ = setup_srp(None, "task", cfg, None, None, sun_msg)
    f_cb = srp_app_fn_cb(r_sc, r_sun, sigma_BN, mass)
    print(f"Cannonball (Aperture A = pi*15^2):")
    print(f"  Force: {f_cb} N")
    print(f"  Mag:   {np.linalg.norm(f_cb):.6e}")

if __name__ == "__main__":
    test_srp_plate()
