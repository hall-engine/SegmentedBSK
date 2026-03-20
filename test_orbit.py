import numpy as np
from Basilisk.utilities import orbitalMotion

def test():
    mu = 3.986004418e14
    a = 6378136.3 + 500e3
    e = 0.5
    i = np.radians(10.0)
    Omega = np.radians(10.0)
    omega = np.radians(10.0)
    
    oe_app = orbitalMotion.ClassicElements()
    oe_app.a = a
    oe_app.e = e
    oe_app.i = i
    oe_app.Omega = Omega
    oe_app.omega = omega
    
    # Target peak
    E_peak = np.radians(90.0)
    f_peak = 2.0 * np.arctan(np.sqrt((1.0 + e) / (1.0 - e)) * np.tan(E_peak / 2.0))
    oe_app.f = f_peak
    
    r_c, v_c = orbitalMotion.elem2rv(mu, oe_app)
    r_c, v_c = np.array(r_c), np.array(v_c)
    
    h = np.cross(r_c, v_c)
    n = h / np.linalg.norm(h)
    
    r_d = r_c + n * 5000.0
    v_d = v_c
    
    oe_det = orbitalMotion.rv2elem(mu, r_d, v_d)
    
    print("Aperture:")
    print(f"a: {oe_app.a:.2f}, e: {oe_app.e:.6f}, i: {np.degrees(oe_app.i):.4f}, OM: {np.degrees(oe_app.Omega):.4f}, om: {np.degrees(oe_app.omega):.4f}")
    
    print("Detector:")
    print(f"a: {oe_det.a:.2f}, e: {oe_det.e:.6f}, i: {np.degrees(oe_det.i):.4f}, OM: {np.degrees(oe_det.Omega):.4f}, om: {np.degrees(oe_det.omega):.4f}")

test()
