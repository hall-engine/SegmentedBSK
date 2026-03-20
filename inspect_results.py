import h5py
import numpy as np
import sys
import matplotlib.pyplot as plt

f = h5py.File('./results/multiple_orientations/sim_a45000000.0_e0.8_i10.0_raan10.0_omega10.0/mirror_states.h5', 'r')
time = np.array(f['time'])
phase = np.array(f['phase'], dtype=str)

obs_mask = (phase == "Fine Observation")
if not np.any(obs_mask):
    print("No observation phase found.")
    sys.exit()

import json
cfg = json.load(open('./results/multiple_orientations/sim_a45000000.0_e0.8_i10.0_raan10.0_omega10.0/sim_config.json', 'r'))

try:
    rel_pos_B = np.array(f['rel_pos_B'])
except KeyError:
    print("No rel_pos_B found.")
    sys.exit()

rb_obs = rel_pos_B[obs_mask]

print("Rel Pos B during Fine Observation (X, Y, Z):")
print(f"X: min={rb_obs[:, 1].min():.4f}, max={rb_obs[:, 1].max():.4f}, mean={rb_obs[:, 1].mean():.4f}")
print(f"Y: min={rb_obs[:, 2].min():.4f}, max={rb_obs[:, 2].max():.4f}, mean={rb_obs[:, 2].mean():.4f}")
print(f"Z: min={rb_obs[:, 3].min():.4f}, max={rb_obs[:, 3].max():.4f}, mean={rb_obs[:, 3].mean():.4f}")

f.close()
