import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# provided function
def _computeFFT(array, time_array):
    x = np.array(array)
    x = x - np.mean(x)                  # remove DC offset

    N = len(x)
    dt = time_array[1]-time_array[0]    # set this to your sampling interval
    window = np.hanning(N)              # smooth tapering function
    xw = x * window

    fft_vals = np.fft.rfft(xw)          # fast fourier transform
    freqs = np.fft.rfftfreq(N, d=dt)    # frequencies in terms of Hz
    power = np.abs(fft_vals)**2         # magnitude of each frequency term

    return freqs, power

# Assuming a_vals, i_vals, omega_vals, phase, full_times, eng_times, rel_pos_B are already in scope
# or loaded from the H5 file.

coord_letter = ['x', 'y', 'z']
results_std = [] # List of tuples: (a, i, omega, [std_x, std_y, std_z])
results_fft = [] # List of lists: run -> [ (freqs_x, power_x), (freqs_y, power_y), (freqs_z, power_z) ]

print(f"Processing {num_runs} runs...")

for i in range(num_runs):
    t_eng = eng_times[i]
    rel = rel_pos_B[i]
    
    # Mask exclusively the 'Fine Observation' windows
    fine_idx = np.where(phase[i] == 'Fine Observation')[0]
    if len(fine_idx) > 0:
        fine_start, fine_end = full_times[i][fine_idx[0]], full_times[i][fine_idx[-1]]
        fine_mask = (t_eng >= fine_start) & (t_eng <= fine_end)
    else:
        fine_mask = np.zeros_like(t_eng, dtype=bool)

    fine_t = t_eng[fine_mask]
    fine_rel = rel[fine_mask]

    # Shift time to start at 0
    if len(fine_t) > 0:
        fine_t = fine_t - fine_t[0]
    else:
        # Skip runs with no fine observation data
        results_std.append((a_vals[i], i_vals[i], omega_vals[i], [np.nan, np.nan, np.nan]))
        results_fft.append([ (None, None), (None, None), (None, None) ])
        continue

    run_stds = []
    run_ffts = []
    for coord in range(len(coord_letter)):
        # get the STD
        coord_std = np.std(fine_rel[:, coord])
        run_stds.append(coord_std)
        # get the fourier analysis
        freqs, power = _computeFFT(fine_rel[:, coord], fine_t)
        run_ffts.append((freqs, power))
    
    results_std.append((a_vals[i], i_vals[i], omega_vals[i], run_stds))
    results_fft.append(run_ffts)

# Organize unique values for the heatmap grid
unique_as = sorted(list(set(a_vals)))
unique_is = sorted(list(set(i_vals)))
unique_omegas = sorted(list(set(omega_vals)))

# Loop through each semi-major axis (a) to create a separate figure
for a_target in unique_as:
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1], figure=fig)
    fig.suptitle(f'Statistical Analysis of Relative Position (a={a_target:.2e})', fontsize=16)

    # Filtering data for this specific 'a'
    indices_this_a = [i for i, val in enumerate(a_vals) if val == a_target]
    
    # Subplots for each coordinate
    for coord in range(len(coord_letter)):
        # --- Left Panel: FFT (Freq vs Power) ---
        ax_fft = fig.add_subplot(gs[coord, 0])
        ax_fft.set_title(f'FFT Analysis: {coord_letter[coord]}-axis')
        
        for idx in indices_this_a:
            freqs, power = results_fft[idx][coord]
            if freqs is not None:
                ax_fft.plot(freqs, power, alpha=0.3)
        
        ax_fft.set_yscale('log')
        ax_fft.set_xlabel('Frequency [Hz]')
        ax_fft.set_ylabel('Power')
        ax_fft.grid(True, which='both', linestyle='--', alpha=0.5)

        # --- Right Panel: Heatmap of STD vs Inclination & Omega ---
        ax_heat = fig.add_subplot(gs[coord, 1])
        
        # Build the grid
        std_grid = np.zeros((len(unique_is), len(unique_omegas)))
        std_grid[:] = np.nan
        
        for idx in indices_this_a:
            cur_i = i_vals[idx]
            cur_omega = omega_vals[idx]
            cur_std = results_std[idx][3][coord]
            
            row = unique_is.index(cur_i)
            col = unique_omegas.index(cur_omega)
            std_grid[row, col] = cur_std
        
        im = ax_heat.imshow(std_grid, origin='lower', aspect='auto',
                            extent=[min(unique_omegas), max(unique_omegas), min(unique_is), max(unique_is)],
                            cmap='viridis')
        
        # Adjust ticks to match categories
        ax_heat.set_xticks(unique_omegas)
        ax_heat.set_yticks(unique_is)
        ax_heat.set_title(f'STD Heatmap: {coord_letter[coord]}-axis (Inclination vs Omega)')
        ax_heat.set_xlabel('Omega [deg]')
        ax_heat.set_ylabel('Inclination [deg]')
        
        cbar = fig.colorbar(im, ax=ax_heat)
        cbar.set_label('Standard Deviation [m]')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show() # or plt.savefig(...)
