"""
ROMSA v1.1: Principal Stress Orientations from Faults
=====================================================
Original Algorithm (C++): Bruno Ciscato (1994)
Methodology Reference:    Lisle (1987)
Python Implementation:    2025

DESCRIPTION:
This program calculates the orientation of principal stress axes (σ1, σ2, σ3) 
compatible with a population of fault-slip data using a grid-search 
probability method. It determines the optimal tensor using a robust 
barycenter calculation of the high-probability plateau.

USAGE:
    python romsa.py <filename> [options]

ARGUMENTS:
    filename       Path to the input .dat file containing fault data.

OPTIONS:
    --res {low, medium, high}   
                   Set grid resolution (default: medium).
                   - low:    ~31k points (Fast preview)
                   - medium: ~125k points (Standard)
                   - high:   ~500k points (Publication quality)

    --cmap {Inferno, Viridis, Greys, Blues}
                   Set the default color palette for the probability heatmap.

VISUALIZATION OVERLAYS (Auto-save & Startup):
    --faults       Overlay fault plane traces (Great Circles).
    --striae       Overlay striae (Slickensides) as dots.
    --axes         Overlay the best-fit stress axes (σ1, σ2, σ3).

INPUT FILE FORMAT (.dat):
    A whitespace-separated text file. Each row represents one fault:
    [Dip] [DipDir] [Plunge] [PlungeDir] [Normal/Reverse] [Dextral/Sinistral]
    
    * Angles in degrees.
    * Normal/Reverse: 1 = Normal, -1 = Reverse, 0 = Vertical/Undetermined
    * Dex/Sin:        1 = Dextral, -1 = Sinistral

OUTPUTS:
    1. <filename>_tensors.csv : Ranked list of compatible tensors.
    2. <filename>_plot.png    : High-res plot (saved before UI renders).
    3. Interactive Window     : with mouse-over Trend/Plunge and UI controls.
"""

import os
import sys
import time
import csv
import argparse
from typing import Tuple, Optional, List

# --- Third-Party Imports ---
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.collections import LineCollection
# Numba is used for Just-In-Time compilation to speed up the heavy math loops
from numba import njit, prange

# --- CONSTANTS & CONFIGURATION ---

# Conversion factors
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

# Algorithm settings
SCAN_ANGLE_STEP = 1  # Step size (degrees) for rotating Sigma 3 around Sigma 1
BATCH_SIZE = 1000    # Number of grid points to process in one parallel batch

# Grid Resolution Settings (Grid Step Size)
# 'medium' is the default and offers a good balance of speed vs precision
RES_MAP = {
    'low': 0.01,     # ~31,000 points
    'medium': 0.005, # ~125,000 points
    'high': 0.0025   # ~500,000 points
}

# Color palettes for the probability heatmap
CMAP_OPTIONS = {
    'Inferno': 'inferno',
    'Viridis': 'viridis',
    'Greys': 'Greys',
    'Blues': 'Blues'
}

# --- HELPER CLASSES ---

class ProgressBar:
    """
    A minimal, dependency-free progress bar for CLI feedback.
    Displays percentage and Estimated Time of Arrival (ETA).
    """
    def __init__(self, total: int, prefix: str = '', suffix: str = '', 
                 decimals: int = 1, length: int = 40, fill: str = '█'):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.start_time = time.time()

    def update(self, iteration: int):
        """Updates the visual state of the progress bar."""
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (iteration / float(self.total)))
        filled_length = int(self.length * iteration // self.total)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)
        
        elapsed = time.time() - self.start_time
        if iteration > 0:
            rate = iteration / elapsed
            remaining = (self.total - iteration) / rate
            eta = f"{remaining:.0f}s"
        else:
            eta = "?s"

        sys.stdout.write(f'\r{self.prefix} |{bar}| {percent}% {self.suffix} [ETA: {eta}]')
        if iteration == self.total:
            sys.stdout.write('\n')
        sys.stdout.flush()

# --- MATH KERNELS (JIT COMPILED) ---

@njit(inline='always')
def fast_dot(a: np.ndarray, b: np.ndarray) -> float:
    """
    Optimized dot product for 3D vectors.
    Explicit calculation avoids NumPy overhead for small vectors.
    """
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

@njit(fastmath=True)
def rodrigues_rotate(v: np.ndarray, k: np.ndarray, theta: float) -> np.ndarray:
    """
    Rotates vector 'v' around axis 'k' by angle 'theta' using Rodrigues' formula.
    Used to scan possible Sigma 3 orientations around a fixed Sigma 1.
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    k_cross_v = np.cross(k, v)
    k_dot_v = fast_dot(k, v)
    return v * cos_t + k_cross_v * sin_t + k * k_dot_v * (1.0 - cos_t)

@njit
def dir_cos(plunge: float, trend: float) -> np.ndarray:
    """
    Converts geological Trend/Plunge (radians) to a 3D Unit Vector.
    Coordinate System: [North, East, Down]
    """
    l = np.cos(plunge) * np.cos(trend) # North component
    m = np.cos(plunge) * np.sin(trend) # East component
    n = np.sin(plunge)                 # Down component
    return np.array([l, m, n], dtype=np.float32)

@njit(parallel=True, fastmath=True)
def calculate_batch(grid_vecs: np.ndarray, 
                    normals: np.ndarray, 
                    striae: np.ndarray, 
                    o_axes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    The Core ROMSA Algorithm.
    Calculates the probability for a batch of potential Sigma 1 orientations.
    
    Logic based on Lisle (1987):
    1. Check if the test vector is compatible with the fault movement (P1).
    2. Scan all possible Sigma 3 orientations (perpendicular to Sigma 1).
    3. Calculate P2 (Standard Dihedra) and P3 (Lisle's Quadrant Check).
    """
    n_grid = grid_vecs.shape[0]
    n_faults = normals.shape[0]
    
    batch_probs = np.zeros(n_grid, dtype=np.float32)
    batch_s3 = np.zeros((n_grid, 3), dtype=np.float32)
    
    for g in prange(n_grid):
        giv_dir = grid_vecs[g] # The candidate Sigma 1 vector
        
        # --- P1: Sigma 1 Compatibility Check ---
        num_fault_p1 = 0
        for i in range(n_faults):
            # If Normal and Striae dot products have same sign, S1 is in valid quadrant
            if fast_dot(normals[i], giv_dir) * fast_dot(striae[i], giv_dir) >= 0:
                num_fault_p1 += 1
        
        p1 = num_fault_p1 / n_faults
        if p1 == 0: continue # Optimization: If P1 is 0, total prob is 0.

        # --- Sigma 3 Scan ---
        # Find an arbitrary starting vector perpendicular to Sigma 1
        temp_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if np.abs(fast_dot(giv_dir, temp_up)) > 0.99:
             temp_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        s3_raw = np.cross(giv_dir, temp_up)
        s3_start = (s3_raw / np.sqrt(fast_dot(s3_raw, s3_raw))).astype(np.float32)
        
        p23_max = 0.0
        best_s3_local = s3_start 
        
        # Rotate S3 around S1 in steps
        for angle_deg in range(0, 180, SCAN_ANGLE_STEP):
            s3_rotated = rodrigues_rotate(s3_start, giv_dir, angle_deg * DEG2RAD)
            s3_current = s3_rotated.astype(np.float32)
            
            num_fault_p2 = 0
            num_fault_p3 = 0
            
            for j in range(n_faults):
                stri_dot_s3 = fast_dot(striae[j], s3_current)
                
                # --- P2: Sigma 3 Compatibility Check ---
                # S3 must be in the extensive quadrant (opposite signs)
                if fast_dot(normals[j], s3_current) * stri_dot_s3 < 0:
                    num_fault_p2 += 1
                
                # --- P3: Lisle's Criterion (Mechanical Compatibility) ---
                check_s1 = (fast_dot(o_axes[j], giv_dir) * fast_dot(striae[j], giv_dir)) >= 0
                check_s3 = (fast_dot(o_axes[j], s3_current) * stri_dot_s3) >= 0
                
                # Valid if indices mismatch
                if check_s1 != check_s3:
                    num_fault_p3 += 1
            
            p2 = num_fault_p2 / n_faults
            p3 = num_fault_p3 / n_faults
            
            current_p23 = p2 * p3
            if current_p23 > p23_max:
                p23_max = current_p23
                best_s3_local = s3_current
        
        batch_probs[g] = p1 * p23_max
        batch_s3[g] = best_s3_local

    return batch_probs, batch_s3

# --- GEOMETRY & PROJECTION HELPERS ---

def project_vector(trend_rad, plunge_rad):
    """
    Projects 3D geological angles to 2D Equal Area coordinates (Lambert).
    Used for plotting points on the Stereonet.
    """
    # Formula: R = sqrt(2) * sin((90 - plunge)/2)
    r = np.sqrt(2) * np.sin((np.pi/2 - plunge_rad) / 2.0)
    x = r * np.sin(trend_rad) # East
    y = r * np.cos(trend_rad) # North
    return x, y

def get_great_circle(dip_dir_rad, dip_rad):
    """
    Generates (x, y) coordinates for a Great Circle (Fault Plane trace).
    Calculates the arc in 3D vector space [North, East, Down] then projects it.
    """
    # 1. Calculate Basis Vectors for the plane
    strike_az = dip_dir_rad - (np.pi / 2.0)
    v_strike = np.array([np.cos(strike_az), np.sin(strike_az), 0.0])
    
    v_dip = np.array([
        np.cos(dip_rad) * np.cos(dip_dir_rad), # North
        np.cos(dip_rad) * np.sin(dip_dir_rad), # East
        np.sin(dip_rad)                        # Down
    ])
    
    # 2. Generate arc points
    theta = np.linspace(0, np.pi, 90) # 0 to 180 degrees
    pts_x = []
    pts_y = []
    
    for t in theta:
        v = v_strike * np.cos(t) + v_dip * np.sin(t)
        if v[2] < 0: v = -v # Force to lower hemisphere
        
        plunge = np.arcsin(np.clip(v[2], -1, 1))
        trend = np.arctan2(v[1], v[0]) 
        
        px, py = project_vector(trend, plunge)
        pts_x.append(px)
        pts_y.append(py)
        
    return np.array(pts_x), np.array(pts_y)

def calculate_barycenter(probs, vec_grid, s3_grid, max_p):
    """
    Calculates the 'Barycenter' (Weighted Average) of the best solutions.
    Instead of taking the single highest pixel, this averages all vectors
    within the top 3% of the maximum probability (the "Plateau").
    This provides a statistically robust result less prone to grid artifacts.
    """
    threshold = max_p * 0.97
    indices = np.where(probs >= threshold)[0]
    
    # Fallback if only 1 point is found
    if len(indices) == 0: 
        idx = np.argmax(probs)
        return vec_grid[idx], np.cross(vec_grid[idx], s3_grid[idx]), s3_grid[idx]

    # 1. Barycenter of Sigma 1
    s1_sum = np.zeros(3)
    for idx in indices:
        s1_sum += vec_grid[idx] * probs[idx]
    
    if np.linalg.norm(s1_sum) < 1e-9: mean_s1 = vec_grid[indices[0]]
    else: mean_s1 = s1_sum / np.linalg.norm(s1_sum)

    # 2. Barycenter of Sigma 3
    # (Note: Align S3 vectors to ensure they don't cancel out due to polarity)
    s3_sum = np.zeros(3)
    ref_s3 = s3_grid[indices[0]] 
    for idx in indices:
        curr_s3 = s3_grid[idx]
        if fast_dot(curr_s3, ref_s3) < 0: curr_s3 = -curr_s3
        s3_sum += curr_s3 * probs[idx]
        
    if np.linalg.norm(s3_sum) < 1e-9: mean_s3 = s3_grid[indices[0]]
    else: mean_s3 = s3_sum / np.linalg.norm(s3_sum)

    # 3. Orthogonalize (Recalculate S2 and refine S3)
    mean_s2 = np.cross(mean_s1, mean_s3)
    mean_s3 = np.cross(mean_s2, mean_s1)
    
    return mean_s1, mean_s2, mean_s3

# --- IO & UTILS ---

def load_data(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parses input .dat file.
    Robustness: Checks for a header count but prioritizes actual data found.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The input file '{filename}' was not found.")

    with open(filename, 'r') as f:
        # Clean delimiters (commas/tabs to spaces)
        tokens = f.read().replace('"', ' ').replace(',', ' ').split()
    
    if not tokens: raise ValueError("File is empty.")

    total_tokens = len(tokens)
    
    # Heuristic to detect if the first number is a header count
    if total_tokens % 6 == 1:
        n_data = int(tokens[0])
        data_tokens = tokens[1:]
    elif total_tokens % 6 == 0:
        n_data = total_tokens // 6
        data_tokens = tokens
    else:
        raise ValueError(f"File contains {total_tokens} values. Expected a multiple of 6.")

    normals = np.zeros((n_data, 3), dtype=np.float32)
    striae = np.zeros((n_data, 3), dtype=np.float32)
    o_axes = np.zeros((n_data, 3), dtype=np.float32)
    
    idx = 0
    for i in range(n_data):
        # Format: Dip, DipDir, Plunge, PlungeDir, NormRev, DexSin
        dip, dip_dir, plunge, plunge_dir, norm_rev, dex_sin = map(float, data_tokens[idx:idx+6])
        idx += 6
        
        # Handle "Pitch" notation if PlungeDir is not explicit
        if norm_rev == 0:
            plunge_dir = dip_dir + 90.0 * dex_sin
            norm_rev = 1
            
        n_vec = dir_cos((90.0 - dip) * DEG2RAD, (dip_dir + 180.0) * DEG2RAD)
        s_vec = dir_cos(plunge * DEG2RAD, plunge_dir * DEG2RAD)
        s_vec = s_vec * norm_rev
        o_vec = np.cross(s_vec, n_vec) # Orthogonal axis
        
        normals[i], striae[i], o_axes[i] = n_vec, s_vec, o_vec
        
    return normals, striae, o_axes

def vec_to_geology(vec: np.ndarray) -> Tuple[int, int]:
    """Converts a 3D vector [N, E, D] back to Trend and Plunge (degrees)."""
    # Ensure vector points down for reporting
    if vec[2] < 0: vec = -vec
    
    n = np.clip(vec[2], -1.0, 1.0)
    plunge = np.arcsin(n)
    trend = np.arctan2(vec[1], vec[0]) 
    
    plunge_deg = plunge * RAD2DEG
    trend_deg = trend * RAD2DEG
    
    if trend_deg < 0: trend_deg += 360.0
    
    return int(round(trend_deg)), int(round(plunge_deg))

# --- MAIN LOGIC ---

def main():
    # 1. Argument Parsing
    parser = argparse.ArgumentParser(description='ROMSA: Paleostress Analysis (Python)')
    parser.add_argument('filename', type=str, help='Path to input .dat file')
    parser.add_argument('--res', choices=['low', 'medium', 'high'], default='medium', help='Grid resolution')
    parser.add_argument('--cmap', choices=CMAP_OPTIONS.keys(), default='Inferno', help='Default color palette')
    
    # Toggle Flags for Auto-Save
    parser.add_argument('--faults', action='store_true', help='Overlay fault planes on start')
    parser.add_argument('--striae', action='store_true', help='Overlay striae on start')
    parser.add_argument('--axes', action='store_true', help='Overlay stress axes on start')
    
    args = parser.parse_args()
    input_file = args.filename
    resolution_step = RES_MAP[args.res]
    selected_cmap_name = CMAP_OPTIONS[args.cmap]
    
    abs_input_path = os.path.abspath(input_file)
    work_dir = os.path.dirname(abs_input_path)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    plt.rcParams["savefig.directory"] = work_dir
    csv_filename = os.path.join(work_dir, f"{base_name}_tensors.csv")
    plot_filename = os.path.join(work_dir, f"{base_name}_plot.png")
    
    # 2. Startup Feedback
    print(f"\n--- ROMSA: Paleostress Analysis ---")
    print(f"Input File:  {input_file}")
    print(f"Resolution:  {args.res.upper()} (Step: {resolution_step})")
    print(f"Palette:     {args.cmap}")
    print(f"Overlays:    Faults=[{args.faults}], Striae=[{args.striae}], Axes=[{args.axes}]")
    
    # 3. Load Data
    try:
        normals, striae, o_axes = load_data(input_file)
    except Exception as e:
        print(f"\n[ERROR] {e}"); return

    print(f"Loaded {len(normals)} faults.")
    
    # 4. Grid Generation
    print("Generating search grid...")
    x_range = np.arange(-1.0, 1.0 + resolution_step, resolution_step)
    xx, yy = np.meshgrid(x_range, x_range)
    flat_x, flat_y = xx.ravel(), yy.ravel()
    mask = (flat_x**2 + flat_y**2) <= 1.0 # Circle mask
    valid_x, valid_y = flat_x[mask], flat_y[mask]
    
    n_points = len(valid_x)
    grid_vecs = np.zeros((n_points, 3), dtype=np.float32)
    for i in range(n_points):
        x, y = valid_x[i], valid_y[i]
        r = np.sqrt(x*x + y*y)
        # Stereographic Reverse Projection
        if r < 1e-6: p, t = np.pi/2, 0.0
        else:
            p = (np.pi/2) - 2.0 * np.arcsin(r * 0.707106781)
            t = np.arctan2(x, y) 
            if t < 0: t += 2*np.pi
        grid_vecs[i] = dir_cos(p, t)

    # 5. Calculation Loop (Parallel JIT)
    print(f"Evaluating {n_points} orientations...")
    final_probs = np.zeros(n_points, dtype=np.float32)
    final_s3 = np.zeros((n_points, 3), dtype=np.float32)
    
    print("Compiling JIT kernels (Warmup)...")
    _ = calculate_batch(grid_vecs[0:1], normals, striae, o_axes)

    pb = ProgressBar(n_points, prefix='Progress:', suffix='Complete', length=40)
    for start_idx in range(0, n_points, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, n_points)
        batch_vecs = grid_vecs[start_idx:end_idx]
        b_probs, b_s3 = calculate_batch(batch_vecs, normals, striae, o_axes)
        final_probs[start_idx:end_idx] = b_probs
        final_s3[start_idx:end_idx] = b_s3
        pb.update(end_idx)

    max_prob = np.max(final_probs)
    print(f"\nCalculation Done. Max Probability: {max_prob*100:.2f}%")
    
    # 6. Barycenter (Refinement)
    print("Calculating barycenter of high-probability plateau...")
    b_s1_vec, b_s2_vec, b_s3_vec = calculate_barycenter(final_probs, grid_vecs, final_s3, max_prob)
    
    # Force lower hemisphere for consistent reporting
    if b_s1_vec[2] < 0: b_s1_vec = -b_s1_vec
    if b_s2_vec[2] < 0: b_s2_vec = -b_s2_vec
    if b_s3_vec[2] < 0: b_s3_vec = -b_s3_vec
    
    b_s1_d, b_s1_p = vec_to_geology(b_s1_vec)
    b_s2_d, b_s2_p = vec_to_geology(b_s2_vec)
    b_s3_d, b_s3_p = vec_to_geology(b_s3_vec)

    # 7. CSV Export
    print(f"Exporting tensors to '{csv_filename}'...")
    indices = np.where(final_probs >= max_prob * 0.95)[0]
    sorted_indices = indices[np.argsort(-final_probs[indices])]

    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Prob_Percent", "S1_Dir", "S1_Dip", "S2_Dir", "S2_Dip", "S3_Dir", "S3_Dip"])
        for idx in sorted_indices:
            p_val = final_probs[idx]
            s1, s3 = grid_vecs[idx], final_s3[idx]
            s2 = np.cross(s1, s3)
            writer.writerow([f"{p_val*100:.1f}", *vec_to_geology(s1), *vec_to_geology(s2), *vec_to_geology(s3)])

    # 8. Visualization Setup
    print("Generating interactive plot...")
    grid_matrix = np.zeros_like(xx, dtype=np.float32)
    grid_matrix.fill(np.nan)
    grid_matrix.ravel()[mask] = final_probs * 100
    
    fig = plt.figure(figsize=(14, 10))
    fig.subplots_adjust(left=0.05, right=0.92, top=0.90, bottom=0.15, wspace=0.3)
    
    gs = fig.add_gridspec(1, 3, width_ratios=[3, 6, 0.3])
    ax_info, ax_net, ax_cbar = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    ax_info.axis('off'); ax_net.axis('off'); ax_net.set_aspect('equal')
    ax_net.add_artist(plt.Circle((0, 0), 1, color='k', fill=False, linewidth=2))
    
    # Draw Contour (100 levels for continuous look)
    levels = np.linspace(0, 100, 101) 
    contour = ax_net.contourf(xx, yy, grid_matrix, levels=levels, cmap=selected_cmap_name, vmin=0, vmax=100)
    
    # Cardinals
    ax_net.text(0, 1.12, "N", ha='center', fontsize=12, fontweight='bold')
    ax_net.text(0, -1.12, "S", ha='center', fontsize=12, fontweight='bold')
    ax_net.text(1.12, 0, "E", va='center', fontsize=12, fontweight='bold')
    ax_net.text(-1.12, 0, "W", va='center', fontsize=12, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(contour, cax=ax_cbar, ticks=np.linspace(0, 100, 11)) 
    cbar.set_label(r'Probability of $\sigma_1$ (%)', fontsize=11)
    ax_cbar.yaxis.set_ticks_position('right')

    # --- Draw Text Panel ---
    t = ax_info.transAxes
    f_head = {'family':'sans-serif', 'weight':'bold', 'size':14, 'color':'#333333'}
    f_body = {'family':'sans-serif', 'weight':'normal', 'size':12, 'color':'#444444'}
    f_mono = {'family':'monospace', 'weight':'bold', 'size':12, 'color':'#000000'}

    ax_info.text(0, 1.0, "ROMSA", fontdict={'family':'sans-serif','weight':'bold','size':24}, va='top', transform=t)
    ax_info.text(0, 0.94, "Paleostress Analysis", fontdict={'family':'sans-serif','size':14,'color':'#666666'}, va='top', transform=t)
    ax_info.plot([0, 1], [0.90, 0.90], color='#aaaaaa', linewidth=1, transform=t)
    
    ax_info.text(0, 0.85, "DATASET", fontdict=f_head, va='top', transform=t)
    ax_info.text(0, 0.81, f"File: {base_name}", fontdict=f_body, va='top', transform=t)
    ax_info.text(0, 0.77, f"Faults: {len(normals)}", fontdict=f_body, va='top', transform=t)
    ax_info.text(0, 0.65, "BARYCENTER (TOP 3%)", fontdict=f_head, va='top', transform=t)
    ax_info.text(0, 0.61, f"Max Prob: {max_prob*100:.1f}%", fontdict=f_body, va='top', transform=t)
    
    y = 0.53; h = 0.05
    ax_info.text(0, y, "Axis", fontdict=f_head, va='top', transform=t)
    ax_info.text(0.3, y, "Trend / Plunge", fontdict=f_head, va='top', transform=t)
    
    ax_info.text(0, y-h, "σ1", fontdict=f_mono, color='#cc3300', va='top', transform=t)
    ax_info.text(0.3, y-h, f"{b_s1_d:03d} / {b_s1_p:02d}", fontdict=f_mono, va='top', transform=t)
    
    ax_info.text(0, y-2*h, "σ2", fontdict=f_mono, color='black', va='top', transform=t)
    ax_info.text(0.3, y-2*h, f"{b_s2_d:03d} / {b_s2_p:02d}", fontdict=f_mono, va='top', transform=t)
    
    ax_info.text(0, y-3*h, "σ3", fontdict=f_mono, color='#0033cc', va='top', transform=t)
    ax_info.text(0.3, y-3*h, f"{b_s3_d:03d} / {b_s3_p:02d}", fontdict=f_mono, va='top', transform=t)
    
    ax_info.text(0, 0.02, "Generated by ROMSA-Py", fontdict={'size':9,'color':'#999999'}, va='bottom', transform=t)

    # --- 9. Draw Overlays (Hidden or Visible based on CLI) ---
    fault_lines = []
    for i in range(len(normals)):
        n = normals[i]
        pole_plunge = np.arcsin(n[2])
        pole_trend = np.arctan2(n[1], n[0])
        dip_rad = (np.pi/2) - pole_plunge
        dip_dir_rad = pole_trend - np.pi 
        gx, gy = get_great_circle(dip_dir_rad, dip_rad)
        fault_lines.append(np.column_stack([gx, gy]))

    lc_faults = LineCollection(fault_lines, colors='#555555', linewidths=0.8, alpha=0.6, visible=args.faults)
    ax_net.add_collection(lc_faults)

    striae_x, striae_y = [], []
    for s in striae:
        if s[2] < 0: s_plot = -s 
        else: s_plot = s
        plunge = np.arcsin(s_plot[2])
        trend = np.arctan2(s_plot[1], s_plot[0]) 
        px, py = project_vector(trend, plunge)
        striae_x.append(px)
        striae_y.append(py)
    
    sc_striae = ax_net.scatter(striae_x, striae_y, c='white', edgecolors='black', s=30, linewidth=0.8, zorder=5, visible=args.striae)

    t1, p1 = np.arctan2(b_s1_vec[1], b_s1_vec[0]), np.arcsin(b_s1_vec[2]) 
    t2, p2 = np.arctan2(b_s2_vec[1], b_s2_vec[0]), np.arcsin(b_s2_vec[2])
    t3, p3 = np.arctan2(b_s3_vec[1], b_s3_vec[0]), np.arcsin(b_s3_vec[2])
    x1, y1 = project_vector(t1, p1); x2, y2 = project_vector(t2, p2); x3, y3 = project_vector(t3, p3)

    sc_s1 = ax_net.scatter([x1], [y1], c='#cc3300', s=150, edgecolors='white', label='S1', zorder=10, visible=args.axes)
    sc_s2 = ax_net.scatter([x2], [y2], c='black',   s=100, edgecolors='white', label='S2', zorder=10, visible=args.axes)
    sc_s3 = ax_net.scatter([x3], [y3], c='#0033cc', s=150, edgecolors='white', label='S3', zorder=10, visible=args.axes)

    # --- 10. Auto-Save (Clean Image) ---
    print(f"Auto-saving plot to '{plot_filename}'...")
    plt.savefig(plot_filename, dpi=300, facecolor='white')

    # --- 11. UI Buttons & Hover Logic ---
    annot = ax_net.text(0, 0, "", ha='center', va='center', bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.9), visible=False, zorder=20)
    
    def on_hover(event):
        if event.inaxes == ax_net:
            mx, my = event.xdata, event.ydata
            r = np.sqrt(mx**2 + my**2)
            if r <= 1.0:
                ang = np.arcsin(r / 1.41421356)
                plunge_deg = 90.0 - np.degrees(2.0 * ang)
                trend_rad = np.arctan2(mx, my) 
                trend_deg = np.degrees(trend_rad)
                if trend_deg < 0: trend_deg += 360
                annot.set_text(f"{int(trend_deg):03d}/{int(plunge_deg):02d}")
                annot.set_position((mx + 0.1, my + 0.1))
                annot.set_visible(True)
                fig.canvas.draw_idle(); return
        if annot.get_visible(): annot.set_visible(False); fig.canvas.draw_idle()
    fig.canvas.mpl_connect("motion_notify_event", on_hover)

    cmap_buttons = {}
    toggle_buttons = {}
    
    def make_cmap_cb(label):
        def on_click(event):
            contour.set_cmap(plt.get_cmap(CMAP_OPTIONS[label]))
            for lbl, b in cmap_buttons.items():
                b.color = '0.70' if lbl == label else '0.95'
                b.ax.set_facecolor(b.color) 
            fig.canvas.draw_idle()
        return on_click

    def make_toggle_cb(label):
        def on_click(event):
            target = toggles[label]
            is_list = isinstance(target, list)
            curr = target[0].get_visible() if is_list else target.get_visible()
            new_state = not curr
            if is_list:
                for item in target: item.set_visible(new_state)
            else:
                target.set_visible(new_state)
            
            toggle_buttons[label].color = '0.70' if new_state else '0.95'
            toggle_buttons[label].hovercolor = '0.60' if new_state else '0.90'
            toggle_buttons[label].ax.set_facecolor(toggle_buttons[label].color)
            fig.canvas.draw_idle()
        return on_click

    # Draw Palettes (Row 1 - Bottom)
    x_pos = 0.05
    btn_w = 0.06
    btn_h = 0.04
    gap = 0.01
    for label in CMAP_OPTIONS.keys():
        ax_btn = plt.axes([x_pos, 0.03, btn_w, btn_h])
        is_active = (label == args.cmap)
        b = Button(ax_btn, label, color='0.70' if is_active else '0.95', hovercolor='0.85')
        b.label.set_fontsize(8)
        ax_btn.add_patch(plt.Rectangle((0,0),1,1,transform=ax_btn.transAxes,fill=False,edgecolor='#cccccc'))
        b.on_clicked(make_cmap_cb(label))
        cmap_buttons[label] = b
        x_pos += btn_w + gap

    # Draw Overlays (Row 2 - Top)
    toggles = {'Faults': lc_faults, 'Striae': sc_striae, 'Axes': [sc_s1, sc_s2, sc_s3]}
    toggle_states = {'Faults': args.faults, 'Striae': args.striae, 'Axes': args.axes}
    x_pos = 0.05
    for label in toggles.keys():
        ax_t = plt.axes([x_pos, 0.08, btn_w, btn_h])
        is_active = toggle_states[label]
        b = Button(ax_t, label, color='0.70' if is_active else '0.95', hovercolor='0.85')
        b.label.set_fontsize(8)
        b.on_clicked(make_toggle_cb(label))
        ax_t.add_patch(plt.Rectangle((0,0),1,1,transform=ax_t.transAxes,fill=False,edgecolor='#cccccc'))
        toggle_buttons[label] = b
        x_pos += btn_w + gap

    plt.show()

if __name__ == "__main__":
    main()
