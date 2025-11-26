"""
ROMSA: Principal Stress Orientations from Faults
================================================
Original Algorithm (C++): Bruno Ciscato (1994)
Based on method by: Lisle (1987)
Modern Python Implementation: 2025

This program calculates the orientation of the principal stress axes 
compatible with a population of fault-slip data using a grid-search 
probability method.
"""

import os
import sys
import time
import csv
import argparse
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from numba import njit, prange

# --- CONSTANTS & CONFIGURATION ---
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
SCAN_ANGLE_STEP = 1  # degrees
BATCH_SIZE = 1000    # Points to process per JIT batch

# Resolution settings (Grid Step Size)
RES_MAP = {
    'low': 0.01,     # ~31k points (Fast preview)
    'medium': 0.005, # ~125k points (Standard)
    'high': 0.0025   # ~500k points (Publication quality)
}

# --- HELPER CLASSES ---

class ProgressBar:
    """A minimal, dependency-free progress bar for CLI feedback."""
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
        """Updates the progress bar state."""
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
    Bypasses Numpy overhead for small arrays and handles mixed precision.
    """
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

@njit(fastmath=True)
def rodrigues_rotate(v: np.ndarray, k: np.ndarray, theta: float) -> np.ndarray:
    """
    Rotates vector v around axis k by angle theta using Rodrigues' formula.
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
    """
    l = np.cos(plunge) * np.cos(trend)
    m = np.cos(plunge) * np.sin(trend)
    n = np.sin(plunge)
    return np.array([l, m, n], dtype=np.float32)

@njit(parallel=True, fastmath=True)
def calculate_batch(grid_vecs: np.ndarray, 
                    normals: np.ndarray, 
                    striae: np.ndarray, 
                    o_axes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Core calculation kernel.
    Evaluates probability of Sigma 1 for a batch of grid orientations.
    
    Returns:
        batch_probs: Array of probability values (0.0 to 1.0)
        batch_s3: Array of best-fit Sigma 3 vectors for each grid point
    """
    n_grid = grid_vecs.shape[0]
    n_faults = normals.shape[0]
    
    batch_probs = np.zeros(n_grid, dtype=np.float32)
    batch_s3 = np.zeros((n_grid, 3), dtype=np.float32)
    
    # Parallel loop over grid points
    for g in prange(n_grid):
        giv_dir = grid_vecs[g] # Candidate Sigma 1
        
        # --- Step 1: P1 (Sigma 1 Dihedra Check) ---
        num_fault_p1 = 0
        for i in range(n_faults):
            norm_ang = fast_dot(normals[i], giv_dir)
            striae_ang = fast_dot(striae[i], giv_dir)
            if (norm_ang * striae_ang) >= 0:
                num_fault_p1 += 1
        
        p1 = num_fault_p1 / n_faults
        if p1 == 0: continue

        # --- Step 2: Scan for Best Sigma 3 (Lisle Constraint) ---
        
        # Find a perpendicular starting vector for S3
        temp_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if np.abs(fast_dot(giv_dir, temp_up)) > 0.99:
             temp_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        s3_raw = np.cross(giv_dir, temp_up)
        # Normalize and cast to float32 to prevent type drift in rotation loop
        s3_start = (s3_raw / np.sqrt(fast_dot(s3_raw, s3_raw))).astype(np.float32)
        
        p23_max = 0.0
        best_s3_local = s3_start 
        
        # Rotate S3 around Sigma 1 (0 to 180 degrees)
        for angle_deg in range(0, 180, SCAN_ANGLE_STEP):
            # Rotate
            s3_rotated = rodrigues_rotate(s3_start, giv_dir, angle_deg * DEG2RAD)
            s3_current = s3_rotated.astype(np.float32)
            
            num_fault_p2 = 0
            num_fault_p3 = 0
            
            for j in range(n_faults):
                # P2: S3 Dihedra Check
                norm_dot_s3 = fast_dot(normals[j], s3_current)
                stri_dot_s3 = fast_dot(striae[j], s3_current)
                
                if (norm_dot_s3 * stri_dot_s3) < 0:
                    num_fault_p2 += 1
                
                # P3: Lisle's A/B Quadrant Check
                o_dot_s1 = fast_dot(o_axes[j], giv_dir)
                s_dot_s1 = fast_dot(striae[j], giv_dir)
                o_dot_s3 = fast_dot(o_axes[j], s3_current)
                # s_dot_s3 is effectively stri_dot_s3
                
                check_s1 = (o_dot_s1 * s_dot_s1) >= 0
                check_s3 = (o_dot_s3 * stri_dot_s3) >= 0
                
                # If they lie in different pairs, the constraint is met
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

# --- IO & UTILS ---

def load_data(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parses input file. Automatically detects if header count is present or missing.
    Returns Tuple of (Normals, Striae, O-Axes).
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The input file '{filename}' was not found.")

    with open(filename, 'r') as f:
        raw_content = f.read()
    
    # Normalize delimiters
    clean_content = raw_content.replace('"', ' ').replace(',', ' ')
    tokens = clean_content.split()
    
    if not tokens: 
        raise ValueError("File is empty.")

    # Auto-detect format (Header vs No Header)
    total_tokens = len(tokens)
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
        # Parse row
        dip = float(data_tokens[idx]); idx += 1
        dip_dir = float(data_tokens[idx]); idx += 1
        plunge = float(data_tokens[idx]); idx += 1
        plunge_dir = float(data_tokens[idx]); idx += 1
        norm_rev = int(float(data_tokens[idx])); idx += 1
        dex_sin = int(float(data_tokens[idx])); idx += 1
        
        # Handle pure strike-slip notation
        if norm_rev == 0:
            plunge_dir = dip_dir + 90.0 * dex_sin
            norm_rev = 1
            
        # Convert to Direction Cosines (Vectors)
        n_vec = dir_cos((90.0 - dip) * DEG2RAD, (dip_dir + 180.0) * DEG2RAD)
        s_vec = dir_cos(plunge * DEG2RAD, plunge_dir * DEG2RAD)
        s_vec = s_vec * norm_rev
        o_vec = np.cross(s_vec, n_vec)
        
        normals[i] = n_vec
        striae[i] = s_vec
        o_axes[i] = o_vec
        
    return normals, striae, o_axes

def vec_to_geology(vec: np.ndarray) -> Tuple[int, int]:
    """Converts 3D vector back to Trend/Plunge (Degrees)."""
    n = np.clip(vec[2], -1.0, 1.0)
    plunge = np.arcsin(n)
    trend = np.arctan2(vec[1], vec[0])
    
    plunge_deg = plunge * RAD2DEG
    trend_deg = trend * RAD2DEG
    
    if trend_deg < 0:
        trend_deg += 360.0
    if plunge_deg < 0:
        plunge_deg = -plunge_deg
        trend_deg = (trend_deg + 180) % 360.0
        
    return int(round(trend_deg)), int(round(plunge_deg))

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(
        description='ROMSA: Principal Stress Orientations (Python Implementation)',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('filename', type=str, help='Path to input .dat file')
    parser.add_argument('--res', choices=['low', 'medium', 'high'], default='medium', 
                        help='Grid resolution:\n'
                             '  low    (Step 0.01   | ~31k pts)\n'
                             '  medium (Step 0.005  | ~125k pts)\n'
                             '  high   (Step 0.0025 | ~500k pts)')
    
    args = parser.parse_args()
    input_file = args.filename
    resolution_step = RES_MAP[args.res]
    
    # Path configuration
    abs_input_path = os.path.abspath(input_file)
    work_dir = os.path.dirname(abs_input_path)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Matplotlib config: Save to input directory by default
    plt.rcParams["savefig.directory"] = work_dir
    
    csv_filename = os.path.join(work_dir, f"{base_name}_tensors.csv")
    plot_filename = os.path.join(work_dir, f"{base_name}_plot.png")
    
    print(f"\n--- ROMSA: Paleostress Analysis ---")
    print(f"Input File:  {input_file}")
    print(f"Resolution:  {args.res.upper()} (Step: {resolution_step})")

    try:
        normals, striae, o_axes = load_data(input_file)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return

    print(f"Loaded {len(normals)} faults.")
    
    # --- 1. Grid Generation ---
    print("Generating search grid...")
    x_range = np.arange(-1.0, 1.0 + resolution_step, resolution_step)
    y_range = np.arange(-1.0, 1.0 + resolution_step, resolution_step)
    xx, yy = np.meshgrid(x_range, y_range)
    
    flat_x = xx.ravel()
    flat_y = yy.ravel()
    # Mask to keep points inside the circle
    mask = (flat_x**2 + flat_y**2) <= 1.0
    
    valid_x = flat_x[mask]
    valid_y = flat_y[mask]
    n_points = len(valid_x)
    
    # Convert grid points to vectors
    grid_vecs = np.zeros((n_points, 3), dtype=np.float32)
    for i in range(n_points):
        x, y = valid_x[i], valid_y[i]
        r = np.sqrt(x*x + y*y)
        if r < 1e-6:
            plunge, trend = np.pi/2, 0.0
        else:
            plunge = (np.pi/2) - 2.0 * np.arcsin(r * 0.707106781) # 0.707... is 1/sqrt(2)
            trend = np.arctan2(x, y) 
            if trend < 0: trend += 2*np.pi
        grid_vecs[i] = dir_cos(plunge, trend)

    # --- 2. Calculation ---
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

    # --- 3. CSV Export ---
    print(f"Exporting tensors to '{csv_filename}'...")
    threshold = max_prob * 0.95
    indices = np.where(final_probs >= threshold)[0]
    sorted_indices = indices[np.argsort(-final_probs[indices])]
    
    # Retrieve Best Tensor
    best_idx = sorted_indices[0]
    best_s1_vec = grid_vecs[best_idx]
    best_s3_vec = final_s3[best_idx]
    best_s2_vec = np.cross(best_s1_vec, best_s3_vec)
    
    b_s1_d, b_s1_p = vec_to_geology(best_s1_vec)
    b_s2_d, b_s2_p = vec_to_geology(best_s2_vec)
    b_s3_d, b_s3_p = vec_to_geology(best_s3_vec)

    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Prob_Percent", "S1_Dir", "S1_Dip", "S2_Dir", "S2_Dip", "S3_Dir", "S3_Dip"])
        for idx in sorted_indices:
            prob_val = final_probs[idx]
            s1_vec = grid_vecs[idx]
            s3_vec = final_s3[idx]
            s2_vec = np.cross(s1_vec, s3_vec)
            s1_d, s1_p = vec_to_geology(s1_vec)
            s2_d, s2_p = vec_to_geology(s2_vec)
            s3_d, s3_p = vec_to_geology(s3_vec)
            writer.writerow([f"{prob_val*100:.1f}", s1_d, s1_p, s2_d, s2_p, s3_d, s3_p])

    # --- 4. Plotting ---
    print("Generating plot...")
    # Map valid points back to the square grid for contouring
    grid_matrix = np.zeros_like(xx, dtype=np.float32)
    grid_matrix.fill(np.nan)
    grid_matrix.ravel()[mask] = final_probs * 100
    
    fig = plt.figure(figsize=(16, 10))
    
    # Layout: Info(Left), Plot(Center), Colorbar(Right)
    gs = fig.add_gridspec(1, 3, width_ratios=[3, 6, 0.2],
                          left=0.08, right=0.92, top=0.90, bottom=0.10,
                          wspace=0.2)

    ax_info = fig.add_subplot(gs[0])
    ax_net = fig.add_subplot(gs[1])
    ax_cbar = fig.add_subplot(gs[2])

    ax_info.axis('off')
    ax_net.set_aspect('equal')

    # Stereonet
    ax_net.add_artist(plt.Circle((0, 0), 1, color='k', fill=False, linewidth=2))
    contour = ax_net.contourf(xx, yy, grid_matrix, levels=30, cmap='inferno')
    
    ax_net.text(0, 1.12, "N", ha='center', fontsize=12, fontweight='bold')
    ax_net.text(0, -1.12, "S", ha='center', fontsize=12, fontweight='bold')
    ax_net.text(1.12, 0, "E", va='center', fontsize=12, fontweight='bold')
    ax_net.text(-1.12, 0, "W", va='center', fontsize=12, fontweight='bold')
    ax_net.axis('off')

    # Colorbar
    cbar = plt.colorbar(contour, cax=ax_cbar)
    cbar.set_label(r'Probability of $\sigma_1$ (%)', fontsize=11)
    ax_cbar.yaxis.set_ticks_position('right')
    ax_cbar.yaxis.set_label_position('right')

    # Typography
    font_header = {'family': 'sans-serif', 'weight': 'bold', 'size': 14, 'color': '#333333'}
    font_body = {'family': 'sans-serif', 'weight': 'normal', 'size': 12, 'color': '#444444'}
    font_mono = {'family': 'monospace', 'weight': 'bold', 'size': 12, 'color': '#000000'}
    font_body_bold = font_body.copy()
    font_body_bold['weight'] = 'bold'

    # Info Panel Content
    ax_info.text(0, 1.0, "ROMSA", fontdict={'family': 'sans-serif', 'weight': 'bold', 'size': 24, 'color': '#222222'}, va='top', transform=ax_info.transAxes)
    ax_info.text(0, 0.94, "Paleostress Analysis", fontdict={'family': 'sans-serif', 'size': 14, 'color': '#666666'}, va='top', transform=ax_info.transAxes)
    ax_info.plot([0, 1], [0.90, 0.90], color='#aaaaaa', linewidth=1, transform=ax_info.transAxes)

    ax_info.text(0, 0.85, "DATASET", fontdict=font_header, va='top', transform=ax_info.transAxes)
    ax_info.text(0, 0.81, f"File: {base_name}", fontdict=font_body, va='top', transform=ax_info.transAxes)
    ax_info.text(0, 0.77, f"Faults Analyzed: {len(normals)}", fontdict=font_body, va='top', transform=ax_info.transAxes)

    ax_info.text(0, 0.65, "BEST FIT TENSOR", fontdict=font_header, va='top', transform=ax_info.transAxes)
    ax_info.text(0, 0.61, f"Max Probability: {max_prob*100:.1f}%", fontdict=font_body, va='top', transform=ax_info.transAxes)
    
    # Tensor Table
    y_start = 0.53
    line_h = 0.05
    
    ax_info.text(0, y_start, "Axis", fontdict=font_body_bold, va='top', transform=ax_info.transAxes)
    ax_info.text(0.3, y_start, "Trend / Plunge", fontdict=font_body_bold, va='top', transform=ax_info.transAxes)
    
    ax_info.text(0, y_start - line_h, "σ1", fontdict=font_mono, va='top', color='#cc3300', transform=ax_info.transAxes)
    ax_info.text(0.3, y_start - line_h, f"{b_s1_d:03d} / {b_s1_p:02d}", fontdict=font_mono, va='top', transform=ax_info.transAxes)
    
    ax_info.text(0, y_start - 2*line_h, "σ2", fontdict=font_mono, va='top', transform=ax_info.transAxes)
    ax_info.text(0.3, y_start - 2*line_h, f"{b_s2_d:03d} / {b_s2_p:02d}", fontdict=font_mono, va='top', transform=ax_info.transAxes)
    
    ax_info.text(0, y_start - 3*line_h, "σ3", fontdict=font_mono, va='top', color='#0033cc', transform=ax_info.transAxes)
    ax_info.text(0.3, y_start - 3*line_h, f"{b_s3_d:03d} / {b_s3_p:02d}", fontdict=font_mono, va='top', transform=ax_info.transAxes)

    ax_info.text(0, 0.0, "Generated by ROMSA-Py", fontdict={'size': 9, 'color': '#999999'}, va='bottom', transform=ax_info.transAxes)

    # Auto-save (Fixed: Removed bbox_inches='tight' to respect GridSpec margins)
    print(f"Auto-saving plot to '{plot_filename}'...")
    plt.savefig(plot_filename, dpi=300, facecolor='white')

    plt.show()

if __name__ == "__main__":
    main()
