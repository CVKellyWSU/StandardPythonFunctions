import os
import glob
import math
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tif
from phasorpy.io import signal_from_ptu
from phasorpy.phasor import phasor_from_signal, phasor_transform
from phasorpy.filter import phasor_threshold
from phasorpy.lifetime import phasor_to_apparent_lifetime
"""
Batch processes FLIM data from PTU files, calculates phasor-based lifetimes, 
and segments regions of interest using headless Ilastik classification.

This script recursively searches a target directory for PicoQuant `.ptu` files. 
For each file, it uses `phasorpy` to extract the photon decay signal, calculate 
phasor coordinates, and apply phase and modulation referencing. It then exports a 
2-channel TIFF (Intensity and Phase Lifetime) to run a headless Ilastik pixel 
classification project. 

The resulting Ilastik mask is used to isolate a specific region of interest (ROI) 
and calculate an intensity-weighted "Super Pixel" apparent lifetime. Finally, it 
generates a 4-panel summary visualization and cleans up temporary TIFF files.

Dependencies
------------
numpy, matplotlib, tifffile, phasorpy, subprocess
External: Ilastik desktop application

Configuration
-------------
superfolder : str
    Absolute path to the root directory containing the `.ptu` files.
ilastik_exe : str
    Absolute path to the Ilastik executable (`ilastik.exe`).
ilastik_project : str
    Absolute path to the trained Ilastik project file (`.ilp`).
target_class : int
    The integer value from the Ilastik output corresponding to the desired ROI mask.
reference_phase : float
    The phase angle (in degrees) of the reference standard for phasor calibration.
reference_modulation : float
    The modulation ratio of the reference standard for phasor calibration.

Outputs
-------
Console
    Prints progress updates and the calculated "Super Pixel Lifetime" (in ns) 
    for each processed file.
.png files
    A 4-panel summary plot saved alongside each `.ptu` file, showing the raw 
    intensity, Ilastik mask, global decay histogram, and a phasor scatter plot 
    with the universal semicircle and superpixel coordinate.

Notes
-----
- The script currently assumes a single-channel extraction (`channel=1`) and a 
  single frame (`frame=0`).
- The Ilastik input TIFF is scaled identically to the training data: intensity 
  is multiplied by 5000, and lifetime is converted to picoseconds (both cast to uint16).
- A commented-out section at the bottom contains preliminary logic for directly 
  processing `.tif` files instead of `.ptu` files.
"""
# --- Core Directories & Ilastik Setup ---
superfolder = r"Y:\For James\2026-05-12 BA cells RERUN\IMG0012_SR3420 T-series_ROI01"

ilastik_exe = r"C:\Program Files\ilastik-1.4.2\ilastik.exe"
ilastik_project = r"Y:\For James\Ilastik BA cells\BA cells ilastik.ilp"
target_class = 2  # The integer class representing your ROI in Ilastik

# --- Setup Lists & Fixed Parameters ---
processed_files = []
taus = []
tiff = True # i finput files are tiffs then set to true
channel = 1
frame = 0
reference_phase = 47
reference_modulation = 1.1
intensity_min = 0
bins = 5000  # Added back in for scaling

# --- File Discovery ---
search_pattern = os.path.join(superfolder, "**", "*.ptu")
files = glob.glob(search_pattern, recursive=True)

print(f"Found {len(files)} PTU files. Starting batch analysis...")

# --- Batch Processing Loop ---
# if tiff == False:
for ptu_path in files:
    file_name = os.path.basename(ptu_path)
    print(f"\nProcessing: {file_name}...")
    processed_files.append(file_name)
    
    # 1. Load and process signal
    signal = signal_from_ptu(ptu_path, channel=channel, frame=frame)
    frequency = signal.attrs['frequency']
    mean, real, imag = phasor_from_signal(signal)
    
    # Transform and threshold
    real, imag = phasor_transform(real, imag, -math.radians(reference_phase), 1 / reference_modulation)
    mean, real, imag = phasor_threshold(mean, real, imag, mean_min=intensity_min)
    
    # Calculate initial lifetime for the 2-channel stack (No median filter, matching training script)
    phase_lifetime, _ = phasor_to_apparent_lifetime(real, imag, frequency)
    
    # --- 2. ILASTIK MASK GENERATION (Matching Training Data Perfectly) ---
    # Scale exactly as you did during training
    intens_final = np.round(mean * bins).astype(np.uint16)
    tau_final = np.round(phase_lifetime * 1000).astype(np.uint16)
    
    # Stack into 2 Channels (CYX)
    stacked_data = np.stack([intens_final, tau_final], axis=0)
    
    temp_dir = os.path.dirname(ptu_path)
    temp_tif_path = os.path.join(temp_dir, "temp_2channel_input.tif")
    mask_output_path = os.path.join(temp_dir, "temp_2channel_input_mask.tiff")
    
    # Save the 2-channel temporary TIFF with ImageJ CYX formatting
    tif.imwrite(
        temp_tif_path, 
        stacked_data, 
        imagej=True, 
        photometric='minisblack',
        metadata={'axes': 'CYX'}
    )
    
    # Run Ilastik
    ilastik_cmd = [
        ilastik_exe,
        "--headless",
        f"--project={ilastik_project}",
        "--export_source=Simple Segmentation",
        "--output_format=tiff",
        f"--output_filename_format={mask_output_path}",
        temp_tif_path
    ]
    
    print("  -> Running Ilastik segmentation...")
    subprocess.run(ilastik_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists(mask_output_path):
        # Ilastik outputs the mask matching the spatial dimensions (Y, X)
        ilastik_mask = tif.imread(mask_output_path)
        bool_mask = (ilastik_mask == target_class)
        
        # --- 3. SUPER PIXEL MATH ---
        # Apply the mask to the raw phasor arrays
        masked_mean = mean[bool_mask]
        masked_real = real[bool_mask]
        masked_imag = imag[bool_mask]
        
        super_pixel_mean = np.sum(masked_mean)
        
        if super_pixel_mean > 0: 
            super_pixel_real = np.sum(masked_mean * masked_real) / super_pixel_mean
            super_pixel_imag = np.sum(masked_mean * masked_imag) / super_pixel_mean
            
            tau, _ = phasor_to_apparent_lifetime(super_pixel_real, super_pixel_imag, frequency)
            tau_ns = tau  
            taus.append(tau_ns)
            print(f"  -> Super Pixel Lifetime: {tau_ns:.3f} ns")
        else:
            print("  -> Warning: Ilastik mask was completely empty.")
            taus.append(np.nan)
            super_pixel_real, super_pixel_imag = 0, 0
            
        # --- 4. PLOTTING & SAVING ---
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # A) Intensity Image
        axs[0, 0].imshow(mean, cmap='viridis')
        axs[0, 0].set_title('Intensity Image')
        axs[0, 0].axis('off')
        
        # B) Ilastik Mask
        axs[0, 1].imshow(bool_mask, cmap='gray')
        axs[0, 1].set_title('Ilastik Mask')
        axs[0, 1].axis('off')
        
        # C) Decay Histogram
        decay_histogram = np.sum(np.array(signal), axis=(0, 1))
        axs[1, 0].plot(decay_histogram, color='black')
        axs[1, 0].set_yscale('log') 
        axs[1, 0].set_title('Global Decay Histogram')
        axs[1, 0].set_xlabel('Time Bins')
        axs[1, 0].set_ylabel('Photon Counts (Log Scale)')
        
        # D) Phasor Coordinates
        x_semi = np.linspace(0, 1, 200)
        y_semi = np.sqrt(0.25 - (x_semi - 0.5)**2)
        axs[1, 1].plot(x_semi, y_semi, 'k-', linewidth=1.5, label='Universal Semicircle')
        
        # Scatter plot of masked pixels
        axs[1, 1].scatter(masked_real, masked_imag, c='blue', alpha=0.1, s=2, label='Masked Pixels')
        
        # Plot superpixel
        if super_pixel_mean > 0:
            axs[1, 1].plot(super_pixel_real, super_pixel_imag, 'r*', markersize=12, label='Superpixel')
            
        axs[1, 1].set_xlim([0, 1])
        axs[1, 1].set_ylim([0, 0.6])
        axs[1, 1].set_title('Phasor Coordinates')
        axs[1, 1].set_xlabel('G (Real)')
        axs[1, 1].set_ylabel('S (Imaginary)')
        axs[1, 1].legend()
        
        plt.tight_layout()
        png_path = f"{ptu_path[:-4]}_summary.png"
        plt.savefig(png_path, dpi=300)
        plt.close(fig)
        print(f"  -> Saved summary plot: {os.path.basename(png_path)}")
        
        # Clean up temporary 2-channel file and mask
        os.remove(temp_tif_path)
        os.remove(mask_output_path)
        
    else:
        print("  -> ERROR: Ilastik failed to generate a mask.")
        taus.append(np.nan)
        if os.path.exists(temp_tif_path):
            os.remove(temp_tif_path)

print("\n--- Processing Complete ---")
#%%
# if tiff == True:
#     for tiff_path in files:
#         file_name = os.path.basename(tiff_path)
#         print(f"\nProcessing: {file_name}...")
#         processed_files.append(file_name)
        
#         # 1. Load and process signal
#         with tif.TiffFile(tiff_path) as tif:
#             for page in tif.pages:
#                 image_layer = page.asarray()
                
                
#                 # Stack into 2 Channels (CYX)
        
                
#                 temp_dir = os.path.dirname(tiff_path)
#                 temp_tif_path = os.path.join(temp_dir, "temp_channel_input.tif")
#                 mask_output_path = os.path.join(temp_dir, "temp_channel_input_mask.tiff")
                
#                 # Save the 2-channel temporary TIFF with ImageJ CYX formatting
#                 tif.imwrite(
#                     temp_tif_path, 
#                     stacked_data, 
#                     imagej=True, 
#                     photometric='minisblack',
#                     metadata={'axes': 'TYX'}
#                 )
                
#                 # Run Ilastik
#                 ilastik_cmd = [
#                     ilastik_exe,
#                     "--headless",
#                     f"--project={ilastik_project}",
#                     "--export_source=Simple Segmentation",
#                     "--output_format=tiff",
#                     f"--output_filename_format={mask_output_path}",
#                     temp_tif_path
#                 ]
                
#                 print("  -> Running Ilastik segmentation...")
#                 subprocess.run(ilastik_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
#                 if os.path.exists(mask_output_path):
#                     # Ilastik outputs the mask matching the spatial dimensions (Y, X)
#                     ilastik_mask = tif.imread(mask_output_path)
#                     bool_mask = (ilastik_mask == target_class)
                    
#                     # --- 3. SUPER PIXEL MATH ---
#                     # Apply the mask to the raw phasor arrays
#                     masked_mean = image_layer[bool_mask]
                    
#                     fig, axs = plt.subplots(2, 2, figsize=(12, 10))
                    
#                     # A) Intensity Image
#                     axs[0, 0].imshow(mean, cmap='viridis')
#                     axs[0, 0].set_title('Intensity Image')
#                     axs[0, 0].axis('off')
                    
#                     # B) Ilastik Mask
#                     axs[0, 1].imshow(bool_mask, cmap='gray')
#                     axs[0, 1].set_title('Ilastik Mask')
#                     axs[0, 1].axis('off')
#                     plt.tight_layout()
#                     png_path = f"{tiff_path[:-4]}_summary.png"
#                     plt.savefig(png_path, dpi=300)
#                     plt.close(fig)
                        
