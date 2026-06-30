# StandardPythonFunctions

These functions have been written by Prof. Christopher V Kelly and researchers in his laboratory at Wayne State University.


FLIM_FUNCTIONS_JGS:
  
  Self-Contained FLIM Pipeline designed for Abberior PTUs
    Extracts PTU data, runs headless Ilastik masking, calculates lifetimes,
    and generates batch summary visualizations without relying on PhasorPy.

run_ilastik:

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
