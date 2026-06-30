# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:58:52 2026

@author: James
"""

import numpy as np
import matplotlib.pyplot as plt
import ptufile as pf
from scipy.optimize import curve_fit as fit
def calc_xcorr(inten0,inten1,time_per_frame,maxt,numdt=100):
    dt = np.logspace(0,np.log10(maxt/time_per_frame),numdt)
    dt = dt.astype(np.uint32)
    dt = np.unique(dt)
    dt = dt[dt<(inten0.shape[0]/3)]
    G1 = np.zeros(dt.shape[0])
    G2 = np.zeros(dt.shape[0])
    for i in range(dt.shape[0]):
        G1[i] = np.mean(inten0[:(len(inten0)-dt[i])]*inten1[dt[i]:])
        G2[i] = np.mean(inten1[:(len(inten1)-dt[i])]*inten0[dt[i]:])
    G1 = G1/np.mean(inten0)/np.mean(inten1)-1
    G2 = G2/np.mean(inten0)/np.mean(inten1)-1
    G = (G1+G2)/2
    return dt,G
def plot_xcorr_and_fit(G,dtlistsec,f,fres,actuallyplot):
    xfit = np.logspace(np.log10(np.min(dtlistsec)),np.log10(np.max(dtlistsec)),1000)
    yfit = np.zeros(xfit.shape[0])
    for i in range(xfit.shape[0]):
        yfit[i] = f(xfit[i],*fres)
    if actuallyplot:
        plt.semilogx(dtlistsec,G,'o',label='data')
        plt.semilogx(xfit,yfit,'-',label='fit')
        plt.xlabel('tau (sec)')
        plt.ylabel('G(tau)')
#        plt.title('Correlation')
        plt.show()
    return dtlistsec, G, xfit, yfit
def fit_xcorr(G,dtlistsec,model):
    ## Model:
    #    '2d'     = Brownian 2D model
    #    '2danom' = Anomalous 2D model
    #    '3d'     = Brownian 3D model

    if (model.find('2danom') > -1 ):
        f = lambda x,td,a,alpha,c: a/(1+(x/td)**alpha)+c
        numparam = 3
        start = (0.005,1,0.9,0.0)
        bound = ((0.0,0.0,0.0,0.0),(1.0,np.inf,np.inf,np.inf))
    elif (model.find('2d') > -1 ):
        f = lambda x,td,a,c: a/(1+(x/td))+c
        numparam = 3
        start = (0.005,1,0)
        bound = ((0.0,-np.inf,-np.inf),(1.0,np.inf,np.inf))##sonali changed lower bound from 0.0 to -np.inf on 11/22/21

    elif (model.find('3dx') > -1):
        f = lambda x,td,td2,a,a2,r: a/(1+(x/td))/(1+r**-2*np.sqrt(x/td)) + a2/(1+(x/td2))/(1+r**-2*np.sqrt(x/td2))
        numparam = 3
        start = (0.005,0.01,0.1,0.5,0.5)
        bound = ((0.0,0.0,0.0,0.0,0.0),(1.0,np.inf,np.inf,np.inf,np.inf))
    #elif (model.find('3d') > -1):
        #f = lambda x,td,a,r: a/(1+(x/td))/(1+r**-2*np.sqrt(x/td))
        #numparam = 3
        #start = (0.005,1,1)
        #bound = ((0.0,-np.inf,0.01),(1.0,np.inf,np.inf))#sonali changed lower bound from 0.0 to -np.inf on 11/22/21
    elif (model.find('3d') > -1):
        f = lambda x,td,a,r: a/(1+(x/td))/np.sqrt(1+(r**-2)*(x/td))
        numparam = 3
        start = (0.005,1,1)
        bound = ((0.0,-np.inf,0.01),(1.0,np.inf,np.inf))

    else:
        print('no fitting function used')
        f = lambda x,a,b: a/x+b
        numparam = 2
        start = (1.0,1.0)
        bound = ((0.0,0.0),(np.inf,np.inf))

    fres,covf = fit(f,dtlistsec,G,p0=start,bounds=bound)
#    print(fres)
#    param = np.zeros(numparam)
#    for i in range(numparam):
#        param[i] = fres[i][0]
    return f, fres, covf, numparam  #, param


ptu_path = r"Y:\for Anu\2026-05-04 Anu FCCS PLIN5 & ABHD5\2026-06-09 FCCS ABHD5 1_100 and PLIN5 1_100 lysates\IMG0042_PLIN5 yfp and A5 mch 1_100 lysates\PQSpcmDriver_2026-06-09_11-57-04.896.ptu"
ch0 = 0
ch2 = 2
ptu = pf.PtuFile(str(ptu_path))
records = ptu.decode_records()


photons = records[records['dtime'] >= 0]
t_max = records['time'][-1] * ptu.global_resolution


ch0_times = photons['time'][photons['channel'] == ch0] * ptu.global_resolution
ch2_times = photons['time'][photons['channel'] == ch2] * ptu.global_resolution


time_per_frame = 100e-6 
time_bins = np.arange(0, t_max, time_per_frame)

# 4. Bin the photons for each channel into a matching time-series
inten0, _ = np.histogram(ch0_times, bins=time_bins)
inten1, _ = np.histogram(ch2_times, bins=time_bins)

# 5. Calculate Cross-Correlation (Channel 0 vs Channel 1)
dt_indices, G_cross = calc_xcorr(inten0, inten1, time_per_frame, maxt=t_max, numdt=100)
dt_seconds = dt_indices * time_per_frame
# --- Calculate Autocorrelations for comparison ---
_, G_auto0 = calc_xcorr(inten0, inten0, time_per_frame, maxt=t_max, numdt=100)
_, G_auto1 = calc_xcorr(inten1, inten1, time_per_frame, maxt=t_max, numdt=100)

# --- NEW: Apply a 1-Second Cutoff for the Fitting ---
fit_mask = dt_seconds <= 1.0

# Filter the time lag array
dt_fit = dt_seconds[fit_mask]

# Filter the correlation arrays for fitting
G_auto0_fit = G_auto0[fit_mask]
G_auto1_fit = G_auto1[fit_mask]
G_cross_fit = G_cross[fit_mask]


# --- Fit the Truncated Data ---
selected_model = '2d'

# Use the filtered arrays (dt_fit and G_***_fit) for the fit_xcorr function
f0, fres0, covf0, _ = fit_xcorr(G_auto0_fit, dt_fit, model=selected_model)
f1, fres1, covf1, _ = fit_xcorr(G_auto1_fit, dt_fit, model=selected_model)
fx, fresx, covfx, _ = fit_xcorr(G_cross_fit, dt_fit, model=selected_model)


# --- Plotting the Data and the Fits ---
plt.figure(figsize=(9, 6))

# Plot Channel 0 (Showing ALL data points as circles, but the fit line stops at 1s)
_, _, xfit0, yfit0 = plot_xcorr_and_fit(G_auto0_fit, dt_fit, f0, fres0, actuallyplot=False)
plt.semilogx(dt_seconds, G_auto0, 'o', ms=4, label="Ch 0 Auto (PLIN5-YFP)", alpha=0.3, color='green')
plt.semilogx(xfit0, yfit0, '-', label="Ch 0 Fit (<= 1s)", color='green', linewidth=2)

# Plot Channel 2
_, _, xfit1, yfit1 = plot_xcorr_and_fit(G_auto1_fit, dt_fit, f1, fres1, actuallyplot=False)
plt.semilogx(dt_seconds, G_auto1, 'o', ms=4, label="Ch 2 Auto (ABHD5-mCh)", alpha=0.3, color='red')
plt.semilogx(xfit1, yfit1, '-', label="Ch 2 Fit (<= 1s)", color='red', linewidth=2)

# Plot Cross-Correlation
_, _, xfitx, yfitx = plot_xcorr_and_fit(G_cross_fit, dt_fit, fx, fresx, actuallyplot=False)
plt.semilogx(dt_seconds, G_cross, 'o', ms=5, label="Cross-Correlation", color='black', alpha=0.4)
plt.semilogx(xfitx, yfitx, '-', label="Cross Fit (<= 1s)", color='black', linewidth=2.5)

# Standard Plot Polish
plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, label='Fit Boundary (1s)')
plt.xlabel("Lag Time $\tau$ (seconds)")
plt.ylabel("G($\tau$)")
plt.title("FCCS Analysis: Truncated Fits (Up to 1 Second)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

print(f"Ch 0 Diffusion Time (td): {fres0[0]*1000:.2f} ms")
print(f"Ch 2 Diffusion Time (td): {fres1[0]*1000:.2f} ms")
print(f"Cross-Corr Diffusion Time (td): {fresx[0]*1000:.2f} ms")
