#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:21:40 2020

@author: gilleschardon
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import cmf_damasMI

import time

# CMF-NNLS

# Ref.: G. Chardon, J. Picheral, F. Ollivier, Theoretical analysis of the DAMAS algorithm and efficient implementation of the Covariance Matrix Fitting method for large-scale problems, Journal of Sound and Vibration,

mat = loadmat("data_sfw.mat")
k = mat['k'][0,0] # wavenumber
Sigma = mat['Data'] # covariance matrix
Array = mat['Pmic'] # array geometry


# Source grid

# distance between array and source plane
Z = 4.6

# Green's function
g2D = lambda x : cmf_damasMI.freefield2D(Array, x, Z, k)

# discretization grid
Xgrid, dimgrid = cmf_damasMI.grid3D([-2, -1, Z], [1, 0, Z], 0.02)

# matrix of Green's functions between sources and microphones
G = g2D(Xgrid)

# beamforming
bmap = cmf_damasMI.bmap_unc(Sigma, G)

Niter = 5000
Niterdamas = 10*G.shape[1]

source_map_LH, obj, Tlh = cmf_damasMI.cmf_nnls_lh(G, Sigma)

source_map_damas_mi, *_, Tmi = cmf_damasMI.DAMAS_MI(G, Sigma, Niter)

source_map_damas, *_, Tdamas = cmf_damasMI.DAMAS(G, Sigma, Niterdamas)

def plotmap(pmap, name, dynrange):
    mapDB = 10*np.log10(pmap)
    m = np.max(mapDB)    
    plt.imshow(mapDB, cmap='hot', vmax=m, vmin=m-dynrange, origin="lower", interpolation='None')
    current_cmap = plt.cm.get_cmap()
    current_cmap.set_bad(color='black')
    plt.axis('image')
    plt.title(name)
    ax = plt.gca()
    ax.set_facecolor('black')
    plt.colorbar()
    
#%%
plt.figure()

plt.subplot(2, 2, 1)
plotmap(np.reshape(bmap, [dimgrid[1], dimgrid[0]]), 'Beamforming', 20)


plt.subplot(2, 2, 2)
plotmap(np.reshape(source_map_LH, [dimgrid[1], dimgrid[0]]), f'Lawson-Hanson ({Tlh[-1]:.1f} sec.)', 20)

plt.subplot(2, 2, 3)
plotmap(np.reshape(source_map_damas_mi, [dimgrid[1], dimgrid[0]]), f'DAMAS-MI ({Niter} iter., {Tmi[-1]:.1f} sec.)', 20)

plt.subplot(2, 2, 4)
plotmap(np.reshape(source_map_damas, [dimgrid[1], dimgrid[0]]), f'DAMAS ({Niterdamas} iter., {Tdamas[-1]:.1f} sec.)', 20)
