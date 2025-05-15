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
k = mat['k'][0,0]
Sigma = mat['Data']
Array = mat['Pmic']


# Source grid
Z = 4.6

g2D = lambda x : cmf_damasMI.freefield2D(Array, x, Z, k)

Xgrid, dimgrid = cmf_damasMI.grid3D([-2, -1, Z], [1, 0, Z], 0.02)

# g2D = lambda x : freefield(Array, x, k)

# Xgrid, dimgrid = grid3D([-2, -1, 4], [1, 0, 5], 0.05)


G = g2D(Xgrid)

T = time.time()

bmap = cmf_damasMI.bmap_unc(Sigma, G)

Tbf = time.time() - T



T = time.time()
cmf_map, objLH, TiterLH = cmf_damasMI.cmf_nnls_lh(G, Sigma)
TLH = time.time() - T

Nitertime = 10000
T = time.time()
xqtime, supportq, objq, support_sizeq, visited_sizeq, Titerq = cmf_damasMI.DAMAS_MI_norm(G, Sigma, Nitertime)
T_DAMAS_MI = time.time() - T


cmfDB = 10*np.log10(cmf_map)

def plotmap(pmap, name, dynrange):
    mapDB = 10*np.log10(pmap)
    m = np.max(mapDB)    
    plt.imshow(mapDB, cmap='hot', vmax=m, vmin=m-dynrange, origin="lower", interpolation='None', extent=(-2,1,-1,0))
    current_cmap = plt.cm.get_cmap()
    current_cmap.set_bad(color='black')
    plt.axis('image')
    plt.title(name)
    ax = plt.gca()
    ax.set_facecolor('black')
    plt.colorbar()
    

Niter = Xgrid.shape[0]*10


xq, supportq, objq, support_sizeq, visited_sizeq, Titerq = cmf_damasMI.DAMAS_MI_norm(G, Sigma, Niter)
xd, supportd, objd, support_sized, visited_sized, Titerd = cmf_damasMI.DAMAS(G, Sigma, Niter)
xr, supportr, objr, support_sizer, visited_sizer, Titerr = cmf_damasMI.DAMAS(G, Sigma, Niter, randomized=True)
xar, supportar, objar, support_sizear, visited_sizear, Titerar = cmf_damasMI.DAMAS(G, Sigma, Niter, roundtrip=True)

objcmf = np.sum(np.abs( Sigma - G @ (cmf_map * G.conj()).T)**2)/2

#%%
plt.figure()
plt.subplot(3, 1, 1)

plotmap(np.reshape(bmap, [dimgrid[1], dimgrid[0]]), 'Beamforming', 20)


plt.subplot(3, 1, 2)
plotmap(np.reshape(cmf_map, [dimgrid[1], dimgrid[0]]), 'CMF-NNLS', 20)

plt.subplot(3, 1, 3)
plotmap(np.reshape(xqtime, [dimgrid[1], dimgrid[0]]), 'DAMAS-MI', 20)

plt.figure()

plt.subplot(4, 1, 1)
plotmap(np.reshape(xq, [dimgrid[1], dimgrid[0]]), 'DAMAS-MI', 20)

plt.subplot(4, 1, 2)
plotmap(np.reshape(xr, [dimgrid[1], dimgrid[0]]), 'random DAMAS', 20)

plt.subplot(4, 1, 3)
plotmap(np.reshape(xd, [dimgrid[1], dimgrid[0]]), 'cyclic DAMAS', 20)
plt.subplot(4, 1, 4)

plotmap(np.reshape(xar, [dimgrid[1], dimgrid[0]]), 'roundtrip DAMAS', 20)

#%%

plt.figure()

plt.loglog((objd - objcmf)/objcmf, '-', label="cyclic DAMAS")
plt.loglog((objr - objcmf)/objcmf, '--', label="random DAMAS")
plt.loglog((objar - objcmf)/objcmf, '-.', label="roundtrip DAMAS")

plt.loglog((objq - objcmf)/objcmf, '-', linewidth=4, label="DAMAS-MI")

plt.xlabel('Iteration')
plt.ylabel("$\delta$")
plt.legend()

plt.figure()

plt.loglog(Titerd, (objd - objcmf)/objcmf, '-', label="cyclic DAMAS")
plt.loglog(Titerr, (objr - objcmf)/objcmf, '--', label="random DAMAS")
plt.loglog(Titerar, (objar - objcmf)/objcmf, '-.',label="roundtrip DAMAS")


#plt.loglog(Titerdon, (objd - objcmf)/objcmf, label="cyclic DAMAS on")
#plt.loglog(Titerron, (objr - objcmf)/objcmf, label="random DAMAS on")
#plt.loglog(Titeraron, (objar - objcmf)/objcmf, label="roundtrip DAMAS on")


plt.loglog(Titerq, (objq - objcmf)/objcmf, '-', linewidth=4, label="DAMAS-MI")
plt.loglog(TiterLH, (objLH - objcmf)/objcmf, '--', linewidth=4, label="Lawson-Hanson")

plt.xlabel('Time (s)')
plt.ylabel("$\delta$")
plt.legend()
#%%

plt.figure()

plt.plot(support_sized, '-', label="cyclic DAMAS")
plt.plot(support_sizer, '--', label="random DAMAS")
plt.plot(support_sizear, '-.', label="roundtrip DAMAS")

plt.plot(support_sizeq, '-', linewidth=4, label="DAMAS-MI")
plt.xlabel('Iteration')
plt.ylabel('Support')
plt.legend()
plt.xlim([0, Niter])
plt.ylim([0, 1500])

plt.figure()


plt.plot(visited_sized, '-', label="cyclic DAMAS")
plt.plot(visited_sizer, '--', label="random DAMAS")
plt.plot(visited_sizear, '-.', label="roundtrip DAMAS")

plt.plot(visited_sizeq, '-', linewidth=4, label="DAMAS-MI")
plt.xlim([0, Niter])
plt.ylim([0, Xgrid.shape[0]*1.1])
plt.xlabel('Iteration')
plt.ylabel('Columns of A')
plt.legend()



