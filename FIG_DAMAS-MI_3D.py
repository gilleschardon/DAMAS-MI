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

# Normalization, for stability
#Sigma = Sigma / np.max(Sigma)

# Source grid

g = lambda x : cmf_damasMI.freefield(Array, x, k)

Xgrid, dimgrid = cmf_damasMI.grid3D([-2.2, -1.3, 3.7], [1.2, 0.3, 5.2999], 0.02)

G = g(Xgrid)

#cmf_map = cmf.cmf_nnls_lh(G, Sigma, dr = False)

#cmfDB = 10*np.log10(cmf_map)

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
    

Niter=80000



T = time.time()
cmf_map, objLH, TiterLH  = cmf_damasMI.cmf_nnls_lh(G, Sigma)
TLH = time.time() - T

T = time.time()
xq, supportq, objq, support_sizeq, visited_sizeq, Titerq = cmf_damasMI.DAMAS_MI(G, Sigma, Niter)
Tgslq = time.time() - T

objcmf = np.sum(np.abs( Sigma - G @ (cmf_map * G.conj()).T)**2)/2


#%%
support_cmf = cmf_map>0

Niter1=100
xq1, supportq1, objq1, support_sizeq1, visited_sizeq1, Titerg1 = cmf_damasMI.DAMAS_MI(G, Sigma, Niter1)

#%%

fig = plt.figure()
ax = fig.add_subplot(2,1,1, projection='3d')
ax.scatter(Xgrid[supportq1, 0], Xgrid[supportq1, 2], Xgrid[supportq1, 1], s=xq1[supportq1]/np.max(xq1)*50, linewidth=0)

ax.set_xlabel('X (m)')
ax.set_ylabel('Z (m)')
ax.set_zlabel('Y (m)')

ax = fig.add_subplot(2,1,2, projection='3d')
ax.scatter(Xgrid[support_cmf, 0], Xgrid[support_cmf, 2], Xgrid[support_cmf, 1], s=cmf_map[support_cmf]/np.max(cmf_map)*50, linewidth=0)


ax.set_xlabel('X (m)')
ax.set_ylabel('Z (m)')
ax.set_zlabel('Y (m)')



#%%
Xgt = np.array([[-1.6065,   -0.4699,    4.5994],[-0.8333  , -0.4910 ,   4.6172],[ 0.1476 ,  -0.4970  ,  4.5261],[-0.6867,   -0.7253   , 4.6501]])


fig = plt.figure()
ax = fig.add_subplot(2,2,1)
ax.scatter(Xgrid[supportq1, 0], Xgrid[supportq1, 1], s=xq1[supportq1]/np.max(xq1)*50, linewidth=0)
ax.scatter(Xgt[:, 0], Xgt[:, 1], c='none', label="Sources", marker='o', edgecolor='k')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')

ax = fig.add_subplot(2,2,2)
ax.scatter(Xgrid[supportq1, 0], Xgrid[supportq1, 2], s=xq1[supportq1]/np.max(xq1)*50, linewidth=0)
ax.scatter(Xgt[:, 0], Xgt[:, 2], c='none', label="Sources", marker='o', edgecolor='k')

ax.set_xlabel('X (m)')
ax.set_ylabel('Z (m)')

ax = fig.add_subplot(2,2,3)
ax.scatter(Xgrid[support_cmf, 0], Xgrid[support_cmf, 1], s=cmf_map[support_cmf]/np.max(cmf_map)*50, linewidth=0)
ax.scatter(Xgt[:, 0], Xgt[:, 1], c='none', label="Sources", marker='o', edgecolor='k')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')

ax = fig.add_subplot(2,2,4)
ax.scatter(Xgrid[support_cmf, 0], Xgrid[support_cmf, 2], s=cmf_map[support_cmf]/np.max(cmf_map)*50, linewidth=0)
ax.scatter(Xgt[:, 0], Xgt[:, 2], c='none', label="Sources", marker='o', edgecolor='k')

ax.set_xlabel('X (m)')
ax.set_ylabel('Z (m)')






#%%
plt.figure()


plt.loglog(Titerq, (objq - objcmf)/objcmf, '-', linewidth=4, label="DAMAS-MI")
plt.loglog(TiterLH, (objLH - objcmf)/objcmf, '--', linewidth=4, label="Lawson-Hanson")

plt.xlabel('Time (s)')
plt.ylabel("$\delta$")
plt.legend()
#%%
plt.figure()

plt.plot(support_sizeq , label="Support")
plt.plot(visited_sizeq, label="Columns")

plt.xlabel('Iteration')
plt.ylabel('Count')
plt.legend()
