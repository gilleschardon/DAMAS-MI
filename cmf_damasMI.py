#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:36:35 2020

@author: gilleschardon
"""

import numpy as np
import scipy.linalg as la
import time
import line_profiler



def normalize(A):
    return A / np.sqrt(np.real(np.sum(A * np.conj(A), axis=0)))

def freefield(PX, PS, k, derivatives=False):
      
    if PS.ndim == 1:
        dx = PX[:, 0] - PS[0]
        dy = PX[: ,1] - PS[1]
        dz = PX[:, 2] - PS[2]
        
    else:    
        dx = PX[:, 0:1] - PS[:, 0:1].T
        dy = PX[:, 1:2] - PS[:, 1:2].T
        dz = PX[:, 2:3] - PS[:, 2:3].T

    d = np.sqrt(dx*dx + dy*dy + dz*dz);
        
    D = np.exp( -1j * k * d) / d
    
    if derivatives:
        Dd = -1j * k * D - D/d
        
        Dx = - Dd * dx / d
        Dy = - Dd * dy / d
        Dz = - Dd * dz / d
        
        return D, Dx, Dy, Dz

    return D    


def freefield2D(PX, PS, Z, k, derivatives=False):
      
    if PS.ndim == 1:
        dx = PX[:, 0] - PS[0]
        dy = PX[: ,1] - PS[1]
        
    else:    
        dx = PX[:, 0:1] - PS[:, 0:1].T
        dy = PX[:, 1:2] - PS[:, 1:2].T

    d = np.sqrt(dx*dx + dy*dy + Z*Z);
        
    D = np.exp( -1j * k * d) / d

    if derivatives:
        Dd = -1j * k * D - D/d

        Dx = - Dd * dx / d
        Dy = - Dd * dy / d
            
        return D, Dx, Dy
 

    return D



def grid3D(lb, ub, step):
    xx = np.linspace(lb[0], ub[0], int((ub[0]-lb[0]) // step + 1))
    yy = np.linspace(lb[1], ub[1], int((ub[1]-lb[1]) // step + 1))
    zz = np.linspace(lb[2], ub[2], int((ub[2]-lb[2]) // step + 1))

    dims = [xx.size, yy.size, zz.size]

    Xg, Yg, Zg = np.meshgrid(xx, yy, zz)
    
    return np.vstack([Xg.ravel(), Yg.ravel(), Zg.ravel()]).T, dims

def bmap_unc(Sigma, A):
    '''Beamforming map, with matrix of source vectors A'''
    A_norm = normalize(A)
    bf_map = np.real(  np.sum((A_norm.conj().T @ Sigma) * A_norm.T, 1))
    return bf_map


# some fast matrix products

# beamforming
def proddamastranspose(G, Data):    
    x = np.real( np.sum( (G.conj().T @ Data) * G.T, 1))    
    return x

# product by the DAMAS matrix
def proddamas(G, Gconj, x, support=None):    
    if support is None:
        z = G @ (x * Gconj).T
    else:
        z = G[:, support] @ (x[support] * Gconj[:, support]).T    
    return proddamastranspose(G, z)

def proddamasdr(G, x, support=None):    
    if support is None:
        z = G @ (x * G.conj()).T
    else:
        z = G[:, support] @ (x[support] * G[:, support].conj()).T
    z = z - np.diag(np.diag(z))    
    return proddamastranspose(G, z)
    
# local unconstrained least-squares problem
def solve_ls(G, Gconj, bf, support):
    Gram = np.abs(Gconj[:, support].T @ G[:, support]) ** 2
    return la.solve(Gram, bf[support], assume_a='pos'), Gram

# local unconstrained least-squares problem for diagonal removal
def solve_ls_dr(G, bf, support):
    aG2 = np.abs(G[:, support] ** 2)
    Gram = np.abs(G[:, support].conj().T @ G[:, support]) ** 2 - aG2.T @ aG2
    return la.solve(Gram, bf[support], assume_a='pos'), Gram

## CMF-NNLS
def cmf_nnls_lh(G, Data, lreg = 0, verbose=False):
       
    
    Dataloc = Data
    
    Gconj = G.conj()
    
        
        
    prodA = lambda x, support : proddamas(G, Gconj, x, support)
    solve = lambda b, support : solve_ls(G, Gconj, b, support)
    T = time.time()

    bf = proddamastranspose(G, Dataloc) - lreg
    x, obj, Titer, unique = lawson_hanson(prodA, solve, bf, G.shape[1], T, verbose=verbose)

    obj = np.array(obj) + np.sum(np.abs(Data)**2)/2

    return x, obj, Titer


## Lawson-Hanson solver, custom implementation

# ||Mx - y||_2^2, with M described by D
# prodA(D, x, support) = M*x
# solve(D, b, support) solve the local LS problem
# b = M'*y
def lawson_hanson(prodA, solve, b, dim, T,  verbose = True):
    
    T0 = time.perf_counter()
    
    n = dim
    R = np.ones([n], dtype=bool) # active set
    N = np.arange(n);
    x = np.zeros([n])
    
    # residual
    
    w = b
    
    it = 0
    
    obj = []
    Titer = []
    
    

    
    while np.any(R) and (np.max(w[R]) > 0):
        if verbose:
            print(f"iter {it} tol {np.max(w[R]):.2}")
        it = it + 1

        # update of the active set        
        idx = np.argmax(w[R])
        Ridx = N[R]
        idx = Ridx[idx]        
        R[idx] = 0
        
        # least-square in the passive set        
        s = np.zeros(x.shape)      
        s[~R], Gram = solve(b, ~R)
        
        # removal of negative coefficients
        while np.min(s[~R]) <= 0:
            
            Q = (s <= 0) & (~R)
            alpha = np.min(x[Q] / (x[Q] - s[Q]))
            x = x + alpha * (s - x)
            R = (( x <= np.finfo(float).eps) & ~R) | R
            
            s = np.zeros(x.shape)
            s[~R], Gram = solve(b,  ~R)

            
        # update of the solution
        x = s
        # update of the residual
        pA = prodA(x, ~R)
        
        w = b - pA
        
        obj.append(np.dot( x,  pA/2 - b) )
        Titer.append(time.time() - T)
    try:
        la.cholesky(Gram)
        unique = True

        if verbose:
            print(f'Solution is unique, T = {time.perf_counter() - T0:.2}')
    except:
        unique = False

        if verbose:
            print(f'Solution is not unique, T = {time.perf_counter() - T0:.2}')
                   
    return x, obj,Titer,  unique


@line_profiler.profile
def DAMAS_MI2(G, Sigma, niter):

    Gconj = G.conj()
    
    norms2 =  (np.sum(np.abs(G*Gconj), axis=0))
    norms4 =  norms2 * norms2
   
    obj = np.zeros([niter+1])
    obj[0] = np.sum(np.abs(Sigma)**2)/2



    b = proddamastranspose(G, Sigma)

    x = np.zeros(b.shape)
    acols = {} 
    grad = - b
    

    for ni in range(niter):        
        
        gn = grad/norms4
        
        neg = (x - gn) < 0
        
        improvgrad = grad*gn/2
        improvgrad[neg] = 0
        improv0 = grad*x - x*x/2 * norms4
                
        idxgrad = np.argmax(improvgrad)
        idx0  = np.argmax(improv0)
        
        if improvgrad[idxgrad] > improv0[idx0]:           
            d = grad[idxgrad] / norms4[idxgrad]
            x[idxgrad] -= d
            try:
                a = acols[idxgrad]
            except KeyError:
                aa = Gconj.T @ G[:, idxgrad]
                a = aa.real*aa.real + aa.imag*aa.imag
                acols[idxgrad] = a
                
            grad -= d * a

        else:           
            xold =x[idx0]
            x[idx0] = 0

            try:
                a = acols[idx0]
            except KeyError:
                aa = np.abs(Gconj.T @ G[:, idxgrad])
                a = aa*aa
                acols[idx0] = a
                
            grad -=  xold * a           
        

    return x

@line_profiler.profile
def DAMAS_MI2_norm(G, Sigma, niter):

    norms =  np.sqrt((np.sum(np.real(G*G.conj()), axis=0)))
    
    Gnorm = G / norms

    Gnormconj = Gnorm.conj()
    
   
    obj = np.zeros([niter+1])
    obj[0] = np.sum(np.abs(Sigma)**2)/2



    b = proddamastranspose(Gnorm, Sigma)

    x = np.zeros(b.shape)
    acols = {} 
    grad = - b
    

    for ni in range(niter):        
                
        neg = (x - grad) < 0
        
        improvgrad = grad*grad/2
        
        improvgrad[neg] = 0
        improv0 = (grad - x/2)*x
                
        idxgrad = np.argmax(improvgrad)
        idx0  = np.argmax(improv0)
        
        if improvgrad[idxgrad] > improv0[idx0]:           
            d = grad[idxgrad]
            x[idxgrad] -= d
            try:
                a = acols[idxgrad]
            except KeyError:
                aa = Gnormconj.T @ Gnorm[:, idxgrad]
                a = aa.real*aa.real + aa.imag*aa.imag
                acols[idxgrad] = a
                
            grad -= d * a

        else:           
            xold =x[idx0]
            x[idx0] = 0

            try:
                a = acols[idx0]
            except KeyError:
                aa = np.abs(Gnormconj.T @ Gnorm[:, idx0])
                a = aa*aa
                acols[idx0] = a
                
            grad -=  xold * a           
        

    return x / (norms*norms)


@line_profiler.profile
def DAMAS_MI(G, Sigma, niter):
    T = time.time()
    Titer = np.zeros([niter+1])


    Gconj = G.conj()
    
    norms2 =  (np.sum(np.abs(G*Gconj), axis=0))
    norms4 =  norms2 * norms2
   
    obj = np.zeros([niter+1])
    obj[0] = np.sum(np.abs(Sigma)**2)/2



    b = proddamastranspose(G, Sigma)


    support_size = np.zeros([niter])
    visited_size = np.zeros([niter])
    support = np.zeros(b.shape, dtype=bool)
    visited = np.zeros(b.shape, dtype=bool)
     
    x = np.zeros(b.shape)
    acols = {} 
    grad = - b
    
    Titer[0] = time.time() - T

    for ni in range(niter):        
        
        improvgrad = grad*grad/2 * ((x - grad/norms4) > 0) / norms4
        improv0 = grad*x - x*x/2 * norms4
                
        idxgrad = np.argmax(improvgrad)
        idx0  = np.argmax(improv0)
        
        if improvgrad[idxgrad] > improv0[idx0]:           
            d = grad[idxgrad] / norms4[idxgrad]
            x[idxgrad] -= d
            try:
                a = acols[idxgrad]
            except KeyError:
                aa = np.abs(Gconj.T @ G[:, idxgrad])
                a = aa*aa
                acols[idxgrad] = a
                
            grad -= d * a
            
            support[idxgrad] = True
            visited[idxgrad] = True
            obj[ni+1] = obj[ni] - improvgrad[idxgrad]
            
        else:           
            xold =x[idx0]
            x[idx0] = 0
            support[idx0] = False

            try:
                a = acols[idx0]
            except KeyError:
                aa = np.abs(Gconj.T @ G[:, idx0])
                a = aa*aa
                acols[idx0] = a
                
            grad -=  xold * a           
            obj[ni+1] = obj[ni] - improv0[idx0]
        
        support_size[ni] = np.sum(support)
        visited_size[ni] = np.sum(visited)
        Titer[ni+1] =time.time() - T

    return x, support, obj, support_size, visited_size, Titer#, grad * ((x - grad/norms4) > 0) * (x>0) / norms2
@line_profiler.profile
def DAMAS_MI_norm(G, Sigma, niter):
    T = time.time()
    Titer = np.zeros([niter+1])

    norms =  np.sqrt((np.sum(np.real(G*G.conj()), axis=0)))
    
    Gnorm = G / norms

    Gnormconj = Gnorm.conj()

  
    obj = np.zeros([niter+1])
    obj[0] = np.sum(np.abs(Sigma)**2)/2



    b = proddamastranspose(Gnorm, Sigma)


    support_size = np.zeros([niter])
    visited_size = np.zeros([niter])
    support = np.zeros(b.shape, dtype=bool)
    visited = np.zeros(b.shape, dtype=bool)
     
    x = np.zeros(b.shape)
    acols = {} 
    grad = - b
    
    Titer[0] = time.time() - T

    for ni in range(niter):        
        
        improvgrad = grad*grad/2 * ((x - grad) > 0)
        improv0 = (grad - x/2)*x
                
        idxgrad = np.argmax(improvgrad)
        idx0  = np.argmax(improv0)
        
        if improvgrad[idxgrad] > improv0[idx0]:           
            d = grad[idxgrad]
            x[idxgrad] -= d
            try:
                a = acols[idxgrad]
            except KeyError:
                aa = np.abs(Gnormconj.T @ Gnorm[:, idxgrad])
                a = aa*aa
                acols[idxgrad] = a
                
            grad -= d * a
            
            support[idxgrad] = True
            visited[idxgrad] = True
            obj[ni+1] = obj[ni] - improvgrad[idxgrad]
            
        else:           
            xold =x[idx0]
            x[idx0] = 0
            support[idx0] = False

            try:
                a = acols[idx0]
            except KeyError:
                aa = np.abs(Gnormconj.T @ Gnorm[:, idx0])
                a = aa*aa
                acols[idx0] = a
                
            grad -=  xold * a           
            obj[ni+1] = obj[ni] - improv0[idx0]
        
        support_size[ni] = np.sum(support)
        visited_size[ni] = np.sum(visited)
        Titer[ni+1] =time.time() - T

    return x/(norms*norms), support, obj, support_size, visited_size, Titer#, grad * ((x - grad/norms4) > 0) * (x>0) / norms2


@line_profiler.profile
def DAMAS(G, Sigma, niter, randomized=False, roundtrip=False, online=False):
    
    Gconj = G.conj()
    
    T = time.time()


    if not online:
        A = np.abs(G.conj().T@G)**2
    else:
        acols = {} 

    
    norms2 =  (np.sum(np.abs(G)**2, axis=0))
    norms4 =  norms2 * norms2
   
    obj = np.zeros([niter+1])
    obj[0] = np.sum(np.abs(Sigma)**2)/2

    b = proddamastranspose(G, Sigma)

    support_size = np.zeros([niter])
    visited_size = np.zeros([niter])
    support = np.zeros(b.shape, dtype=bool)
    visited = np.zeros(b.shape, dtype=bool)
     
    x = np.zeros(b.shape)
    b = proddamastranspose(G, Sigma)
    grad = - b
    
    N = b.shape[0]

    if randomized:
        idxs = np.random.randint(N, size=niter)
    elif roundtrip:
        idxs = np.arange(niter) % (2*N)
        idxs[idxs >= N] = -idxs[idxs >= N] + 2 * N - 1
    else:
        idxs = np.arange(niter) % N

    Titer = np.zeros([niter+1])
    Titer[0] = time.time() - T


    for ni in range(niter):   
        
        idx = idxs[ni]
        improvgrad = grad[idx]**2/2 * ((x[idx] - grad[idx]/norms4[idx]) > 0) / norms4[idx]
        improv0 = grad[idx]*x[idx] - x[idx]**2/2 * norms4[idx]

        if online:
            try:
                a = acols[idx]
            except KeyError:
                a = np.abs(Gconj.T @ G[:, idx])**2
                acols[idx] = a
        else:
            a = A[:, idx] 
        
        visited[idx] = True

        
        if improvgrad > improv0:           
            d = grad[idx] / norms4[idx]
            x[idx] = x[idx]  - d
            
                
            grad -= d * a
            
            support[idx] = True
            obj[ni+1] = obj[ni] - improvgrad
            
        else:           
            xold =x[idx]
            x[idx] = 0
            support[idx] = False


            grad -=  xold * a           
            obj[ni+1] = obj[ni] - improv0
        
        support_size[ni] = np.sum(support)
        visited_size[ni] = np.sum(visited)
        Titer[ni + 1] = time.time() - T


    return x, support, obj, support_size, visited_size, Titer



@line_profiler.profile
def DAMAS_MI2_norm2(G, Sigma, niter):

    norms =  np.sqrt((np.sum(np.real(G*G.conj()), axis=0)))
    
    Gnorm = G / norms

    Gnormconj = Gnorm.conj()
    
   
    obj = np.zeros([niter+1])
    obj[0] = np.sum(np.abs(Sigma)**2)/2



    b = proddamastranspose(Gnorm, Sigma)

    x = np.zeros(b.shape)
    acols = {} 
    grad = - b
    

    for ni in range(niter):        
                
        neg = (x - grad) < 0
        
        improvgrad = np.abs(grad)
        idxgrad = np.argmax(improvgrad)
        
        

        if x[idxgrad] > grad[idxgrad]:
            case1 = True

        else:
            neg = (x - grad) < 0       
            improvgrad[neg] = 0
            improv0 = (grad - x/2)*x
            
            idxgrad = np.argmax(improvgrad)
            idx0  = np.argmax(improv0)
            
                
            if improvgrad[idxgrad]**2/2 > improv0[idx0]:   
                case1 = True
            else:
                case1 = False
        
        if case1:           
            d = grad[idxgrad]
            x[idxgrad] -= d
            try:
                a = acols[idxgrad]
            except KeyError:
                aa = Gnormconj.T @ Gnorm[:, idxgrad]
                a = aa.real*aa.real + aa.imag*aa.imag
                acols[idxgrad] = a
                
            grad -= d * a

        else:           
            xold =x[idx0]
            x[idx0] = 0

            try:
                a = acols[idx0]
            except KeyError:
                aa = np.abs(Gnormconj.T @ Gnorm[:, idx0])
                a = aa*aa
                acols[idx0] = a
                
            grad -=  xold * a           
        

    return x / (norms*norms)