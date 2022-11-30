# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import collections

import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import argparse
import scipy.io
from scipy.interpolate import interp1d
from numba import jit, njit
import scipy.ndimage
from numpy.random import default_rng
import xarray as xr
import copy
import pickle
import gzip


#%%

# preallocate empty array and assign slice by chrisaycock
def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

#%%


class gm1s:
    def __init__(self, L=None, ts=None, H=None, bt=None, tau=None, g=-9.81, rho=916.8, b_p=0, dbdt=None, mode=None):
        if isinstance(dbdt, (int, float)) & (dbdt is not None):
            self.mode = 'linear'
            self.bt_p = np.full_like(ts, fill_value=dbdt) + bt
            self.b_p = np.full_like(ts, fill_value=dbdt)
            self.bt_eq = bt
        elif isinstance(b_p, (collections.abc.Sequence, np.ndarray)):
            self.mode = 'discrete'
            self.bt_p = bt + b_p
            self.bt_eq = bt
            self.b_p = b_p
        elif isinstance(b_p, (int, float)):
            # step change
            # todo: implement sol'n for step change
            self.mode = 'discrete'
            b_p = np.full_like(ts, fill_value=b_p)
            self.bt_p = bt + b_p
            self.bt_eq = bt
            self.b_p = b_p
        
        self.ts = ts
        self.dt = np.diff(ts, prepend=-1)
        self.L_bar = L  # steady state without perturbations (m)
        self.H = H  # m 
        self.dbdt = dbdt
        
        self.beta = self.L_bar/H  # eq. 1
        if tau is None:
            self.tau = -H / bt
        else:
            self.tau = tau
            
        if self.mode == 'linear':
            self.linear()
        elif self.mode == 'discrete':
            self.discrete()
    
    
    def linear(self):
        
        self.tau = -self.H / self.bt_p
        self.L_eq = self.tau * self.beta * self.bt_p * (self.ts - self.tau)
        self.L_p = np.zeros_like(self.ts, dtype='float')
        
        # Christian et al eq. 4
        if self.mode == 'linear':
            for i, t in enumerate(self.ts):
                if self.bt_p[i] == 0:
                    self.L_p[i] = 0
                    continue
                    
                self.L_p[i] = self.tau[i] * self.beta * self.bt_p[i] * (t - self.tau[i] * (1 - np.exp(-t / self.tau[i])))
                

        self.L = self.L_bar + np.cumsum(self.L_p)
        
    
    def discrete(self):
        self.L_p = np.empty_like(self.ts, dtype='float')

        
        for i, t in enumerate(self.ts):
            # Roe and Baker (2014) eq. 8
            if i == 0:
                self.L_p[i] = self.beta * t * self.b_p[i]
                continue
            self.L_p[i] = (1 - self.dt[i] / self.tau) * self.L_p[i - 1] + self.beta * self.dt[i] * self.b_p[i]
        self.L = self.L_bar + self.L_p
        self.L_eq = self.tau * self.beta * self.bt_p 
        self.dL = abs(self.L_p[-1])
                        
                

    def to_xarray(self):
        import xarray as xr

        ds = xr.Dataset(
            data_vars=dict(
                bp=('t', self.b_p),
                Pp=('t', self.Pp),
                Tp=('t', self.Tp),
            ),
            coords=dict(
                t=self.ts
            ),
            attrs=dict(

            )
        )

        return ds

    def to_pandas(self):
        import pandas as pd

        df = pd.DataFrame(data=dict(bp=self.bp,
                                    Pp=self.Pp,
                                    Tp=self.Tp, ),
                          index=pd.Index(self.t, name='t'))

        return df

    def to_tidy(self):
        import pandas as pd

        df = self.to_pandas().reset_index()
        df = pd.melt(df, id_vars=['t'])

        return df


################################################################################
class gm3s:
    """ Real 3-stage linear glacier model
    
    Assumes Lambda << L_bar and h2_p, h3_p ~= H. Therefore, V1_p ~= h1_p * L_bar, 
    V2_p ~= h2_p * lambda, and V3_p ~= (L_p - Lambda)*H ... (see figure)

    Parameters
    ----------
    mode : str, optional
        Calculate anomalies based on mass balance 'b' or length 'l'. Default is 'b'.
    ts : int
        Timeseries of years.
    mu : numeric
        Melt factor [m yr**-1 K**-1]
    Atot : numeric
        Total area of the glacier [m**2]
    ATgt0 : numeric
        Area of the glacier where some melting occurs [m**2]
    Aabl : numeric
        Ablation area [m**2] 
    w : numeric
        Characteristic width of the glacier tongue [m]. 
    H : numeric
        Characteristic ice thickness near the terminus [m]
    gamma : numeric
        Assumed surface lapse rate [K m**-1] 
    dzdx : numeric
        Assumed basal slope [no units]
    sigP : numeric
        Std. dev. of accumulation variability [m yr**-1]
    sigT : numeric
        Std. dev. of melt-season temperature variability [m yr**-1]
    sigb : numeric
        Std. dev. of annual-mean mass balance [m yr**-1]. One of `b` or `sigb` is required for `mode='b'`
    T : array-like
        Annual melt season temperature anomaly. One of `T` or `sigT` is required for `mode='l'`.
    P : array-like
        Annual accumulation anomaly. One of `P` or `sigP` is required  for `mode='l'`.
    b : array-like
        Annual mass balance anomaly. One of `b` or `sigb` is required for `mode='b'`.

    Returns
    -------
    linear_1s : object
        
        
    
    
    
    """
    
    def __init__(self, L, H, ts, mode='b', bt=None, b_p=None, tau=None, dzdx=None,
                 Atot=None, W=None, mu=None, gamma=None, Aabl=None, 
                 sigT=None, sigP=None, sigb=None, P0=None, T_p=None, P_p=None,
                 ATgt0=None):
        self.ts = ts
        self.dt = np.diff(self.ts+1, prepend=ts[0])  # works idk why
        #self.dt = np.diff(self.ts)  # works idk why
        
        if mode == 'l':
            # note at this point you could create your own climate time series, using
            # random forcing, trends, oscillations etc.
            # Tp = array of melt-season temperature anomalise
            # Pp = array of accumlation anomalies
            # bp = array of mass balance anomalies
            if T_p is None:
                self.T_p = norm.rvs(scale=sigT, size=len(ts))
            if P_p is None:
                self.P_p = norm.rvs(scale=sigP, size=len(ts)) + P0
            if b_p is None and mode == 'b':
                self.b_p = norm.rvs(scale=sigb, size=len(ts))
            self.alpha = -mu * ATgt0 * self.dt / (w * H)
            self.ATgt0 = ATgt0
            self.sigT = sigT
            self.sigP = sigP
            self.sigb = sigb
            self.P0 = P0
            self.Aabl = Aabl
            self.gamma = gamma
            self.mu = mu
        if mode == 'b':
            if isinstance(b_p, (collections.abc.Sequence, np.ndarray)):
                pass
            elif isinstance(b_p, (int, float)):
                # step change
                # todo: implement sol'n for step change
                b_p = np.full_like(ts, fill_value=b_p)

        self.mode = mode
        self.bt_p = bt + b_p
        self.bt_eq = bt
        self.b_p = b_p
        self.L_bar = L  # m
        self.H = H  # m 
        self.dzdx_s = dzdx
        
        self.Atot = Atot
        self.W = W

        # glacier memory [ys]
        # this is the glacier response time  (i.e., memory) based on the above glacier geometry
        # if you like, just pick a different time scale to see what happens. 
        # Or also, use the simple, tau = hbar/b_term, if you know the terminus
        # balance rate from, e.g., observations
        if tau is None:
            try:
                self.tau = W * H / (mu * gamma * dzdx * Aabl)
            except:
                try:
                    self.tau = H / bt
                except:
                    pass
        else:
            self.tau = tau
        
        # coefficient needed in the model integrations
        # keep fixed - they are intrinsic to 3-stage model
        self.eps = 1 / np.sqrt(3)
        self.K = 1 - self.dt / (self.eps * self.tau)
        self.beta = self.Atot * self.dt / (self.W * self.H)
        
    
    def copy(self):
        return copy.deepcopy(self)
    
    def to_pandas(self):
        import pandas as pd
        
        df = pd.DataFrame(data=dict(L_p=self.L_p,
                                    L_eq=self.L_eq,
                                    dL=self.dL,
                                    bt_p=self.bt_p, ),
                          index=pd.Index(self.ts, name='t'))
        return df
        
    def linear(self, bt=None):
        # preeeetty sure this doesn't work at all
        
        if bt is not None:
            self.b_p = bt
            self.bt_p = self.bt_eq + self.b_p
        
        # convenience renaming
        tau = self.tau
        L_bar = self.L_bar
        H = self.H
        eps = self.eps
        ts = self.ts
        
        n_steps = len(ts)
        self.h = np.zeros(n_steps)
        self.F = np.zeros(n_steps)
        self.L = np.zeros(n_steps)
        self.L_p = np.zeros(n_steps)
        self.L_debug = np.zeros(n_steps)

        self.h[0] = 0
        # self.F[0] = 0
        self.L[0] = self.L_bar
        # self.L_p[0] = 0
        # self.L_debug[0] = 0

        for i,t in enumerate(ts):

            self.h[i] = (1 - self.dt[i] / (eps * tau)) * self.h[t - self.dt[i]] + self.bt_p[i]
            self.F[i] = (1 - self.dt[i] / (eps * tau)) * self.F[t - self.dt[i]] + L_bar / (eps * tau)**2 * self.h[i]  # writing F2 as F
            self.L[i] = (1 - self.dt[i] / (eps * tau)) * self.L[t - self.dt[i]] + self.F[i] / (eps * H)
            self.L_p[i] = self.L[i] - self.L[t - self.dt[i]]
    
            try:
                self.L_debug = self.dt * self.L_bar / (eps * self.H) * (self.dt / (eps * tau))**2 * self.bt_p[t - 3]  # if I specified everything right, this should be the same is L[t]
            except:
                pass
    
            
    def run(self):
        # convenience renaming
        K = self.K
        eps = self.eps
        tau = self.tau
        beta = self.beta
        
        L_p = np.zeros_like(self.ts, dtype='float')


        # L3s(i) = 3 * phi * L3s(i - 1) -
        # 3 * phi ^ 2 * L3s(i - 2)
        # + 1 * phi ^ 3 * L3s(i - 3)...
        # + dt ^ 3 * tau / (eps * tau) ^ 3 * (beta * bp(i - 3))
        
        if self.mode=='b':
            for i, t in enumerate(self.ts):
                if i <= 3:
                    continue
                L_p[i] = 3 * K[i] * L_p[i - 1] - \
                         3 * K[i]**2 * L_p[i - 2] + \
                         1 * K[i]**3 * L_p[i - 3] + \
                         self.dt[i]**3 * tau / (eps * tau)**3 * (beta[i] * self.b_p[i - 3])

        if self.mode=='l':  
            for i, t in enumerate(self.ts):
                if i <= 3:
                    continue
                L_p[i] = 3 * K[i] * L_p[i - 1] - \
                         3 * K[i]**2 * L_p[i - 2] + \
                         1 * K[i]**3 * L_p[i - 3] + \
                         self.dt[i]**3 * tau / (eps * tau)**3 * (beta[i] * self.P_p[i - 3] - self.alpha * self.T_p[i - 3])
                
        self.L_p = L_p
        self.L = self.L_bar + self.L_p
        self.L_eq = self.tau * self.beta * self.bt_p
        self.dL = abs(L_p[-1])
        
        return self
       
            
    def power_spectrum(self, freq, sig_L_1s):
        P0 = 4 * self.tau * sig_L_1s  # power spectrum in the limit f -> 0 using the variance from the 1s model
        P_spec = P0 * (1-self.K)**6 / (1 - 2 * self.K * np.cos(2*np.pi*freq*self.dt) + self.K**2)**3  # eq. 20
        return P_spec
    
    def phase(self, freq):
        ''' Mostly correct?'''
        
        H = np.exp(-6 * np.pi * 1j * freq * self.dt)/(1 - self.K * np.exp(-2 * np.pi * 1j * freq * self.dt))**3  # eq. 19b
        phase = np.angle(H * 1j, deg=True)
        
        return phase
    
    def acf(self, t):
        '''Based on the continuous form of the 3-stage equations'''
        eps = self.eps
        tau = self.tau
        
        acf = np.exp(-t/(eps*tau)) * (1 + t/(eps*tau) + 1/3*(t/(eps*tau))**2)
        return acf
        
            

###############################################################################            
class flowline:
    
    def __init__(self, L0, h0, W, dzdx, mu, Tbar, sigT, Pbar, sigP, zb,
                 t1, t0=0, delx=200, gamma=6.5e-3, rho=916.8, f_d=1.9e-24, f_s=5.7e-20,
                 dhdb=None):
        '''
        This code solves the shallow-ice equations for ice flow down a 
        flowline with prescribed bed geometry. See Roe, JGlac, 2011, or 
        many, many other places for equations.
        
        You can provide a time series of climate (accumlation and melt-season 
        temperature). The code then calculates integrates the shallow-ice 
        equations.
        
        The standard set of parameters are chosen(ish) to be typical of small Alpine
        glaciers around Mt. Baker, WA. The flowline assumes constant width (1D),
        so the glacier is longer than typical around Mt Baker in order to
        emulate a realistic accumulation area. 
        
        This version of the model was just for fun, so the parameters are
        different from those in Roe, JGalc, 11. 
        
        The code uses an explicit numberical scheme, so is not particularly
        stable. If it blows up, try and shorter time step (delt), or a coarser
        spatial resolution (delx). I am a crappy coder, so there are no
        guarantees it is free of bugs.
        
        I have tuned the mean glacier length primarily by specifying average 
        melt-season temperature at z=0, and mean-annual accumulation, Pbar.
        
        Right now the code loads in an initial glacier profile from the file 
        "glac_init_baker.mat". If you change parameters, you may want to
        integrate the glacier in time with climate anomalies set to zero until
        the glacier attains equilibrium, save that profile, and use it as the
        initial conditions for your new integrations (if that makes sense).
        
        You can change the basal geometry, glacier size, climate, flow
        parameters,
        
        
        Feel free to use for anything, and to change anything. Have fun! 
        Gerard (groe@uw.edu - although negligible technical support will be offered!). 
        
        "Most numerical models
        solve the shallow-ice equations (which neglect longitudinal
        stresses; e.g. Hutter, 1983) and incorporate a representation
        of basal sliding. For a one-dimensional flowline following the
        longitudinal profile of a glacier"

        "The first equation represents local
        mass conservation, while the second represents the transla-
        tion and deformation of ice associated with shear stresses. In
        combination, the equations have the form of a nonlinear
        diffusion equation in thickness."

        '''
        
        #%%
        # model parameters
        self.ts = t0  # starting time, yrs
        self.tf = t1  # final time, yrs
        self.delt = 0.05  # initial time step in yrs
        self.delx = delx  # grid spacing, m
        self.xmx = len(dzdx) * delx  # the domain size in m
        self.nxs = int(np.round(self.xmx / self.delx)) - 1  # number of grid points
        self.x = np.arange(0, self.xmx, self.delx)  # x array
        # self.nts = int(np.floor(
        #     (self.tf - self.ts) / self.delt))  # number of time steps ('round' used because need nts as integer)
        
        # glacier parameters
        self.L0 = L0
        self.h0 = h0
        self.W = W
        self.dzdx = dzdx
        self.rho = rho  # kg/m^3
        self.f_s = f_s
        self.f_d = f_d
        self.g = 9.81  # m/s^2
        self.n = 3  # shallow ice param
        self.A = 2.4e-24  # #Pa^-3 s^-1
        self.K = 2 * self.A * (rho * self.g)**self.n / (self.n + 2)
        self.K = self.K * np.pi * 1e7  # make units m^2/yr

        # glacier geometry
        self.zb = zb
        self.dzbdx = np.gradient(self.zb, self.x)  # slope of glacer bed geometries.

        # climate parameters
        self.Tbar = Tbar    # average temperature at z=0 [^oC]
        self.sigT = sigT    # standard deviation of temperature [^oC]        
        self.Pbar = Pbar    # average value of accumulation [m yr^-1]
        self.sigP = sigP    # standard deviation of accumulation [m yr^-1]
        self.gamma = gamma  # lapse rate  [K m^-1]
        self.mu = mu      # melt factor for ablation [m yr^-1 K^-1]
        self.rng = default_rng()
        
        
        # find zeros of flux to find glacier length
        # Note oerlemans value is c=10
        # self.b0 = (3 - (4 / 5000) * self.x)  # mass balance in m/yr
        # self.c = 1  # equals 2tau/(rho g) plastic ice sheet constant from paterson
        # #self.c = 10
        # idx = np.where(np.cumsum(self.b0) < 0)  # indices where flux is less than zero
        # idx = np.min(idx)
        # self.L = self.x[idx]  # approximate length of glacier
        self.L = self.L0 * 0.5
        self.c = 10
        idx = len(self.x * 0.5)

        ###########################################
        # plastic glacier initial profile
        ###########################################
        if isinstance(self.h0, np.ndarray):
            self.h = self.h0
        else:
            self.h = np.zeros_like(self.x, dtype='float')  # zero out initial height array
            self.h[0:idx-1] = np.sqrt(self.c * (self.L - self.x[0:idx-1]))  # plastic ice profile as initial try
            self.h = np.nan_to_num(self.h)


        self.nyrs = self.tf - self.ts
        self.Pout = np.zeros(self.nyrs)
        self.Tout = np.zeros(self.nyrs)

        self.dhdt_out = []
        self.h_out = []
        self.Qp_out = []
        self.Qm_out = []
        self.edge_out = []
        self.t_out = []
        self.P_out = []
        self.T_out = []
        self.b_out = []
        self.dzbdx_out = []
        self.zb_out = []
        self.melt_out = []
        
    
    def run(self, tout=50):
        ###########################################
        # begin loop over time
        ###########################################
        self.yr = self.ts - 1  # so the climate init works properly
        self.steps = []
        idx_out = 0
        self.last_t = self.ts
        self.b = np.zeros_like(self.x, dtype='float')
        
        def step_time(t, _):    
            # t = self.delt*(i)  # time in years 
            print(t)
            # define climate if it is the start of a new year
            try:
                delt = t - self.steps[-1]
            except:
                delt = t
                
            
            if (np.floor(t) > self.yr):  # equal to for first step
                self.yr = self.yr+1
                print(f'yr = {self.yr} | timestep = {t}')
                self.P = np.ones_like(self.x, dtype='float') * (self.Pbar + self.sigP * self.rng.standard_normal(1))
                self.T_wk = (self.Tbar + self.sigT * self.rng.standard_normal(1)) * np.ones_like(self.x) - self.gamma*self.zb  # todo: this should be based on the ice surface?
                # T_wk = (self.Tbar) * \
                #        np.ones_like(self.x, dtype='float') - \
                #        self.gamma * self.zb + \
                #        yr * 0.01
                
                self.melt = np.clip(self.mu * self.T_wk, 0, None)
                self.b = (self.P - self.melt) * delt
        
            
        ###########################################
        # begin loop over space
        ###########################################

            self.Qp = np.zeros_like(self.x, dtype='float')  # Qp equals j+1/2 flux
            self.Qm = np.zeros_like(self.x, dtype='float')  # Qm equals j+1/2 flux
            self.dhdt = np.zeros_like(self.x, dtype='float')  # zero out thickness rate of change
            for j in np.arange(0, self.nxs, 1):
                if j == 0:
                    h_ave = (self.h[0] + self.h[1]) / 2
                    dhdx = (self.h[1] - self.h[0]) / (self.delx/2)
                    dzdx = (self.dzbdx[0] + self.dzbdx[1]) / 2
                    self.Qp[0] = -self.K * (dhdx + dzdx)**3 * h_ave**5 * delt  # flux at plus half grid point
                    self.Qm[0] = 0  # flux at minus half grid point
                    self.dhdt[0] = (self.b[0] - self.Qp[0]) / (self.delx / 2)
                    
                elif (self.h[j] <= 0) & (self.h[j - 1] > 0):  # glacier toe condition
                    h_ave = self.h[j - 1] / 2
                    dhdx = -self.h[j - 1] / self.delx  # correction inserted ght nov-24-04
                    dzdx = (self.dzbdx[j - 1] + self.dzbdx[j]) / 2
                    self.Qm[j] = -self.K * (dhdx + dzdx)**3 * h_ave**5 * delt
                    self.Qp[j] = 0
                    self.dhdt[j] = (self.b[j] + self.Qm[j]) / (self.delx/2)
                    edge = j  # index of glacier toe - used for fancy plotting
                    
                elif (self.h[j] <= 0) & (self.h[j - 1] <= 0):  # beyond glacier toe - no glacier flux
                    self.dhdt[j] = self.b[j]
                    self.Qp[j] = 0
                    self.Qm[j] = 0
                    
                else:  # within the glacier
                    h_ave = (self.h[j + 1] + self.h[j]) / 2
                    dhdx = (self.h[j + 1] - self.h[j]) / self.delx  # correction inserted ght nov-24-04
                    dzdx = (self.dzbdx[j] + self.dzbdx[j + 1]) / 2
                    self.Qp[j] = -self.K * (dhdx + dzdx)**3 * h_ave**5 * delt
                    
                    h_ave = (self.h[j - 1] + self.h[j]) / 2
                    dhdx = (self.h[j] - self.h[j - 1]) / self.delx
                    dzdx = (self.dzbdx[j] + self.dzbdx[j - 1]) / 2
                    self.Qm[j] = -self.K * (dhdx + dzdx)**3 * h_ave**5 * delt
                    self.dhdt[j] = (self.b[j] - (self.Qp[j] - self.Qm[j])) / self.delx

                # self.dhdt[self.nxs] = 0  # enforce no change at boundary
                
                
            # end of loop over space
            self.dhdt = np.nan_to_num(self.dhdt)
            self.h = self.h + self.dhdt
            self.h = np.clip(self.h, a_min=0, a_max=None)

            # end of loop over time
            if len(self.steps) % tout == 0:
                print(f'output saved at time {t}')
                # np.argmin(np.gradient(res.h.to_numpy())[1], axis=1)
                edge = np.argmin(np.gradient(self.h)) #* self.delx
                self.edge_out.append(edge)
                self.dhdt_out.append(self.dhdt)
                self.h_out.append(self.h)
                self.Qp_out.append(self.Qp)
                self.Qm_out.append(self.Qm)
                self.P_out.append(self.P)
                self.T_out.append(self.T_wk)
                self.b_out.append(self.b)
                self.dzbdx_out.append(self.dzbdx)
                self.zb_out.append(self.zb)
                self.melt_out.append(self.melt)
            
            # to get delt
            self.steps.append(t)
            
            # diagnostic result for integration step sizing
            #return self.Qp + self.Qm
            return self.dhdt
        
        
        # run the model with rk4
        Qp0 = np.zeros_like(self.x, dtype='float')  # initial value for diagnostic value
        rk4 = scipy.integrate.RK45(step_time, t_bound=self.tf, max_step=1, t0=self.ts, y0=Qp0, first_step=0.0025, vectorized=True)
        while True:
            try:
                rk4.step()
            except:
                print('Instability :(')
                break

        # collect results
        res = xr.Dataset(data_vars=dict(
            edge=(['t'], np.array(self.edge_out)),
            dzbdx=(['t', 'x'], np.vstack(self.dzbdx_out)),
            zb=(['t', 'x'], np.vstack(self.zb_out)),
            dhdt=(['t', 'x'], np.vstack(self.dhdt_out)),
            h=(['t', 'x'], np.vstack(self.h_out)),
            Qp=(['t', 'x'], np.vstack(self.Qp_out)),
            Qm=(['t', 'x'], np.vstack(self.Qm_out)),
            P=(['t', 'x'], np.vstack(self.P_out)),
            T=(['t', 'x'], np.vstack(self.T_out)),
            b=(['t', 'x'], np.vstack(self.b_out)),
            melt=(['t', 'x'], np.vstack(self.melt_out)),
        ),
        coords=dict(
            x=('x', self.x),
            t=('t', np.arange(0, rk4.nfev // tout, 1)),
            
        ),
        attrs=dict(
            
        ))
        
        self.res = res

        
        #%%

        mpl.rcParams["image.aspect"] = 'auto'
        fig, ax = plt.subplots(1,1, figsize=(4, 6), dpi=150)
        im = ax.imshow(res.h.to_numpy(), norm=mpl.colors.Normalize(vmin=-0.25, vmax=500))
        ax.set_title('h')
        plt.colorbar(im, ax=ax)
        plt.show()

        fig, ax = plt.subplots(1,1, figsize=(4, 6), dpi=150)
        im = ax.imshow(res.T.to_numpy(), norm=mpl.colors.Normalize(vmin=-20, vmax=20))
        ax.set_title('T')
        plt.colorbar(im, ax=ax)
        plt.show()

        fig, ax = plt.subplots(1,1, figsize=(4, 6), dpi=150)
        im = ax.imshow(res.melt.to_numpy(), norm=mpl.colors.Normalize(vmin=-1, vmax=15))
        ax.set_title('melt')
        plt.colorbar(im, ax=ax)
        plt.show()
        
        
        # col 1
        fig, ax = plt.subplots(1,1, figsize=(4, 6), dpi=150)
        im = ax.imshow(res.Qp.to_numpy(), vmin=0, vmax=100)
        ax.set_title('Qp')
        plt.colorbar(im, ax=ax)
        plt.show()

        fig, ax = plt.subplots(1,1, figsize=(4, 6), dpi=150)
        im = ax.imshow(res.Qm.to_numpy(), vmin=0, vmax=100)
        ax.set_title('Qm')
        plt.colorbar(im, ax=ax)
        plt.show()

        fig, ax = plt.subplots(1,1, figsize=(4, 6), dpi=150)
        norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=-0.25, vmax=0.25)
        im = ax.imshow(np.gradient(res.h.to_numpy())[0], cmap='bwr_r', norm=norm)
        ax.set_title('gradient(h)[0]')
        plt.colorbar(im, ax=ax, norm=norm)
        plt.show()
        
        # col 2
        fig, ax = plt.subplots(1,1, figsize=(4, 6), dpi=150)
        norm = mpl.colors.SymLogNorm(linthresh=0.01, linscale=0.01)
        im = ax.imshow(res.dhdt.to_numpy(), cmap='bwr_r', norm=norm)
        ax.set_title('dhdt')
        plt.colorbar(im, ax=ax, norm=norm)
        plt.show()
        
        fig, ax = plt.subplots(1,1, figsize=(4, 6), dpi=150)
        norm = mpl.colors.SymLogNorm(linthresh=0.01, linscale=1, vmin=-10, vmax=10)
        im = ax.imshow(res.b.to_numpy(), cmap='bwr_r', norm=norm)
        ax.set_title('b')
        plt.colorbar(im, ax=ax, norm=norm)
        plt.show()

        fig, ax = plt.subplots(1,1, figsize=(4, 6), dpi=150)
        im = ax.imshow(res.P.to_numpy())
        ax.set_title('P')
        plt.colorbar(im, ax=ax)
        plt.show()

        # col 3
        fig, ax = plt.subplots(1,1, figsize=(4, 6), dpi=150)
        norm = mpl.colors.TwoSlopeNorm(vcenter=0)
        im = ax.imshow(np.gradient(res.h.to_numpy())[1], cmap='bwr_r',
                             norm=norm)
        ax.set_title('gradient(h)[1]')
        plt.colorbar(im, ax=ax, norm=norm)
        plt.show()
        
        fig, ax = plt.subplots(1,1, figsize=(4, 6), dpi=150)
        edge = res.edge.to_series()
        im = ax.plot(edge, edge.index)
        ax.invert_yaxis()
        ax.set_title('edge')
        plt.show()

        fig, ax = plt.subplots(1,1, figsize=(4, 6), dpi=150)
        norm = mpl.colors.SymLogNorm(linthresh=0.01, linscale=0.01)
        im = ax.imshow(res.Qm.to_numpy() - res.Qp.to_numpy(), norm=norm)
        ax.set_title('Qm - Qp')
        plt.colorbar(im, ax=ax, norm=norm)
        plt.show()


###############################################################################            
class flowline2d:
    def _init_plot(self):
        fig = plt.figure(figsize=(12, 10), dpi=200)
        gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=(1, 1, 1, 1))
        ax = np.empty((4, 2), dtype='object')
        plt.show(block=False)

        ax[0, 0] = fig.add_subplot(gs[0, 0])
        ax[0, 0].set_xlabel('Time (years)')
        ax[0, 0].set_ylabel('Glacier Area ($km^2$)')

        ax[1, 0] = fig.add_subplot(gs[1, 0])
        ax[1, 0].set_xlabel('Time (years)')
        ax[1, 0].set_ylabel('Equilibrium Line Altitude (m)')

        ax[0, 1] = fig.add_subplot(gs[0:2, 1])
        ax[0, 1].set_xlabel('Distance (km)')
        ax[0, 1].set_ylabel('Elevation (m)')

        ax[2, 0] = fig.add_subplot(gs[2, 0])
        ax[2, 0].set_ylabel('T ($^o$C)')

        ax[2, 1] = fig.add_subplot(gs[2, 1])
        ax[2, 1].set_ylabel('L (km)')

        ax[3, 0] = fig.add_subplot(gs[3, 0])
        ax[3, 0].set_ylabel('Bal (m $yr^{-1}$)')
        ax[3, 0].set_xlabel('Time (years)')

        ax[3, 1] = fig.add_subplot(gs[3, 1])
        ax[3, 1].set_xlabel('Time (years)')
        ax[3, 1].set_ylabel('Cum. bal. (m)')

        for axis in ax.ravel():
            if axis is not None:  # this handles gridspec col/rowspans > 1
                axis.grid(axis='both', alpha=0.5)
                axis.set_axisbelow(True)
        plt.tight_layout()

        return fig, ax

    def __init__(self, x_gr, zb_gr, x_geom, w_geom, x_init, h_init, temp, xmx,
                 Trand=None, Prand=None, t_stab=50, bed_shape='rectangular',
                 t0=0, n=3, gamma=6.5e-3, mu=0.65, g=9.81, rho=916.8, fd=1.9e-24, fs=5.7e-20, sigT=0.8, sigP=1.0,
                 T0=14.70, P0=5, delx=50, delt=0.0125 / 8, ts=0, tf=2025, dt_plot=100, rt_plot=True, xlim0=1500):
        """2d flowline model

        This module demonstrates documentation as specified by the `NumPy
        Documentation HOWTO`_. Docstrings may extend over multiple lines. Sections
        are created with a section header followed by an underline of equal length.
        
        Parameters
        ----------
        x_gr : array-like
            x dimension of bed topography; m
        zb_gr : array-like
            z dimension of bed topography; m
        x_geom : array-like
            x dimension of glacier width; m
        w_geom : array-like
            width of glacier along flowline; m
        x_init : array-like
            x dimension of initial thickness profile; m
        h_init : array-like
            initial ice thickness profile; m
        Trand : array-like
            random state for temperature
        Prand : array-like
            random state for precipitation
        rho : float
            Ice density kg/m^3
        g : float
            Gravity m/s^2
        mu : float
            Melt rate m/yr/degC
        n : float
            Glenn's flow law parameter. Default n = 3
        gamma : float
            Temperature lapse rate degC/km
        sigT : float
            Temperature standard deviation degC
        sigP : float
            Precipitation standard deviation m/yr
        T0 : float
            Baseline temperature degC
        fd : float
            Deformation parameter Pa^-3 s^-2
        fs : float
            Sliding parameter Pa^-3 s^-1 m^2
        xmx : int
            Domain size in m
        delx : int
            Grid spacing in m
        delt : float
            Time step in yrs suitable for 200m yr^-1
        ts : float
            Starting time yr
        tf : float
            Ending time yr
        dt_plot : int
            Plotting interval yr
        rt_plot : bool
            Whether there should be real time plotting while the model is running.
        xlim0 : float
            Left limit for plots (yrs)
    
    
        Returns
        -------
        bool
            True if successful, False otherwise.
        
        """
    
        
        # # How well do the equilibrium responses match?
        # # How well does the timescale match?
        # # What is the distribution of trends?
        # # How does the run length work? Does it follow a Poisson process?
        # # Is the dynamical model consistent with a Gaussian pdf?
        @njit(fastmath={'contract', 'arcp', 'nsz', 'ninf', 'nnan'})
        def space_loop(h, b):
            Qp = np.zeros(x.size)  # Qp equals j+1/2 flux
            Qm = np.zeros(x.size)  # Qm equals j-1/2 flux
            dhdt = np.zeros(x.size)  # zero out thickness rate of change array

            # -----------------------------------------
            # begin loop over space
            # -----------------------------------------
            for j in range(0, nxs - 1):  # this is a kloudge -fix sometime
                if j == 0:
                    h_ave = (h[0] + h[1]) / 2
                    dhdx = (h[1] - h[0]) / delx
                    dzdx = (dzbdx[0] + dzbdx[1]) / 2
                    Qp[0] = -(dhdx + dzdx)**3 * h_ave**4 * (rho * g)**3 * (
                            fd * h_ave + fs / h_ave)  # flux at plus half grid point
                    Qm[0] = 0  # flux at minus half grid point
                    dhdt[0] = b[0] - Qp[0] / (delx / 2) - (Qp[0] + Qm[0]) / (2 * w[0]) * dwdx[0]
                elif (h[j] == 0) & (h[j - 1] > 0):  # glacier toe condition
                    Qp[j] = 0
                    h_ave = h[j - 1] / 2
                    dhdx = -h[j - 1] / delx  # correction inserted ght nov-24-04
                    dzdx = (dzbdx[j - 1] + dzbdx[j]) / 2
                    Qm[j] = -(rho * g)**3 * h_ave**4 * (dhdx + dzdx)**3 * (fd * h_ave + fs / h_ave)
                    dhdt[j] = b[j] + Qm[j] / delx - (Qp[j] + Qm[j]) / (2 * w[j]) * dwdx[j]
                    edge = j  # index of glacier toe - used for fancy plotting
                elif (h[j] <= 0) & (h[j - 1] <= 0):  # beyond glacier toe - no glacier flux
                    dhdt[j] = b[j]  # todo: verify that this is actually being evaluated
                    Qp[j] = 0
                    Qm[j] = 0
                else:  # within the glacier
                    h_ave = (h[j + 1] + h[j]) / 2
                    dhdx = (h[j + 1] - h[j]) / delx  # correction inserted ght nov-24-04
                    dzdx = (dzbdx[j] + dzbdx[j + 1]) / 2
                    Qp[j] = -(rho * g)**3 * h_ave**4 * (dhdx + dzdx)**3 * (fd * h_ave + fs / h_ave)
                    h_ave = (h[j - 1] + h[j]) / 2
                    dhdx = (h[j] - h[j - 1]) / delx
                    dzdx = (dzbdx[j] + dzbdx[j - 1]) / 2
                    Qm[j] = -(rho * g)**3 * h_ave**4 * (dhdx + dzdx)**3 * (fd * h_ave + fs / h_ave)
                    dhdt[j] = b[j] - (Qp[j] - Qm[j]) / delx - (Qp[j] + Qm[j]) / (2 * w[j]) * dwdx[j]
                dhdt[nxs - 1] = 0  # enforce no change at boundary
            # ----------------------------------------
            # end loop over space
            # ----------------------------------------
            h = np.clip(h + dhdt * delt, 0, 10000)
            return h, edge
        
        if rt_plot:
            plt.ioff()
            mpl.use('qt5agg')

        # #-----------------
        # #define parameters
        # #-----------------
        nxs = round(xmx / delx)  # number of grid points
        nts = round(np.floor((tf - ts) / delt))  # number of time steps ('round' used because need nts as integer)
        nyrs = tf - ts
        
        
        # xmx = xmx + (delx - (xmx % delx))  # round up xmx to the nearest delx
        xmx = xmx - (xmx % delx)  # round down xmx to the nearest delx
        x = np.arange(0, xmx, delx)  # x array
        
        fd = fd * np.pi * 1e7
        fs = fs * np.pi * 1e7  # convert from seconds to years
        
        # ---------------------------------
        # different glacier bed geometries
        # ---------------------------------
        zb = interp1d(x_gr, zb_gr)
        zb = zb(x)
        w = interp1d(x_geom, w_geom)
        w = w(x)
        w = np.clip(w, 700, None)
        # w = 500 * np.ones(x.size)  # no width data in this version
        dzbdx = np.gradient(zb, x)  # slope of glacer bed geometries.
        dwdx = np.gradient(w, x)

        # ----------------------------------------------------------------------------------------------
        # alternative starting geometry for a glacierusing results from a previus integration
        # stored in glac_init.mat
        # probably only worth it for mass balance parameterizations similar to the default one we tried
        # ----------------------------------------------------------------------------------------------
        
        h0 = interp1d(x_init, h_init, 'linear')
        h0 = h0(x)
        h = np.full(x.size, fill_value=h0)  # initialize height array

        # initialize climate forcing
        Tp = sigT * Trand[0:nyrs + 1]
        Pp = sigP * Prand[0:nyrs + 1]  # initialize climate forcing
        # Tp[:] = 0  # todo: testing ,remove
        # Pp[:] = 0  # todo: testing ,remove
        Tp[:t_stab] = 0  # todo: turn this into an argument

        # -----------------------------------------
        # begin loop over time
        # -----------------------------------------
        yr = 0
        idx_out = 0
        deltout = 1  # this is supposed to be one... I forget why...
        nouts = round(nts * delt)
        edge_out = np.full(nouts, fill_value=np.nan, dtype='float')
        t_out = np.full(nouts, fill_value=np.nan, dtype='float')
        T_out = np.full(nouts, fill_value=np.nan, dtype='float')
        P_out = np.full(nouts, fill_value=np.nan, dtype='float')
        bal_out = np.full(nouts, fill_value=np.nan, dtype='float')
        ela_out = np.full(nouts, fill_value=np.nan, dtype='float')
        area_out = np.full(nouts, fill_value=np.nan, dtype='float')
        h_out = np.full((nouts, nxs), fill_value=np.nan, dtype='float')
        
        if rt_plot:
            fig, ax = self._init_plot()

        for i in range(0, nts):
            t = delt * i  # time in years 
            # define climate every year
            if (t == np.floor(t)):
                yr = yr + 1
                print(f'yr = {yr}')
                P = np.ones(x.size) * (P0 + 1 * Pp[yr])
                T_wk = (T0 + 1 * Tp[yr]) * np.ones(x.size) - gamma * (zb + h)
                T_wk = T_wk + temp(yr)
                melt = np.core.umath.clip(mu * T_wk, 0, 100000)
                b = P - melt

            # loop over space (solve SIA)
            # this is the entire model, really
            h, edge = space_loop(h, b)

            # ----------------------------
            # plot glacier every so often
            # save h , and edge
            # ----------------------------

            if t / deltout == np.floor(t / deltout):
                area = np.trapz(w[:edge+1], dx=delx)
                
                # Save outputs
                T_out[idx_out] = T_wk[0]  # temp at the top of the glacier todo: verify this
                P_out[idx_out] = P[0]
                t_out[idx_out] = t
                edge_out[idx_out] = edge * delx
                h_out[idx_out, :] = h
                area_out[idx_out] = area
                # edge_out[idx_out] = delx * max(h[h > 10])
                bal = b * w * delx  # mass added in a given cell units are m^3 yr^-1
                bal_out[idx_out] = np.trapz(bal[:edge+1])  # should add up all the mass up to the edge, and be zero in equilibrium (nearly zero)
                ela_idx = np.abs(b).argmin()
                ela_out[idx_out] = zb[ela_idx] + h[ela_idx]
                
                idx_out = idx_out + 1
            if ((t / dt_plot == np.floor(t / dt_plot)) | (i == nts - 1)) & rt_plot:  # force plotting on the last time step
                print('outputting')
                pad = 10
                x1 = x[:edge + pad]
                z0 = zb[:edge + pad]
                z1 = zb[:edge + pad] + h[:edge + pad]
                
                try:
                    ax[0, 1].collections[0].remove()  # remove the glacier profile before redrawing
                except:
                    pass
                poly = ax[0, 1].fill_between(x1 / 1000, z0, z1, fc='lightblue')
                ax[0, 1].plot(x1 / 1000, z0, c='black', lw=2, )
                ax[0, 0].plot(t_out, scipy.ndimage.uniform_filter1d(area_out/1e6, 20, mode='mirror'), c='black')
                ax[1, 0].plot(t_out, scipy.ndimage.uniform_filter1d(ela_out, 20, mode='mirror'), c='black')
                ax[2, 0].plot(t_out, T_out, c='blue', lw=0.25)
                ax[2, 1].plot(t_out, scipy.ndimage.uniform_filter1d(edge_out, 20, mode='mirror') / 1000, c='black', lw=2)
                ax[3, 0].plot(t_out, bal_out / area_out, c='blue', lw=0.25)
                ax[3, 1].plot(t_out, scipy.ndimage.uniform_filter1d(np.cumsum(bal_out / area_out), 20, mode='mirror'), c='blue', lw=2)
                
                # update the plot
                fig.canvas.flush_events()
                fig.canvas.draw()

                # -----------------------------------------
                # end loop over time
                # -----------------------------------------
        if rt_plot:
            ax[0, 1].set_xlim(0, x1.max() / 1000 * 1.1)
            ax[1, 0].set_xlim(xlim0, tf)
            ax[2, 0].plot(t_out, scipy.ndimage.uniform_filter1d(T_out, 20, mode='mirror'), c='blue', lw=2)
            ax[2, 0].set_xlim(xlim0, tf)
            ax[2, 1].set_xlim(xlim0, tf)
            ax[3, 0].plot(t_out, scipy.ndimage.uniform_filter1d(bal_out / area_out, 20, mode='mirror'), c='blue', lw=2)
            ax[3, 0].set_xlim(xlim0, tf)
            ax[3, 1].plot(t_out, scipy.ndimage.uniform_filter1d(np.cumsum(bal_out / area_out), 20, mode='mirror'), c='blue', lw=2)
            ax[3, 1].set_xlim(xlim0, tf)
            plt.draw()
            
        
        # collect output
        self.T = T_out
        self.P = P_out
        self.t = t_out + ts
        self.edge = edge_out
        self.bal = bal_out
        self.ela = ela_out
        self.h = h_out
        self.x = x
        self.zb = zb
        self.ts = ts
        self.tf = tf
        self.nxs = nxs
        self.delx = delx
        self.area = area_out
        
        return None
        
    def plot(self, xlim0=1500, compare_fp=None):
        """This is a docstring
        
        This is the longer portion of the docstring.
        
        Parameters
        ----------------
        xlim0 : float
            left x-limit for figure (years)
            
        Returns
        ----------------
        fig : Figure
            It's a figure??
        
        """
        pad=10
        pedge = int(self.edge[-1]/self.delx + pad)
        self.pedge = pedge
        x1 = self.x[:pedge]
        z0 = self.zb[:pedge]
        z1 = z0 + self.h[-1, :pedge]
        
        fig, ax = self._init_plot()
        poly1 = ax[0, 1].fill_between(x1 / 1000, z0, z1, fc='lightblue', ec='lightblue')
        ax[0, 1].plot(x1 / 1000, z0, c='black', lw=2, )
        ax[0, 0].plot(self.t, scipy.ndimage.uniform_filter1d(self.area/1e6, 20, mode='mirror'), c='black')
        ax[1, 0].plot(self.t, scipy.ndimage.uniform_filter1d(self.ela, 20, mode='mirror'), c='blue')
        ax[1, 0].set_xlim(xlim0, self.tf)
        ax[2, 1].set_xlim(0, x1.max() / 1000 * 1.1)
        ax[2, 0].plot(self.t, self.T, c='blue', lw=0.25)
        ax[2, 0].plot(self.t, scipy.ndimage.uniform_filter1d(self.T, 20, mode='mirror'), c='blue', lw=2)
        ax[2, 0].set_xlim(xlim0, self.tf)
        ax[2, 1].plot(self.t, scipy.ndimage.uniform_filter1d(self.edge, 20, mode='mirror') / 1000, c='black', lw=2)
        ax[2, 1].set_xlim(xlim0, self.tf)
        ax[3, 0].plot(self.t, self.bal / self.area, c='blue', lw=0.25)
        ax[3, 0].plot(self.t, scipy.ndimage.uniform_filter1d(self.bal / self.area, 20, mode='mirror'), c='blue', lw=2)
        ax[3, 0].set_xlim(xlim0, self.tf)
        ax[3, 1].plot(self.t, scipy.ndimage.uniform_filter1d(np.cumsum(self.bal / self.area), 20, mode='mirror'), c='blue', lw=2)
        ax[3, 1].plot(self.t, scipy.ndimage.uniform_filter1d(np.cumsum(self.bal / self.area), 20, mode='mirror'), c='blue', lw=2)
        ax[3, 1].set_xlim(xlim0, self.tf)

        # plot extras
        if compare_fp:
            with open(compare_fp, 'rb') as f:
                anth = pickle.load(f)
            #anth = pd.DataFrame().from_dict(anth)
            anth_h = anth.pop('h')
            anth = pd.DataFrame.from_dict(anth)
            self.anth = anth
            
            pad=10
            pedge = int(anth['edge'].iloc[-1]/self.delx + pad)
            self.pedge = pedge
            x2 = self.x[:pedge]
            z1 = self.zb[:pedge]
            z2 = z1 + anth_h[-1][:pedge]

            poly2 = ax[0, 1].fill_between(x2 / 1000, z1, z2, fc='none', ec='lightblue')
            if len(x1) > len(x2):
                poly1.set_hatch('....')
                poly1.set_facecolor('none')
                poly2.set_facecolor('lightblue')
            else:
                poly1.set_facecolor('lightblue')
                poly2.set_hatch('....')
                
            
            anth['T_sm'] = scipy.ndimage.uniform_filter1d(anth['T'], 20, mode='mirror')
            anth['edge_sm'] = scipy.ndimage.uniform_filter1d(anth['edge'], 20, mode='mirror') / 1000
            anth['bal_sm'] = scipy.ndimage.uniform_filter1d(anth['bal'] / anth['area'], 20, mode='mirror')
            anth['cumbal_sm'] = scipy.ndimage.uniform_filter1d(np.cumsum(anth['bal'] / anth['area']), 20, mode='mirror')
            anth['ela_sm'] = scipy.ndimage.uniform_filter1d(anth['ela'], 20, mode='mirror')
            #anth = anth.iloc[1850:]
            ax[0, 0].plot(anth.t, anth['area']/1e6, c='black', lw=2, ls='dashed')
            ax[1, 0].plot(anth.t, anth['ela_sm'], c='red', lw=2,)
            ax[2, 0].plot(anth.t, anth['T_sm'], c='red', lw=2)
            ax[2, 0].plot(anth.t, anth['T'], c='red', lw=0.25)
            ax[2, 1].plot(anth.t, anth['edge_sm'], c='black', lw=2, ls='dashed')
            ax[3, 0].plot(anth.t, anth['bal_sm'], c='red', lw=2)
            ax[3, 0].plot(anth.t, anth['bal'] / anth.area, c='red', lw=0.25)
            ax[3, 1].plot(anth.t, anth['cumbal_sm'], c='red', lw=2)
            fig.show()
        
        return fig

    def to_pickle(self, fp):
        # save output?
        output = dict(t=self.t, T=self.T, P=self.P, edge=self.edge, bal=self.bal, ela=self.ela, area=self.area, h=self.h, x=self.x)
        with open(fp, 'wb') as f:
            pickle.dump(output, f)
        
        return None
