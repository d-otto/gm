# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import collections

import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import scipy.io
from numpy.random import default_rng
import xarray as xr
import copy

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
        self.dt = np.diff(self.ts+1, prepend=-1)  # works idk why
        
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

        self.mode=mode
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
        if bt is not None:
            self.b_p = bt
            self.bt_p = self.bt_eq + self.b_p
        
        # convenience renaming
        tau = self.tau
        L_bar = self.L_bar
        H = self.H
        eps = self.eps
        ts = self.ts
        
        # fluxes (discritized versions of eq14)
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
    
            
    def discrete(self):
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
                         self.dt[i]**3 * tau / (eps * tau)**3 * (beta[i] * self.P_p[i - 3] - alpha * self.T_p[i - 3])
                
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
        ''' Mostly correct? Why do I have to multiply by 1j?'''
        
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
    
    def __init__(self, L0, h0, W, dzdx, mu, Tbar, sigT, Pbar, sigP, gamma=6.5e-3, rho=916.8, f_d=1.9e-24, f_s=5.7e-20):
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


        # set model parameters
        self.ts = 0  # starting time
        self.tf = 100  # final time
        self.delt = 0.05  # # time step in yrs
        # delt = 0.00125# # time step in yrs. needed if delx = 50

        self.xmx = L0  # the domain size in m
        self.delx = 200  # grid spacing in m
        self.nxs = int(np.round(self.xmx / self.delx)) - 1  # number of grid points
        self.x = np.arange(0, self.xmx, self.delx)  # x array

        self.nts = int(np.floor(
            (self.tf - self.ts) / self.delt))  # number of time steps ('round' used because need nts as integer)


        # climate parameters
        self.Tbar = Tbar    # average temperature at z=0 [^oC]
        self.sigT = sigT    # standard deviation of temperature [^oC]
        
        self.Pbar = Pbar    # average value of accumulation [m yr^-1]
        self.sigP = sigP    # standard deviation of accumulation [m yr^-1]
        
        self.gamma = gamma  # lapse rate  [K m^-1]
        self.mu = mu      # melt factor for ablation [m yr^-1 K^-1]
        
        
        self.rng = default_rng()
        
        
       

        # glacier geometry
        self.delta = 15e3 / np.log(3)
        self.zb = 3000 * np.exp(-self.x / self.delta)
        self.dzbdx = np.gradient(self.zb, self.x)  # slope of glacer bed geometries.


        # find zeros of flux to find glacier length
        # Note oerlemans value is c=10
        self.b0 = (3 - (4 / 5000) * self.x)  # mass balance in m/yr
        self.c = 1  # equals 2tau/(rho g) plastic ice sheet constant from paterson
        #self.c = 10
        idx = np.where(np.cumsum(self.b0) < 0)  # indices where flux is less than zero
        idx = np.min(idx)
        self.L = self.x[idx]  # approximate length of glacier

        ###########################################
        # plastic glacier initial profile
        ###########################################
        self.h = np.zeros_like(self.x, dtype='float')  # zero out initial height array
        
        self.h[0:idx-1] = np.sqrt(self.c * (self.L - self.x[0:idx-1]))  # plastic ice profile as initial try

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
        
    
    # def run(self):
        ###########################################
        # begin loop over time
        ###########################################
        yr = 0
        idx_out = 0
        b = np.zeros_like(self.x, dtype='float')
        for i in np.arange(0, self.nts-1, 1):
            
            t = self.delt*(i)  # time in years 
            
            # define climate if it is the start of a new year
            if t == np.floor(t):
                yr = yr+1
                print(f'yr = {yr} | timestep = {i}')
                P = np.ones_like(self.x, dtype='float') * (self.Pbar + self.sigP * self.rng.standard_normal(1))
                T_wk = (self.Tbar + self.sigT * self.rng.standard_normal(1)) * np.ones_like(self.x) - self.gamma*self.zb
                # T_wk = (self.Tbar) * \
                #        np.ones_like(self.x, dtype='float') - \
                #        self.gamma * self.zb + \
                #        yr * 0.01
                
                self.melt = np.clip(self.mu * T_wk, 0, None)
                b = P - self.melt
                

            
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
                    self.Qp[0] = -self.K * (dhdx + dzdx)**3 * h_ave**5  # flux at plus half grid point
                    self.Qm[0] = 0  # flux at minus half grid point
                    self.dhdt[0] = b[0] - self.Qp[0] / (self.delx / 2)
                elif (self.h[j] <= 0) & (self.h[j - 1] > 0):  # glacier toe condition
                    h_ave = self.h[j - 1] / 2
                    dhdx = -self.h[j - 1] / self.delx  # correction inserted ght nov-24-04
                    dzdx = (self.dzbdx[j - 1] + self.dzbdx[j]) / 2
                    self.Qm[j] = -self.K * (dhdx + dzdx)**3 * h_ave**5
                    self.Qp[j] = 0
                    self.dhdt[j] = b[j] + self.Qm[j] / (self.delx/2)
                    edge = j  # index of glacier toe - used for fancy plotting
                elif (self.h[j] <= 0) & (self.h[j - 1] <= 0):  # beyond glacier toe - no glacier flux
                    self.dhdt[j] = b[j]
                    self.Qp[j] = 0
                    self.Qm[j] = 0
                else:  # within the glacier
                    h_ave = (self.h[j + 1] + self.h[j]) / 2
                    dhdx = (self.h[j + 1] - self.h[j]) / self.delx  # correction inserted ght nov-24-04
                    dzdx = (self.dzbdx[j] + self.dzbdx[j + 1]) / 2
                    self.Qp[j] = -self.K * (dhdx + dzdx)**3 * h_ave**5
                    
                    h_ave = (self.h[j - 1] + self.h[j]) / 2
                    dhdx = (self.h[j] - self.h[j - 1]) / self.delx
                    dzdx = (self.dzbdx[j] + self.dzbdx[j - 1]) / 2
                    self.Qm[j] = -self.K * (dhdx + dzdx)**3 * h_ave**5
                    self.dhdt[j] = b[j] - (self.Qp[j] - self.Qm[j]) / self.delx

                # self.dhdt[self.nxs] = 0  # enforce no change at boundary
            
            # end of loop over space
            self.h = self.h + self.dhdt*self.delt
            self.h = np.clip(self.h, a_min=0, a_max=None)

            # end of loop over time
            # np.argmin(np.gradient(res.h.to_numpy())[1], axis=1)
            edge = np.argmin(np.gradient(self.h)) #* self.delx
            self.edge_out.append(edge)
            self.dhdt_out.append(self.dhdt)
            self.h_out.append(self.h)
            self.Qp_out.append(self.Qp)
            self.Qm_out.append(self.Qm)
            self.P_out.append(P)
            self.T_out.append(T_wk)
            self.b_out.append(b)
            self.dzbdx_out.append(self.dzbdx)
            self.zb_out.append(self.zb)
            self.melt_out.append(self.melt)
        
        
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
            t=('t', np.arange(1, self.nts, 1)),
            
        ),
        attrs=dict(
            
        ))

        
        #%%
        fig, ax = plt.subplots(3, 4, figsize=(12, 8))
        
        
        im = ax[0, 0].imshow(res.h.to_numpy(), vmin=-1, vmax=300)
        ax[0,0].set_xlabel('h')
        plt.colorbar(im, ax=ax[0, 0])

        im = ax[1, 0].imshow(res.T.to_numpy())
        ax[1, 0].set_xlabel('T')
        plt.colorbar(im, ax=ax[1, 0])
        
        im = ax[2, 0].imshow(res.melt.to_numpy())
        ax[2, 0].set_xlabel('melt')
        plt.colorbar(im, ax=ax[2, 0])
        
        
        # col 1
        im = ax[0, 1].imshow(res.Qp.to_numpy(), vmin=0, vmax=1500)
        ax[0, 1].set_xlabel('Qp')
        plt.colorbar(im, ax=ax[0, 1])

        im = ax[1, 1].imshow(res.Qm.to_numpy(), vmin=0, vmax=1500)
        ax[1, 1].set_xlabel('Qm')
        plt.colorbar(im, ax=ax[1, 1])

        im = ax[2, 1].imshow(np.gradient(res.h.to_numpy())[0], cmap='bwr_r', norm=mpl.colors.TwoSlopeNorm(vcenter=0, vmin=-0.25, vmax=0.25))
        ax[2,1].set_xlabel('gradient(h)[0]')
        plt.colorbar(im, ax=ax[2, 1])
        
        
        # col 2
        im = ax[0, 2].imshow(res.dhdt.to_numpy(), cmap='bwr_r', norm=mpl.colors.TwoSlopeNorm(vcenter=0, vmin=-5, vmax=5))
        ax[0, 2].set_xlabel('dhdt')
        plt.colorbar(im, ax=ax[0, 2])

        im = ax[1, 2].imshow(res.b.to_numpy(), cmap='bwr_r', norm=mpl.colors.CenteredNorm(vcenter=0))
        ax[1,2].set_xlabel('b')
        plt.colorbar(im, ax=ax[1, 2])

        im = ax[2, 2].imshow(res.P.to_numpy())
        ax[2,2].set_xlabel('P')
        plt.colorbar(im, ax=ax[2, 2])


        # col 3
        im = ax[0, 3].imshow(np.gradient(res.h.to_numpy())[1], cmap='bwr_r',
                             norm=mpl.colors.TwoSlopeNorm(vcenter=0))
        ax[0,3].set_xlabel('gradient(h)[1]')
        plt.colorbar(im, ax=ax[0,3])

        im = ax[1, 3].plot(res.edge.to_series())
        ax[1, 3].set_xlabel('edge')


        [axis.set_aspect(0.025) for axis in ax.flatten()]
        ax[1, 3].set_aspect(20)
        plt.tight_layout()
        
        fig.show()
        
        
        
        

