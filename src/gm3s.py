# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import argparse
import scipy.io
from numpy.random import default_rng


class gm1s:
    
    #def __init__(self, mode='b', years=None, dt=1, mu=None, Atot=None, ATgt0=None, Aabl=None, w=None, H=None, gamma=None, dzdx=None, sigP=None, sigT=None, sigb=None, Tp=None, Pp=None, bp=None, tau=None, P0=None):
    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        mode : str, optional
            Calculate anomalies based on mass balance 'b' or length 'l'. Default is 'b'.
        years : int
            Length of integration (years).
        ts : int, optional
            Starting year. Default is 0.
        dt : numeric, optional
            Time step (years). Recommended to keep at 1 (default).
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
        
        self.run(**kwargs)
        
        
    def run(self, mode='b', years=None, dt=1, mu=None, Atot=None, ATgt0=None, Aabl=None, w=None, H=None, gamma=None, dzdx=None, sigP=None, sigT=None, sigb=None, Tp=None, Pp=None, bp=None, tau=None, P0=None):
        """

       Parameters
       ----------
       mode : str, optional
           Calculate anomalies based on mass balance 'b' or length 'l'. Default is 'b'.
       years : int
           Length of integration (years).
       ts : int, optional
           Starting year. Default is 0.
       dt : numeric, optional
           Time step (years). Recommended to keep at 1 (default).
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
        
        # unpack kwargs 
        # vars = locals().copy()
        # ns = argparse.Namespace()
        # for k, v in kwargs.items():
        #     if k not in vars:
        #         exec(f'{k} = {v}')  # Don't ever do this. Code would just become unreadable if I did setattr(self, k, v).
        
        nts = int((years) / dt)  # number of time steps
        t = np.arange(0, years, dt)  # array of times [yr]

        # note at this point you could create your own climate time series, using
        # random forcing, trends, oscillations etc.
        # Tp = array of melt-season temperature anomalise
        # Pp = array of accumlation anomalies
        # bp = array of mass balance anomalies
        if Tp is None:
            Tp = norm.rvs(scale=sigT, size=nts)
        if Pp is None:
            Pp = norm.rvs(scale=sigP, size=nts) + P0
        if bp is None and mode == 'b':
            bp = norm.rvs(scale=sigb, size=nts)
        
        ## linear model coefficients, combined from above parameters
        ## play with their values by choosing different numbers...
        alpha = -mu*ATgt0*dt/(w * H)
        beta = Atot*dt/(w * H)
        
        # glacier memory [ys]
        # this is the glacier response time  (i.e., memory) based on the above glacier geometry
        # if you like, just pick a different time scale to see what happens. 
        # Or also, use the simple, tau = hbar/b_term, if you know the terminus
        # balance rate from, e.g., observations
        if tau is None:
            tau = w * H / (mu * gamma * dzdx * Aabl)
        
        # coefficient needed in the model integrations
        # keep fixed - they are intrinsic to 3-stage model
        eps = 1/np.sqrt(3)
        phi = 1-dt/(eps*tau)
        
        L = np.zeros(nts)  # create array of length anomalies
        
        ## integrate the 3 stage model equations forward in time
        for i in range(5, nts):
            if mode == 'l': 
                L[i] = 3*phi*L[i-1]-3*phi**2*L[i-2]+1*phi**3*L[i-3] \
                         + dt**3*tau/(eps*tau)**3 * (beta*Pp[i-3] - alpha*Tp[i-3])
            
            if mode == 'b':
                L[i] = 3*phi*L[i-1]-3*phi**2*L[i-2]+1*phi**3*L[i-3] \
                         + dt**3*tau/(eps*tau)**3 * (beta*bp[i-3])
        
        # calculate function properties & spectral stuff
        sig_L = tau * dt / 2 * (alpha**2 * sigT**2 + beta**2 * sigP**2)  # variance of length in the one-stage model
        
        # assign all local variables as attributes
        vars = locals().copy()
        for k,v in vars.items():
            if k not in self.__dict__:
                setattr(self, k, v)
                

    def to_xarray(self):
        import xarray as xr

        ds = xr.Dataset(
            data_vars=dict(
                bp=('t', self.bp),
                Pp=('t', self.Pp),
                Tp=('t', self.Tp),
            ),
            coords=dict(
                t=self.t
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
# class flowline_1s:
#     '''
#     "Most numerical models
#     solve the shallow-ice equations (which neglect longitudinal
#     stresses; e.g. Hutter, 1983) and incorporate a representation
#     of basal sliding. For a one-dimensional flowline following the
#     longitudinal profile of a glacier"
#     
#     "The first equation represents local
#     mass conservation, while the second represents the transla-
#     tion and deformation of ice associated with shear stresses. In
#     combination, the equations have the form of a nonlinear
#     diffusion equation in thickness."
#     
#     '''
#     
#     def __init__(self):
#         ts =
#         nts = int((tf - ts) / dt)  # number of time steps
#         t = np.arange(ts, tf, dt)  # array of times [yr]
#         
#         
#         
#         dh = 
#         dF = 
#         
#         rho =
#         g =
#         f_s = 
#         h = 
#         dz_s = 
#         dx = 
#         x = 
#         
#         Fx = rho**3 * g**3 * (f_s*h**2 + f_s) * h**3 * (dz_s/dx)**3


################################################################################
class gm3s:
    """ Real 3-stage linear glacier model
    
    Assumes Lambda << L_bar and h2_p, h3_p ~= H. Therefore, V1_p ~= h1_p * L_bar, 
    V2_p ~= h2_p * lambda, and V3_p ~= (L_p - Lambda)*H ... (see figure)
    
    
    """
    
    def __init__(self, L, H, tau, dzdx, rho=916.8, g=-9.81, f_d=1.9e-24, f_s=5.7e-20):
        self.L_bar = L  # m
        self.H = H  # m 
        self.rho = rho  # kg/m^3
        self.g = g  # m/s^2
        self.f_d = f_d
        self.f_s = f_s
        self.tau = tau
        self.dzdx_s = dzdx
        
        
    def diff(self):
        tau1 = tau2 = tau3 = tau * eps

        F2_p = eps * H * (dL_p + L_p / (eps * tau))
        bd_p = dh1_p + h1_p / tau1
        h1_p = tau1 * tau2 / L_bar * (dF2_p + F2_p)

        # fluxes
        F1_p = h1_p * L_bar / tau1  # 13a
        F2_p = h2_p * Lambda / tau2  # 13b
        
        # volume changes (12a-c)
        dV1_p = bd_p * L_bar - F1_p  # w.r.t. time
        dV2_p = F1_p - F2_p  # w.r.t. time
        dV3_p = F2_p - L_p * bd_term  # w.r.t. time

        heq_p = bd_p * tau1
        Feq_p = bd_p * L_bar
        Leq_p = Feq_p / bd_term

        # fluxes (differential form of 3-stage)
        dh1_p = bd_p - h1_p / tau1  # 14a
        dF2_p = L_bar / (tau1 * tau2) * h1_p - F2_p / tau2  # 14b
        dL_p = F2_p / H - L_p / tau3  # 14c
        
    
    def discrete(self, years, dt, bt):
        # convenience renaming
        tau = self.tau
        L_bar = self.L_bar
        H = self.H
        eps = 1 / np.sqrt(3)
        
        # saving input parameters for use in other methods
        self.dt = dt
        self.bt = bt
        self.K = 1 - dt / (eps * tau)
        ts = np.array(range(0, years, dt))
        self.ts = ts
        
        # fluxes (discritized versions of eq14)

        n_steps = len(ts)
        self.h = np.zeros(n_steps)
        self.F = np.zeros(n_steps)
        self.L = np.zeros(n_steps)
        self.L_debug = np.zeros(n_steps)

        for t in ts:
            if t == ts[0]:
                continue  # first value is 0
                
            #self.bt[t] = self._bt(t-1)
            self.h[t] = (1 - dt/(eps*tau))*self.h[t-dt] + self.bt[t]
            self.F[t] = (1 - dt/(eps*tau))*self.F[t-dt] + L_bar/(eps*tau)**2*self.h[t]  # writing F2 as F
            self.L[t] = (1 - dt/(eps*tau))*self.L[t-dt] + self.F[t]/(eps*H)

            try:
                self.L_debug = dt*self.L_bar/(eps*self.H) * (dt/(eps*tau))**2 * self.bt[t-3]  # if I specified everything right, this should be the same is L[t]
            except:
                pass

    
    def arma(self):
        
        # discretized
        for t in T:
            # L_p[t] = 3*K*L_p[t-1] - 3*K**2*L_p[t-2] - 3*K**3*L_p[t-3]
            L_p[t] = dt * L_bar / (eps * H) * (dt / (eps * tau))**2 * bd[t - 3]
       
            
    def power_spectrum(self, freq, sig_L_1s):
        P0 = 4 * self.tau * sig_L_1s  # power spectrum in the limit f -> 0 using the variance from the 1s model
        P_spec = P0 * (1-self.K)**6 / (1 - 2 * self.K * np.cos(2*np.pi*freq*self.dt) + self.K**2)**3
        return P_spec
            
      
            
class flowline:
    
    def __init__(self, H, L0, dzdx, rho=916.8, g=9.81, f_d=1.9e-24, f_s=5.7e-20):
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
        
        self.L0 = L0
        self.H = H
        self.dzdx = dzdx
        self.rho = rho
        self.g = g
        self.f_s = f_s
        self.f_d = f_d
        
        ###########################################
        #define parameters
        ###########################################
        
        rng = default_rng()
        
        rho=910 #kg/m^3
        g=9.81# #m/s^2
        n=3#
        A=2.4e-24# #Pa^-3 s^-1
        K=2*A*(rho*g)**n/(n+2)
        K=K*np.pi*1e7  # make units m^2/yr
        K=K#

        xmx = 20000  # the domain size in m
        delx = 200  #grid spacing in m
        nxs = round(xmx/delx) - 1  #number of grid points

        #delt = 0.00125# # time step in yrs. needed if delx = 50
        delt = 0.025# # time step in yrs
        ts = 0  # starting time
        tf = 10000  # final time
        nts=int(np.floor((tf-ts)/delt) + 1)  #number of time steps ('round' used because need nts as integer)

        ###########################################
        # climate parameters
        ###########################################
        Tbar = 20.5    # average temperature at z=0 [^oC]
        sigT = 0.9     # standard deviation of temperature [^oC]
        
        Pbar = 3.0     # average value of accumulation [m yr^-1]
        sigP = 1.0     # standard deviation of accumulation [m yr^-1]
        
        gamma = 6.5e-3  # lapse rate  [K m^-1]
        mu = 0.5       # melt factor for ablation [m yr^-1 K^-1]
        
        x = np.arange(0, xmx, delx)  # x array

        ###########################################
        # define functions
        ###########################################
        b = (3-(4/5000)*x)  # mass balance in m/yr

        ###########################################
        # different glacier bed geometries
        ###########################################
        #zb=2000-.1*x; #bed profile in m
        #zb =  2000.*exp(-x/4000);
        delta = 15e3 / np.log(3)
        zb = 3000 * np.exp(-x/delta)
        #zb = 2000-(.1*x) + 200*(1-exp(-x/4000));
        #zb = 1000*ones(size(x));

        dzbdx = np.gradient(zb,x)  # slope of glacer bed geometries.
        #Q = (2 * x[j] - (4/5000) * (x[j]**2)/2) * (1e-7/np.pi)  # mass flux m^2/s

        # Not needed if you load your own initial profile in.
        ###########################################
        # find zeros of flux to find glacier length
        ###########################################
        # Note oerlemans value is c=10
        c = 1  #equals 2tau/(rho g) plastic ice sheet constant from paterson
        idx=np.where(np.cumsum(b)<0)  # indices where flux is less than zero
        idx = np.min(idx)
        L=x[idx]  #approximate length of glacier

        ###########################################
        # plastic glacier initial profile
        ###########################################
        h = np.zeros_like(x)  # zero out initial height array
        h[1:idx-1] = np.sqrt(c * (L-x[1:idx-1])) #plastic ice profile as initial try
        Qp = np.zeros_like(x)
        Qm = np.zeros_like(x)
        dhdt = np.zeros_like(x)
        
        
        nyrs = tf-ts
        Pout=np.zeros(nyrs)
        Tout=np.zeros(nyrs)
        ###########################################
        # begin loop over time
        ###########################################
        yr = 0
        idx_out = 0
        
        deltout = 5
        nouts = round(nts/5)
        edge_out = np.zeros(nouts)
        t_out = np.zeros(nouts)
        
        for i in range(0, nts):
        
            t = delt*(i-1) # time in years 
            
            # define climate if it is the start of a new year
            if t == np.floor(t):
                yr = yr+1
                print('yr = ', str(yr))
                P = np.ones_like(x) * (Pbar + sigP * rng.standard_normal(1))
                T_wk = (Tbar + sigT * rng.standard_normal(1)) * np.ones_like(x) - gamma*zb
                Pout[yr] = P[1]
                Tout[yr] = T_wk[1]
        #          if yr<500;
        #              P = ones(size(x))*(3.0);
        #              T_wk    = (17.5)*ones(size(x)) - 6.5e-3*zb;
        #          else
        #              P = ones(size(x))*(3.0);
        #              T_wk    = (20.5)*ones(size(x)) - 6.5e-3*zb;
        #          end
                melt = np.max(mu*T_wk)
                b = P-melt
        #        b(yr,:)=(3-(4/5000)*x); #mass balance in m/yr
            
            
        #     b(i) = P(i) - melt(i);
        #   if t(i) >= 250 
        #       idx = find(b>0);
        #       b(idx) = 4*b0(idx);
        #   end
            
        ###########################################
        # begin loop over space
        ###########################################
            for j in range(0, nxs):  # this is a kloudge -fix sometime
                   
               if j==1:
                    h_ave =(h[1] + h[2])/2
                    dhdx = (h[2] - h[1])/delx
                    dzdx = (dzbdx[1] + dzbdx[2])/2
                    Qp[1] = -K*(dhdx+dzdx)**3 * h_ave**5 # flux at plus half grid point
                    Qm[1] = 0 # flux at minus half grid point
                    dhdt[1] = b[1] - Qp[1]/(delx/2)
               elif h[j]==0 & h[j-1]>0:  # glacier toe condition
                    Qp[j] = 0
                    h_ave = h[j-1]/2
                    dhdx = -h[j-1]/delx			# correction inserted ght nov-24-04
                    dzdx = (dzbdx[j-1] + dzbdx[j])/2
                    Qm[j] = -K*(dhdx + dzdx)**3 * h_ave**5
                    dhdt[j] = b[j] + Qm[j]/delx
                    edge = j  #index of glacier toe - used for fancy plotting
               elif h[j]<=0 & h[j-1]<=0: # beyond glacier toe - no glacier flux
                    dhdt[j] = b[j]
                    Qp[j] = 0
                    Qm[j] = 0
               else:  # within the glacier
                   h_ave = (h[j+1] + h[j])/2
                   dhdx = (h[j+1] - h[j])/delx		# correction inserted ght nov-24-04
                   dzdx = (dzbdx[j] + dzbdx[j+1])/2
                   Qp[j] = -K*(dhdx+dzdx)**3 * h_ave**5
                   h_ave = (h[j-1] + h[j])/2
        #           dhdx = (h(i,j-1) - h(i,j))/delx
                   dhdx = (h[j] - h[j-1])/delx
                   dzdx = (dzbdx[j] + dzbdx[j-1])/2
                   Qm[j] = -K*(dhdx+dzdx)**3 * h_ave**5
                   dhdt[j] = b[j] - (Qp[j] - Qm[j])/delx
                   
                   dhdt[nxs] = 0 # enforce no change at boundary
            
            
            # end of loop over space
            h = np.max(h + dhdt*delt)
        
        ###########################################
        # plot glacier every so often
        # save h , and terminus position
        ###########################################
            if t/20 == np.floor(t/20):
                idx_out = idx_out+1
                
                edge_out[idx_out] = edge 
                t_out[idx_out] = t
                edge_out[idx_out] = delx*np.max(np.where(h>10))
                
                
    
    def run(self, years, dt, bt):
        self.dt = dt
        self.bt = bt
        self.years = years
        ts = np.array(range(0, self.years, self.dt))
        self.ts = ts
        slope_pct = self.dzdx
        
        self.h = np.zeros_like(ts, dtype='float')
        self.F = np.zeros_like(ts, dtype='float')
        self.L = np.zeros_like(ts, dtype='float')
        
        for t in ts:
            if t == ts[0]:
                #self.h[0] = self.H
                #self.L[0] = self.L0
                continue
                     
            self.h[t] = (1 - dt)*self.h[t-dt] + self.F[t-dt] + self.bt[t] 
            self.F[t] = self.rho**3 * self.g**3 * self.h[t]**3 * self.dzdx**3 * (self.f_d * self.h[t] + self.f_s/self.h[t])   # 1b
            self.L[t] = self.L[t-dt] + self.F[t-dt]*dt*(1-slope_pct)
            
        
        
