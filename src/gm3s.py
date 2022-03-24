# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from scipy.stats import norm


class gm3s:
    
    def __init__(self, mode='b', tf=None, ts=0, dt=1, mu=None, Atot=None, ATgt0=None, Aabl=None, w=None, hbar=None, gamma=None, tanphi=None, sigP=None, sigT=None, sigb=None):
        """3-stage glacier model (Roe and Baker, 2014)

        Parameters
        ----------
        mode : str, optional
            Calculate anomalies based on mass balance 'b' or length 'l'. Default is 'b'.
        tf : int
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
        hbar : numeric
            Characteristic ice thickness near the terminus [m]
        gamma : numeric
            Assumed surface lapse rate [K m**-1] 
        tanphi : numeric
            Assumed basal slope [no units]
        sigP : numeric
            Std. dev. of accumulation variability [m yr**-1]
        sigT : numeric
            Std. dev. of melt-season temperature variability [m yr**-1]
        sigb : numeric
            Std. dev. of annual-mean mass balance [m yr**-1]

        Returns
        -------
        gm3s : object
            
            
        """
        
        nts = int((tf-ts)/dt)  # number of time steps
        t = np.arange(ts, tf, dt)  # array of times [yr]

        
        ## linear model coefficients, combined from above parameters
        ## play with their values by choosing different numbers...
        alpha = mu*ATgt0*dt/(w*hbar)
        beta = Atot*dt/(w*hbar)
        
        # glacier memory [ys]
        # this is the glacier response time  (i.e., memory) based on the above glacier geometry
        # if you like, just pick a different time scale to see what happens. 
        # Or also, use the simple, tau = hbar/b_term, if you know the terminus
        # balance rate from, e.g., observations
        tau = w*hbar/(mu*gamma*tanphi*Aabl)
        
        # coefficient needed in the model integrations
        # keep fixed - they are intrinsic to 3-stage model
        eps = 1/np.sqrt(3)
        phi = 1-dt/(eps*tau)
        
        # note at this point you could create your own climate time series, using
        # random forcing, trends, oscillations etc.
        # Tp = array of melt-season temperature anomalise
        # Pp = array of accumlation anomalies
        # bp = array of mass balance anomalies
        Tp = norm.rvs(scale=sigT, size=nts)
        Pp = norm.rvs(scale=sigP, size=nts)
        bp = norm.rvs(scale=sigb, size=nts)
        
        L3s = np.zeros(nts) # create array of length anomalies
        
        ## integrate the 3 stage model equations forward in time
        for i in range(5, nts):
            if mode == 'l': 
                L3s[i] = 3*phi*L3s[i-1]-3*phi**2*L3s[i-2]+1*phi**3*L3s[i-3] \
                         + dt**3*tau/(eps*tau)**3 * (beta*Pp(i-3) - alpha*Tp(i-3))
            
            if mode == 'b':
        # if you want to use mass balance anomalies instead comment out the 2 lines
        # above, and uncomment the 2 lines below
                L3s[i] = 3*phi*L3s[i-1]-3*phi**2*L3s[i-2]+1*phi**3*L3s[i-3] \
                         + dt**3*tau/(eps*tau)**3 * (beta*bp[i-3])
        
        
        # assign all local variables as attributes
        vars = locals().copy()
        for var in vars:
            if var not in self.__dict__:
                setattr(self, f'{var}', eval(var))        
                
    
    
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
        
        df = pd.DataFrame(data=dict(bp = self.bp,
                                    Pp = self.Pp,
                                    Tp = self.Tp,),
                          index=pd.Index(self.t, name='t'))
    
        return df
    
    def to_tidy(self):
        import pandas as pd
        
        df = self.to_pandas().reset_index()
        df = pd.melt(df, id_vars=['t'])
        
        return df
