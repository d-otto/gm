# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from scipy.stats import norm


class gm3s:
    
    def __init__(self, mode='b'):
    
        # Parameters
        tf = 2000          # length of integration [yrs]
        ts = 0             # starting year
        dt = 1             # time step [keep at 1yr always]
        nts = int((tf-ts)/dt) # number of time steps
        t = np.arange(ts, tf, dt)       # array of times [yr]
        
        ## glacier model parameters
        mu = 0.65      # melt factor [m yr**-1 K**-1]
        Atot = 3.95e6  # total area of the glacier [m**2]
        ATgt0 = 3.4e6  # area of the glacier where some melting occurs [m**2]
        Aabl = 1.95e6  # ablation area [m**2] 
        w = 500        # characteristic width of the glacier tongue [m]. 
        dt = 1         # incremental time step [yr]
        hbar = 44.4186 # characteristic ice thickness near the terminus [m]
        gamma = 6.5e-3 # assumed surface lapse rate [K m**-1] 
        tanphi = 0.4   # assumed basal slope [no units]
        
        # natural climate variability - for temperature and precipitation forcing
        sigP = 1.0     # std. dev. of accumulation variability [m yr**-1]
        sigT = 0.8     # std. dev. of melt-season temperature variability [m yr**-1]
        # natural climate variability - for mass balance forcing
        sigb = 1.5     # std. dev. of annual-mean mass balance [m yr**-1]
        
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
