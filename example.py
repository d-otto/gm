# -*- coding: utf-8 -*-

import numpy as np
import lgm


# Example values for Mt. Baker, WA.
params = dict(
    ## glacier model parameters
    mu = 0.8,  # melt factor [m yr**-1 K**-1]
    
    
     
    W = 500,  # characteristic width of the glacier tongue [m]. 
    h0 = 44.4186,  # characteristic ice thickness near the terminus [m]
    gamma = 6.5e-3,  # assumed surface lapse rate [K m**-1] 
    
    # natural climate variability - for temperature and precipitation forcing
    Tbar=20,
    sigT=0.9,  # std. dev. of melt-season temperature variability [m yr**-1
    Pbar=2.0,
    sigP=1.0, # std. dev. of accumulation variability [m yr**-1]
    dzdx=0.4,
    L0=16000,
)

model = lgm.flowline(**params)


###

# # Example values for Mt. Baker, WA.
# params = dict(
#     ## glacier model parameters
#     mu = 0.65,  # melt factor [m yr**-1 K**-1]
#     Atot = 3.95e6,  # total area of the glacier [m**2]
#     ATgt0 = 3.4e6,  # area of the glacier where some melting occurs [m**2]
#     Aabl = 1.95e6,  # ablation area [m**2] 
#     W = 500,  # characteristic width of the glacier tongue [m]. 
#     h0 = 44.4186,  # characteristic ice thickness near the terminus [m]
#     gamma = 6.5e-3,  # assumed surface lapse rate [K m**-1] 
#     tanphi = 0.4,  # assumed basal slope [no units]
#     # natural climate variability - for temperature and precipitation forcing
#     sigP = 1.0,  # std. dev. of accumulation variability [m yr**-1]
#     sigT = 0.8,  # std. dev. of melt-season temperature variability [m yr**-1]
#     # natural climate variability - for mass balance forcing
#     sigb = 1.5,  # std. dev. of annual-mean mass balance [m yr**-1]
# )