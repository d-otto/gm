# -*- coding: utf-8 -*-

from gm3s import gm3s
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

if __name__ == '__main__':
    
    # Example values for Mt. Baker, WA.
    params = dict(
        tf=2000,  # length of integration [yrs]
        ts = 0, # starting year
        dt = 1,  # time step [keep at 1yr always]
        ## glacier model parameters
        mu = 0.65,  # melt factor [m yr**-1 K**-1]
        Atot = 3.95e6,  # total area of the glacier [m**2]
        ATgt0 = 3.4e6,  # area of the glacier where some melting occurs [m**2]
        Aabl = 1.95e6,  # ablation area [m**2] 
        w = 500,  # characteristic width of the glacier tongue [m]. 
        hbar = 44.4186,  # characteristic ice thickness near the terminus [m]
        gamma = 6.5e-3,  # assumed surface lapse rate [K m**-1] 
        tanphi = 0.4,  # assumed basal slope [no units]
        # natural climate variability - for temperature and precipitation forcing
        sigP = 1.0,  # std. dev. of accumulation variability [m yr**-1]
        sigT = 0.8,  # std. dev. of melt-season temperature variability [m yr**-1]
        # natural climate variability - for mass balance forcing
        sigb = 1.5,  # std. dev. of annual-mean mass balance [m yr**-1]
    )
    
    model = gm3s(mode='b', **params)
    df_p = model.to_tidy()
    fig = px.line(df_p, x='t', y='value', facet_row='variable')
    fig.show()
    
    
    