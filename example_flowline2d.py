# -*- coding: utf-8 -*-
"""
example_flowline2d.py

Description.

Author: drotto
Created: 10/27/2022 @ 11:34 AM
Project: glacier-diseq
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import lgm


#%%

geom = pd.read_csv(r'C:\Users\drotto\Documents\USGS\glacier-diseq\matlab\Wolverine\wolv_geom.csv')
profile = pd.read_csv(r'C:\Users\drotto\Documents\USGS\glacier-diseq\features\Wolverine_zb_prof_NoDeep.csv')
x_gr = profile['x_gr']
zb_gr = profile['zb_gr']
w_geom = geom['widths_m']
x_geom = geom['length']

#%%
geom = pd.read_csv(r"C:\Users\drotto\Documents\USGS\glacier-diseq\features\glac_init_wolverine_preLIA.csv")
x_init = geom['x_init']
h_init = geom['h_init']

#%%

climate = pd.read_csv(r"C:\Users\drotto\Documents\USGS\glacier-diseq\features\ClimateRandom.csv")
Trand = climate['Trand']
Prand = climate['Prand']

#%%

def temp(yr, LIA_cooling=True, ANTH_warming=True):
    Tdot = 0.0
    if (yr >= 999) & LIA_cooling:
        Tdot = Tdot -0.25 * (yr - 1000) / 1000
    if (yr >= 1850) & ANTH_warming:
        Tdot = Tdot + 1.3 * (yr - 1850) / 150
    return Tdot

#%%

model = lgm.flowline2d(x_gr=x_gr, zb_gr=zb_gr, x_geom=x_geom, w_geom=w_geom, x_init=x_init, h_init=h_init, temp=temp,
                       ts=0, tf=2025,
                       Trand=Trand, Prand=Prand, rt_plot=True)

fig = model.plot()
fig.show()
fig_output_name = 'flowline2d_output_width_COMB.png'
plt.savefig(fig_output_name)

file_output_name = 'flowline2d_output_width_COMB.csv'
model.to_csv(file_output_name)