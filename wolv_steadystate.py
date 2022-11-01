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
import pickle

#%%

def temp_comb(yr, LIA_cooling=True, ANTH_warming=True):
    Tdot = 0.0
    if (yr >= 999) & LIA_cooling:
        Tdot = Tdot -0.25 * (yr - 1000) / 1000
    if (yr >= 1850) & ANTH_warming:
        Tdot = Tdot + 1.3 * (yr - 1850) / 150
    return Tdot
def temp_nat(yr):
    Tdot = 0.0
    if yr >= 999:
        Tdot = Tdot -0.25 * (yr - 1000) / 1000
    return Tdot
def temp_stable(yr):
    return 0

#%%

geom = pd.read_csv(r'C:\Users\drotto\Documents\USGS\glacier-diseq\matlab\Wolverine\wolv_geom.csv')
profile = pd.read_csv(r'C:\Users\drotto\Documents\USGS\glacier-diseq\features\Wolverine_zb_prof_NoDeep.csv')
x_gr = profile['x_gr']
zb_gr = profile['zb_gr']
w_geom = geom['widths_m']
x_geom = geom['length']

#%%

climate = pd.read_csv(r"C:\Users\drotto\Documents\USGS\glacier-diseq\features\ClimateRandom.csv")
Trand = climate['Trand']
Prand = climate['Prand']

#%%

# geom = pd.read_csv(r"C:\Users\drotto\Documents\USGS\glacier-diseq\features\glac_init_wolverine_preLIA.csv")
# x_init = geom['x_init']
# h_init = geom['h_init']

x_init = np.arange(0, 19000, 50)
with open('flowline2d_wolv_ss.pickle', 'rb') as f:
    last_run = pickle.load(f)
h_init = last_run['h'][-1, :]
# h_init = np.concatenate((np.tile(50, 100), np.tile(0, 280)), axis=0)


model = lgm.flowline2d(x_gr=x_gr, zb_gr=zb_gr, x_geom=x_geom, w_geom=w_geom, x_init=x_init, h_init=h_init, xmx=19000,
                       temp=temp_stable, sigT=1, sigP=0, P0=5,
                       delt=0.0125/8,
                       ts=0, tf=5000, T0=14.75,
                       Trand=Trand, Prand=Prand, gamma=6.5e-3,
                       rt_plot=True, dt_plot=100)

fig = model.plot(xlim0=0)
fig.show()
fig_output_name = 'flowline2d_wolv_ss_with_noise.png'
plt.savefig(fig_output_name)

file_output_name = 'flowline2d_wolv_ss_with_noise.pickle'
model.to_pickle(file_output_name)