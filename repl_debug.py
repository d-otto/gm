# -*- coding: utf-8 -*-
"""
repl_debug.py

Description.

Author: drotto
Created: 4/4/2023 @ 3:59 PM
Project: glacier-attribution
"""

import numpy as np

from gm import gm

#%%

tf = 10
ts = 0
delx = 50
x = np.arange(0, 10000, delx)
x_gr = x.copy()
x_geom = x.copy()
xmx = x.max()
x_init = x.copy()
h_init = np.concatenate([4*np.sqrt(np.arange(0, len(x)//2, 1))[::-1], np.zeros(len(x)//2 )])
zb_gr = np.linspace(3000, 1000, x.size)
w_geom = np.linspace(5000, 500, x.size)

sigT = 1
sigP = 1
P0 = 2
T0 = 15

T = np.zeros(tf)
P = np.zeros(tf)
t_stab = None
temp = None
gamma = 6.5e-3
mu = 0.65
g = 9.81
rho = 916.8
fd = 1.9e-24
fs = 5.7e-20
delt = 0.0125 / 8
deltout = 1/delt
dt_plot = 100
rt_plot = False
xlim0 = None
min_thick = 1
alt_mb_feedback = True

class self:
    def __init__(self):
        pass


#%%

i=0
i=100


#%%

len(self.h)
np.nansum(self.b, axis=1)

bal = self.b * self.w * self.delx
idx = np.c_[np.zeros(len(self.edge_idx)), self.edge_idx].astype(int)
x = np.diag(np.add.reduceat(bal/self.area.reshape(-1, 1), idx.ravel(), axis=1)[:, ::2])
y = self.gwb/self.area

print(x)
print(y)
print(x-y)


#%% Run the model

model = gm.flowline2d(x_gr=x_gr, zb_gr=zb_gr, x_geom=x_geom, w_geom=w_geom,
                      x_init=x_init, h_init=h_init, xmx=xmx,
                      sigT=0, sigP=0,
                      delt=0.0125/8, delx=50,
                      ts=0, tf=500,
                      P0=1.5, T0=38, gamma=14e-3, mu=0.65)


#%%

bal = model.b * model.w * model.delx
idx = np.c_[np.zeros(len(model.edge_idx)), model.edge_idx].astype(int)
x = np.diag(np.add.reduceat(bal/model.area.reshape(-1, 1), idx.ravel(), axis=1)[:, ::2])
y = model.gwb/model.area

print(np.nansum(model.b, axis=1))
print(np.nansum(bal))
print(x)
print(y)

# all 3 methods of summing give the same answer
print(x-y)
print(x-np.nansum(bal))

cumsum = np.cumsum(bal, axis=1)

flux = model.F
Fdiv = bal - flux  # = dh/dt
Fdiv_cumsum = np.cumsum(Fdiv, axis=1)
