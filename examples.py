# -*- coding: utf-8 -*-
"""
examples.py

Description.

Author: drotto
Created: 4/7/2022 @ 11:01 AM
Project: lgm
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from src import lgm


# %% Run the flowline model
# 
# params = dict(
#     L0=8000,
#     H=44,
#     # tau=6.73,
#     dzdx=0.4,
# )
# 
# years = 25
# dt = 1
# 
# model_flowline = lgm.flowline(**params)
# model_flowline.run(years=years, dt=dt, bt=np.full(years, fill_value=5.5))
# 
# t = model_flowline.ts
# fig, ax = plt.subplots(3, 1)
# ax[0].plot(t, model_flowline.h, label='height')
# ax[1].plot(t, model_flowline.L, label='length')
# ax[2].plot(t, model_flowline.F - model_flowline.bt, label='flux')
# ax[0].legend()
# ax[1].legend()
# ax[2].legend()
# 
# # %% run the one stage model
# 
# 
# params_1s = dict(
#     # L=8000,
#     H=44,
#     tau=6.73,
#     dzdx=0.4,
#     P0=5.0,  # m/yr
#     sigP=1.0,
#     sigT=0.8,
# 
#     mu=0.65,  # m/(yr * C)
#     gamma=6.5,  # C/km
#     w=500,
#     Atot=4.0,
#     Aabl=2.0,
#     ATgt0=3.4,
#     # model params
#     years=100,
#     dt=1,
#     mode='l'
# )
# model_1s = lgm.gm1s(**params_1s)
# 
# t = model_1s.t
# fig, ax = plt.subplots()
# ax.plot(t, model_1s.L / model_1s.L.max(), label='length')
# ax.plot(t, model_1s.Pp, label='precip')
# ax.plot(t, model_1s.Tp, label='temp')
# ax.legend()

# %% Fig. 7

params_3s = dict(
    Atot=8000,
    W=1,
    L=8000,
    H=44,
    tau=6.73,
    dzdx=0.4,
    ts=np.arange(0, 50, 1),
    bt=0,
    b_p=np.concatenate([
        np.tile(0, 1), np.tile(0.5, 49)
    ]),
)

years = 100
dt = 1

m3s = lgm.gm3s(**params_3s)
m3s.linear()

t = m3s.ts / m3s.tau
fig, ax = plt.subplots(dpi=200)
ax.plot(t, m3s.L / m3s.L.max(), label='length')
ax.plot(t, m3s.F / m3s.F.max(), label='flux')
ax.plot(t, m3s.h / m3s.h.max(), label='height')
ax.legend()

# %% Fig. 8c
# freq = np.arange(0, 1 / np.e, 2 * np.pi / 10000)
# spectrum = m3s.power_spectrum(freq=freq, sig_L_1s=1)
# fig, ax = plt.subplots()
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.plot(freq, spectrum, label='power spectrum')

#%% Fig 8d.
# freq = np.linspace(1/1e4, 1, int(1e4))
# phase = m3s.phase(freq=freq)
# fig, ax = plt.subplots()
# ax.set_xscale('log')
# ax.plot(freq, phase, label='phase')

#%% Fig 8b.
t = np.arange(0, 40, 1)
acf = m3s.acf(t=t)
fig, ax = plt.subplots()
ax.plot(t, acf, label='ACF')


#%% Fig 8a.

params_3s = dict(
    Atot=8000,
    W=1,
    L=8000,
    H=44,
    tau=6.73,
    dzdx=0.4,
    bt=0,
    ts=np.arange(0, 50, 1)
)

years = 100
dt = 1

m3s = lgm.gm3s(**params_3s)
m3s.linear(bt=np.tile(-0.5, 50))
L_retreat = m3s.L + m3s.L_bar

m3s = lgm.gm3s(**params_3s)
m3s.linear(bt=np.tile(0.5, 50))
L_advance = m3s.L + m3s.L_bar

fig, ax = plt.subplots()
ax.plot(m3s.ts, L_advance, label='Length')
ax.plot(m3s.ts, L_retreat, label='Length')
ax.set_xlim(-5, 30)


#%% Looking more at the linear model
# custom mass balance series
params_1s = dict(
    L=8000,
    H=100,
    bt=-3,
    ts=np.arange(0, 300, 1)
)
b_p = [
    -1/50 * np.arange(0, 50, 1),
    np.tile(-1, 250)
]
params_1s['b_p'] = np.concatenate(b_p)
m1s = lgm.gm1s(**params_1s)


fig, ax = plt.subplots(3,1, figsize=(8,10), dpi=150)
ax[0].plot(m1s.ts, m1s.L_eq, label='L_eq')
ax[0].plot(m1s.ts, m1s.L, label='L')
ax[1].plot(m1s.ts / m1s.tau, m1s.L_p/m1s.L_eq, label='L_p/delta_L')
ax[2].plot(m1s.ts, np.concatenate(b_p), label='mass balance')
#ax[1].plot(m1s.ts / m1s.tau, np.exp(-m1s.ts / m1s.tau) - 1 , label='L_p')
ax[0].set_xlabel('t')
ax[1].set_xlabel('t/tau')
leg = np.array([ax[0].legend(), ax[1].legend(), ax[2].legend()])
fig.show()


 
#%% Comparing 3s and 1s

# todo : look at the 3s model changing before the start of the climate trend

# trend
# b_p = np.concatenate([
#     -1/50 * np.arange(0, 50, 1),
#     np.tile(-1, 250)
# ])

# step
b_p = np.concatenate([
    np.tile(0, 50),
    np.tile(-1, 250)
])

params_1s = dict(
    L=8000,
    H=100,
    bt=-1,
    ts=np.arange(0, 300, 1),
    b_p=b_p,
)
params_3s = dict(
    Atot=8000,
    W=1,
    L=8000,
    H=100,
    bt=-1,
    ts=np.arange(0, 300, 1)
)


m1s = lgm.gm1s(**params_1s)
m3s = lgm.gm3s(**params_3s)
m3s.discrete(bt=b_p)

#print(m3s.L_p)
#print(m3s.bt)
#print(m3s.F)
#print(m3s.L)

fig, ax = plt.subplots(3,1, figsize=(8,10), dpi=150)
ax[0].plot(m1s.ts, m1s.L, label='1s')
ax[0].plot(m3s.ts, m3s.L, label='3s')

ax[1].plot(m1s.ts, m1s.bt_p, label='1s bt_p',)
ax[1].plot(m3s.ts, m3s.bt_p, label='3s bt_p',)

ax[2].plot(m1s.ts, m1s.L_p, label='1s L_p', color='teal')
ax[2].plot(m3s.ts, m3s.L_p, label='3s L_p', color='orange')
ax[2].plot(m1s.ts, m1s.dL, label='1s L_eq', color='blue')
ax[2].plot(m3s.ts, m3s.dL, label='3s L_eq', color='red')

ax[1].set_xlabel('t')
leg = [axis.legend() for axis in ax]
fig.show()