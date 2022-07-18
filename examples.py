# -*- coding: utf-8 -*-
"""
examples.py

Description.

Author: drotto
Created: 4/7/2022 @ 11:01 AM
Project: gm3s
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from src import gm3s


# %% Run the flowline model

params = dict(
    L0=8000,
    H=44,
    # tau=6.73,
    dzdx=0.4,
)

years = 25
dt = 1

model_flowline = gm3s.flowline(**params)
model_flowline.run(years=years, dt=dt, bt=np.full(years, fill_value=5.5))

t = model_flowline.ts
fig, ax = plt.subplots(3, 1)
ax[0].plot(t, model_flowline.h, label='height')
ax[1].plot(t, model_flowline.L, label='length')
ax[2].plot(t, model_flowline.F - model_flowline.bt, label='flux')
ax[0].legend()
ax[1].legend()
ax[2].legend()

# %% run the one stage model


params_1s = dict(
    # L=8000,
    H=44,
    tau=6.73,
    dzdx=0.4,
    P0=5.0,  # m/yr
    sigP=1.0,
    sigT=0.8,

    mu=0.65,  # m/(yr * C)
    gamma=6.5,  # C/km
    w=500,
    Atot=4.0,
    Aabl=2.0,
    ATgt0=3.4,
    # model params
    years=100,
    dt=1,
    mode='l'
)
model_1s = gm3s.gm1s(**params_1s)

t = model_1s.t
fig, ax = plt.subplots()
ax.plot(t, model_1s.L / model_1s.L.max(), label='length')
ax.plot(t, model_1s.Pp, label='precip')
ax.plot(t, model_1s.Tp, label='temp')
ax.legend()

# %% Fig. 7

params = dict(
    L=8000,
    H=44,
    tau=6.73,
    dzdx=0.4,
)

years = 100
dt = 1

model = gm3s.gm3s(**params)
model.discrete(years=years, dt=dt, bt=np.full(years, fill_value=5))

t = model.ts / model.tau
fig, ax = plt.subplots()
ax.plot(t, model.L / model.L.max(), label='length')
ax.plot(t, model.F / model.F.max(), label='flux')
ax.plot(t, model.h / model.h.max(), label='height')
ax.legend()

# %%

freq = np.arange(0, 1 / np.e, 2 * np.pi / 10000)
spectrum = model.power_spectrum(freq=freq, sig_L_1s=model_1s.sig_L)
fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(freq, spectrum, label='power spectrum')