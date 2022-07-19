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
model.discrete(years=years, dt=dt, bt=np.full(years, fill_value=0.5))

t = model.ts / model.tau
fig, ax = plt.subplots()
ax.plot(t, model.L / model.L.max(), label='length')
ax.plot(t, model.F / model.F.max(), label='flux')
ax.plot(t, model.h / model.h.max(), label='height')
ax.legend()




# %% Fig. 8c

freq = np.arange(0, 1 / np.e, 2 * np.pi / 10000)
spectrum = model.power_spectrum(freq=freq, sig_L_1s=1)
fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(freq, spectrum, label='power spectrum')


#%% Fig 8d.

freq = np.linspace(1/1e4, 1, int(1e4))
phase = model.phase(freq=freq)
fig, ax = plt.subplots()
ax.set_xscale('log')
ax.plot(freq, phase, label='phase')


#%% Fig 8b.

t = np.arange(0, 40, 1)
acf = model.acf(t=t)
fig, ax = plt.subplots()
ax.plot(t, acf, label='ACF')


#%% Fig 8a.

params = dict(
    L=8000,
    H=44,
    tau=6.73,
    dzdx=0.4,
)

years = 100
dt = 1

model = gm3s.gm3s(**params)
model.discrete(years=years, dt=dt, bt=np.full(years, fill_value=-0.5))
L_retreat = model.L + model.L_bar

model = gm3s.gm3s(**params)
model.discrete(years=years, dt=dt, bt=np.full(years, fill_value=0.5))
L_advance = model.L + model.L_bar

fig, ax = plt.subplots()
ax.plot(model.ts, L_advance, label='Length')
ax.plot(model.ts, L_retreat, label='Length')
ax.set_xlim(-5, 30)

#%%

model.continuous(3, bt=0.5)

model.continuous(t=np.arange(0, 100, 1), bt=-0.5)


#%% Linear trend in mass balance
linear_params = dict(
    L=8000,
    H=44,
    dbdt=-0.005,
    ts=np.arange(0, 100, 1)
)
linear_model = gm3s.gm1s(**linear_params)


# Step change in mass balance
step_params = dict(
    L=8000,
    H=44,
    bt=-0.5,
    ts=np.arange(0, 100, 1)
)
step_model = gm3s.gm1s(**step_params)


# custom mass balance series
discrete_params = dict(
    L=8000,
    H=44,
    bt=np.arange(0, -0.5, -0.005),
    ts=np.arange(0, 100, 1)
)
discrete_model = gm3s.gm1s(**discrete_params)

print(linear_model.L)
print(step_model.L)
print(discrete_model.L)


#%% Looking more at the linear model
# custom mass balance series
b_p = [
        np.tile(-0.01, 100),
        #np.sin(np.linspace(np.pi/2, 0, 50)) * -0.01,
        np.tile(0, 200)
    ]
b_p = np.concatenate(b_p)
linear_params = dict(
    L=8000,
    H=100,
    bt=-3,
    b_p=b_p,
    ts=np.arange(0, 300, 1)
)
m = gm3s.gm1s(**linear_params)

fig, ax = plt.subplots(2,1, figsize=(8,10), dpi=150)
ax[0].plot(m.ts, m.L_bar + m.L_p, label='L_bar + L_p')
ax[0].plot(m.ts, m.L_p, label='L_p')
ax[0].plot(m.ts, m.L_eq, label='L_eq')
#ax[0].plot(m.ts, np.cumsum(m.L_p), label='cumsum(L_p)')
ax[1].plot(m.ts/m.tau, m.L_p/m.delta_L, label='L_p/delta_L')
ax[0].set_xlabel('t')
ax[1].set_xlabel('t/tau')
leg = np.array([ax[0].legend(), ax[1].legend()])
fig.show()
 
#%%

fig, ax = plt.subplots(2,1, figsize=(8,10), dpi=150)
ax[0].plot(m.ts/m.tau, (m.L_p)/m.delta_L, label='L_bar + L_p')
ax[0].plot(m.ts/m.tau, m.L_p/m.delta_L, label='L_p')
#ax[0].plot(m.ts/m.tau, m.L_eq/m.delta_L, label='L_eq')
ax[1].plot(m.ts/m.tau, m.L_p/m.delta_L, label='L_p/delta_L')
ax[1].set_xlabel('t/tau')
leg = np.array([ax[0].legend(), ax[1].legend()])
fig.show()

# fig, ax = plt.subplots(1,1)
# ax.plot(m.ts, m.L_eq/m.L_p, label='L_p / L_eq')
# #ax.plot(m.ts, m.L/m.L_eq, label='L_p / L_eq')
# leg = ax.legend()
# fig.show()
