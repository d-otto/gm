# -*- coding: utf-8 -*-
"""
viz.py

Description.

Author: drotto
Created: 4/13/2023 @ 3:00 PM
Project: glacier-attribution
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_dbdz(model):
    dbdz = np.gradient(model.b[-1, :]) / np.gradient((model.zb + model.h[-1, :]))
    dzdb = 1 / dbdz
    obs_dbdz = mb.groupby('yr1').apply(lambda mb: (mb['b'] / 1000).diff(-1) / mb['z0'].diff(-1))
    obs_dbdz = pd.DataFrame(obs_dbdz.reset_index(level=1, drop=True).reset_index())
    obs_dbdz.columns = ['yr', 'dbdz']
    obs_dbdz['z'] = mb['z0']
    ela_avg = ela.mean(axis=0)

    fig, ax = plt.subplots(2, 1, figsize=(8, 12), dpi=200, layout='constrained')
    for year, group in mb.groupby('yr1'):
        ax[0].plot(group.b / 1000, group.z0, color='black', alpha=0.25, lw=0.5)
    mb_avg = mb.groupby('z0').mean().reset_index()
    ax[0].plot(mb_avg.b / 1000, mb_avg.z0, color='black', lw=3)
    ax[0].axhline(ela_avg.ela, c='grey', label=f'Obs. balance profile (mean ELA = {ela_avg.ela:.0f}m)')
    ax[0].axvline(0, c='black')

    slope = np.arange(0.001, 0.015, 0.001)
    x = np.arange(-5, 2, 0.1)
    x0 = model.b[-1, model.edge_idx[-1] - 1]
    y0 = model.zb[model.edge_idx[-1] - 1] + model.h[-1, model.edge_idx[-1] - 1]
    X, M = np.meshgrid(x, slope)
    Y = (X - x0) * 1 / M + y0
    contour = ax[0].contour(X, Y, M, colors='grey', linestyles='dashed', levels=len(slope) // 2)
    ax[0].clabel(contour, fontsize='small')

    ax[0].plot(model.b[-1, :], model.zb + model.h[-1, :], color='red', lw=3)

    # ax.fill_between(model.b[0, :], model.zb + model.h[0, :] + model.w/2, model.zb + model.h[0, :] - model.w/2, color='grey', alpha=0.25)
    ax[0].axhline(model.ela[-1], c='red', label=f'Model balance profile (ELA = {model.ela[-1]:.0f}m)')

    # ax[0].axline((model.b[-1, model.edge_idx[0]], model.zb[model.edge_idx[0]] + model.h[-1, model.edge_idx[0]]), slope=slope, color='grey', ls='--')
    ax[0].set_ylim(model.zb[model.edge_idx[-1]], model.zb[0] + model.h[-1, 0])

    for year, group in obs_dbdz.groupby('yr'):
        ax[1].plot(group.z, group.dbdz, color='black', alpha=0.25, lw=0.5)
    obs_dbdz_mean = obs_dbdz.groupby('z').mean().reset_index()
    ax[1].plot(obs_dbdz_mean.z, obs_dbdz_mean.dbdz, color='black', lw=3, label='Obs. dbdz')
    ax[1].plot(model.zb + model.h[-1, :], dbdz, color='red', lw=3, label='Model dbdz')

    for axis in ax.ravel():
        axis.grid(which='both', axis='both', ls=':')
        axis.legend()
    return fig, ax