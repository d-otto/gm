# -*- coding: utf-8 -*-
"""
flowline2d.py

Description.

Author: drotto
Created: 10/25/2022 @ 9:56 AM
Project: glacier-diseq
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from numba import jit, njit
import scipy.ndimage

# # How well do the equilibrium responses match?
# # How well does the timescale match?
# # What is the distribution of trends?
# # How does the run length work? Does it follow a Poisson process?
# # Is the dynamical model consistent with a Gaussian pdf?

plt.ioff()
mpl.use('qt5agg')

fig_output_name = 'flowline2d_output_width_COMB.png'
file_output_name = 'flowline2d_output_width_COMB.csv'
xlim0 = 1500  # left x-limit for figure (years)
dt_plot = 100  # interval to plot during execution (years)

# #-----------------
# #define parameters
# #-----------------
rho = 910  # kg/m^3
g = 9.81  # m/s^2
n = 3

mu = 0.65  # melt rate in m /yr /degC
gamma = 6.5e-3  # lapse rate

# climate forcing
sigT = 0.8  # degC
sigP = 1.0  # m/yr
T0 = 14.70  # baseline temp degC
LIA_cooling = True  # flag for cooling after year 1000
ANTH_warming = False

fd = 1.9e-24  # Deformation parameter Pa^-3 s^-1
fs = 5.7e-20  # Sliding parameter Pa^-3 s^-1 m^2
# fs = 0

fd = fd * np.pi * 1e7
fs = fs * np.pi * 1e7  # convert from seconds to years

xmx = 19000  # the domain size in m
# delx = 100  #grid spacing in m
delx = 50  # grid spacing in m
nxs = round(xmx / delx)  # number of grid points

# delt = 0.0125 # time step in yrs
delt = 0.0125 / 8  # time step in yrs suitable for 200m
ts = 0000  # starting time
#tf = 2030  # final time
tf = 2030
nts = round(np.floor((tf - ts) / delt))  # number of time steps ('round' used because need nts as integer)
nyrs = tf - ts

x = np.arange(0, xmx, delx)  # x array

# -----------------
# define functions
# -----------------
b = (8 - (4 / 5000) * x)  # mass balance in m/yr

# ---------------------------------
# different glacier bed geometries
# ---------------------------------
# load Wolverine_zb_prof_NoDeep.mat
geom = pd.read_csv(r'C:\Users\drotto\Documents\USGS\glacier-diseq\matlab\Wolverine\wolv_geom.csv')
profile = pd.read_csv(r'C:\Users\drotto\Documents\USGS\glacier-diseq\features\Wolverine_zb_prof_NoDeep.csv')
x_gr = profile['x_gr']
zb_gr = profile['zb_gr']
zb = interp1d(x_gr, zb_gr)
zb = zb(x)

# w = interp1(x_fran,width_fran,x)
w = interp1d(geom['length'], geom['widths_m'])
w = w(x)
w = np.clip(w, 600, None)
#w = 500 * np.ones(x.size)  # no width data in this version
dzbdx = np.gradient(zb, x)  # slope of glacer bed geometries.
dwdx = np.gradient(w, x)

##

# define width of glacier
# w0 = 500

# w = w0*ones(size(x))

# ----------------------------------------------------------------------------------------------
# alternative starting geometry for a glacierusing results from a previus integration
# stored in glac_init.mat
# probably only worth it for mass balance parameterizations similar to the default one we tried
# ----------------------------------------------------------------------------------------------

# do this for a constant pre-industrial climate
# load glac_init_wolverine.mat

# do this for a LIA cooling
# load glac_init_wolverine_preLIA.mat
geom = pd.read_csv(r"C:\Users\drotto\Documents\USGS\glacier-diseq\features\glac_init_wolverine_preLIA.csv")
x_init = geom['x_init']
h_init = geom['h_init']
h0 = interp1d(x_init, h_init, 'linear')
h0 = h0(x)

h = np.zeros(x.size)  # zero out height array
h[:] = h0  # initialize height array

# pick climate forcing
climate = pd.read_csv(r"C:\Users\drotto\Documents\USGS\glacier-diseq\features\ClimateRandom.csv")
Trand = climate['Trand']
Prand = climate['Prand']

# initialize climate forcing
Tp = sigT * Trand[0:nyrs + 1]
Pp = sigP * Prand[0:nyrs + 1]  # initialize climate forcing
Tp[0:49] = 0


@njit(fastmath={'contract', 'arcp', 'nsz', 'ninf', 'nnan'})
def space_loop(h, b):
    Qp = np.zeros(x.size)  # Qp equals j+1/2 flux
    Qm = np.zeros(x.size)  # Qm equals j-1/2 flux
    dhdt = np.zeros(x.size)  # zero out thickness rate of change array

    # -----------------------------------------
    # begin loop over space
    # -----------------------------------------
    for j in range(0, nxs - 1):  # this is a kloudge -fix sometime
        if j == 0:
            h_ave = (h[0] + h[1]) / 2
            dhdx = (h[1] - h[0]) / delx
            dzdx = (dzbdx[0] + dzbdx[1]) / 2
            Qp[0] = -(dhdx + dzdx)**3 * h_ave**4 * (rho * g)**3 * (
                        fd * h_ave + fs / h_ave)  # flux at plus half grid point
            Qm[0] = 0  # flux at minus half grid point
            dhdt[0] = b[0] - Qp[0] / (delx / 2) - (Qp[0] + Qm[0]) / (2 * w[0]) * dwdx[0]
        elif (h[j] == 0) & (h[j - 1] > 0):  # glacier toe condition
            Qp[j] = 0
            h_ave = h[j - 1] / 2
            dhdx = -h[j - 1] / delx  # correction inserted ght nov-24-04
            dzdx = (dzbdx[j - 1] + dzbdx[j]) / 2
            Qm[j] = -(rho * g)**3 * h_ave**4 * (dhdx + dzdx)**3 * (fd * h_ave + fs / h_ave)
            dhdt[j] = b[j] + Qm[j] / delx - (Qp[j] + Qm[j]) / (2 * w[j]) * dwdx[j]
            edge = j  # index of glacier toe - used for fancy plotting
        elif (h[j] <= 0) & (h[j - 1] <= 0):  # beyond glacier toe - no glacier flux
            dhdt[j] = b[j]
            Qp[j] = 0
            Qm[j] = 0
        else:  # within the glacier
            h_ave = (h[j + 1] + h[j]) / 2
            dhdx = (h[j + 1] - h[j]) / delx  # correction inserted ght nov-24-04
            dzdx = (dzbdx[j] + dzbdx[j + 1]) / 2
            Qp[j] = -(rho * g)**3 * h_ave**4 * (dhdx + dzdx)**3 * (fd * h_ave + fs / h_ave)
            h_ave = (h[j - 1] + h[j]) / 2
            dhdx = (h[j] - h[j - 1]) / delx
            dzdx = (dzbdx[j] + dzbdx[j - 1]) / 2
            Qm[j] = -(rho * g)**3 * h_ave**4 * (dhdx + dzdx)**3 * (fd * h_ave + fs / h_ave)
            dhdt[j] = b[j] - (Qp[j] - Qm[j]) / delx - (Qp[j] + Qm[j]) / (2 * w[j]) * dwdx[j]
        dhdt[nxs - 1] = 0  # enforce no change at boundary
    # ----------------------------------------
    # end loop over space
    # ----------------------------------------
    h = np.clip(h + dhdt * delt, 0, 10000)
    return h, edge




# -----------------------------------------
# begin loop over time
# -----------------------------------------
yr = 0
idx_out = 0
deltout = 1
nouts = round(nts * delt)
edge_out = np.full(nouts, fill_value=np.nan, dtype='float')
t_out = np.full(nouts, fill_value=np.nan, dtype='float')
T_out = np.full(nouts, fill_value=np.nan, dtype='float')
bal_out = np.full(nouts, fill_value=np.nan, dtype='float')
ela_out = np.full(nouts, fill_value=np.nan, dtype='float')

fig = plt.figure(figsize=(12, 10), dpi=100)
gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=(2, 1, 1))
ax = np.empty((3, 2), dtype='object')
plt.show(block=False)

ax[0, 0] = fig.add_subplot(gs[0, 0])
ax[0,0].set_xlabel('Time (years)')
ax[0,0].set_ylabel('Elevation (m)')

ax[0, 1] = fig.add_subplot(gs[0, 1])
ax[0, 1].set_xlabel('Distance (km)')
ax[0, 1].set_ylabel('Elevation (m)')

ax[1, 0] = fig.add_subplot(gs[1, 0])
ax[1, 0].set_ylabel('T ($^o$C)')

ax[1, 1] = fig.add_subplot(gs[1, 1])
ax[1, 1].set_ylabel('L (km)')

ax[2, 0] = fig.add_subplot(gs[2, 0])
ax[2, 0].set_ylabel('Bal (m $yr^{-1}$)')
ax[2, 0].set_xlabel('Time (years)')

ax[2, 1] = fig.add_subplot(gs[2, 1])
ax[2, 1].set_xlabel('Time (years)')
ax[2, 1].set_ylabel('Cum. bal. (m)')

for axis in ax.ravel():
    axis.grid(axis='both', alpha=0.5)
    axis.set_axisbelow(True)
plt.tight_layout()

for i in range(0, nts):
    t = delt * i  # time in years 
    # define climate every year
    if (t == np.floor(t)):
        yr = yr + 1
        print(f'yr = {yr}')
        P = np.ones(x.size) * (5.0 + 0.0 * Pp[yr])
        #         T_wk    = (25.25+0*Tp(yr))*ones(size(x)) - gamma*(zb+h)

        # choose this for constant pre-industrial climate
        #         T_wk    = (13.20 + 0.0*Tp(yr))*ones(size(x)) - gamma*(zb+h)
        # choose this to allow for a LIA cooling
        # millennial trend
        T_wk = (T0 + 1.0 * Tp[yr]) * np.ones(x.size) - gamma * (zb + h)
        if (yr >= 999) & LIA_cooling:
            T_wk = T_wk - 0.25 * (yr - 1000) / 1000
        if (yr >= 1850) & ANTH_warming:
            T_wk = T_wk + 1.3*(yr-1850)/150

        #           end
        melt = np.core.umath.clip(mu * T_wk, 0, 100000)
        b = P - melt

    
    
    
    h, edge = space_loop(h, b)

    # ----------------------------
    # plot glacier every so often
    # save h , and edge
    # ----------------------------
    
    if t / deltout == np.floor(t / deltout):
        
        T_out[idx_out] = T_wk[1] - 3.2
        t_out[idx_out] = t
        edge_out[idx_out] = edge * delx
        #edge_out[idx_out] = delx * max(h[h > 10])

        bal = b * w * delx  # mass added in a given cell units are m^3 yr^-1
        bal_out[idx_out] = np.trapz(bal[:edge])  # should add up all the mass up to the edge, and be zero in equilibrium (nearly zero)
        ela_idx = np.abs(b).argmin()
        ela_out[idx_out] = zb[ela_idx] + h[ela_idx]
        idx_out = idx_out + 1
    if (t / dt_plot == np.floor(t / dt_plot)) | (i == nts-1):  # force plotting on the last time step
        #         set(gcf,'Units','normalized')

        pad = 10
        x1 = x[:edge+pad]
        z0 = zb[:edge+pad]
        z1 = zb[:edge+pad] + h[:edge+pad]
        
        try:
            ax[0, 1].collections[0].remove()  # remove the glacier profile before redrawing
        except: 
            pass
        poly = ax[0, 1].fill_between(x1 / 1000, z0, z1, fc='lightblue')
        ax[0, 1].plot(x1 / 1000, z0, c='black', lw=2, )

        # h3 = text('position',[0.77, 0.92],'string',time,'units','normalized','fontsize',16)
        print('outputting')
        #        hout(idx_out,:) = h

        ax[1, 0].plot(t_out, T_out, c='blue', lw=0.25)
        ax[1, 1].plot(t_out, scipy.ndimage.uniform_filter1d(edge_out, 20, mode='mirror')/1000, c='black', lw=2)
        ax[2, 0].plot(t_out, bal_out / (edge_out * 500), c='blue', lw=0.25)
        ax[2, 1].plot(t_out, scipy.ndimage.uniform_filter1d(np.cumsum(bal_out / (edge_out * 500)), 20, mode='mirror'), c='blue', lw=2)
        
        fig.canvas.flush_events()
        fig.canvas.draw()

        # plt.tight_layout()

        # -----------------------------------------
        # end loop over time
        # -----------------------------------------
    
ax[0,0].plot(t_out, scipy.ndimage.uniform_filter1d(ela_out, 20, mode='mirror'), c='black')
ax[0,0].set_xlim(xlim0, tf)
ax[0, 1].set_xlim(0, x1.max() / 1000 * 1.1)
ax[1, 0].plot(t_out, scipy.ndimage.uniform_filter1d(T_out, 20, mode='mirror'), c='blue', lw=2)
ax[1, 0].set_xlim(xlim0, tf)
ax[1, 1].set_xlim(xlim0, tf)
#ax[1, 1].set_ylim(edge_out.min()/1000 - 1, edge_out.max()/1000 + 1)
ax[2, 0].plot(t_out, scipy.ndimage.uniform_filter1d(bal_out / (edge_out * 500), 20, mode='mirror'), c='blue', lw=2)
ax[2, 0].set_xlim(xlim0, tf)
ax[2, 1].plot(t_out, scipy.ndimage.uniform_filter1d(np.cumsum(bal_out / (edge_out * 500)), 20, mode='mirror'), c='blue', lw=2)
ax[2, 1].set_xlim(xlim0, tf)
#ax[2, 1].set_ylim(-80, 80)


# plot extras
anth = pd.read_csv('flowline2d_output_width_ANTH.csv')
anth['T_sm'] = scipy.ndimage.uniform_filter1d(anth['T'], 20, mode='mirror')
anth['edge_sm'] = scipy.ndimage.uniform_filter1d(anth.edge, 20, mode='mirror')/1000
anth['bal_sm'] = scipy.ndimage.uniform_filter1d(anth.bal / (anth.edge * 500), 20, mode='mirror')
anth['cumbal_sm'] = scipy.ndimage.uniform_filter1d(np.cumsum(anth.bal / (anth.edge * 500)), 20, mode='mirror')
anth = anth.iloc[1850:]
ax[1, 0].plot(anth.t, anth['T_sm'], c='red', lw=2)
ax[1, 0].plot(anth.t, anth['T'], c='red', lw=0.25)
ax[1, 1].plot(anth.t, anth['edge_sm'], c='black', lw=2, ls='dashed')
ax[2, 0].plot(anth.t, anth['bal_sm'], c='red', lw=2)
ax[2, 0].plot(anth.t, anth['bal'] / (anth['edge'] * 500), c='red', lw=0.25)
ax[2, 1].plot(anth.t, anth['cumbal_sm'], c='red', lw=2)
    
plt.draw()
plt.savefig(fig_output_name)

# save output?
output = pd.DataFrame().from_dict(dict(t=t_out, T=T_out, edge=edge_out, bal=bal_out, ela=ela_out))
output.to_csv(file_output_name, index=False)
