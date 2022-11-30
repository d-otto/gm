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
from numba import njit

# # How well do the equilibrium responses match?
# # How well does the timescale match?
# # What is the distribution of trends?
# # How does the run length work? Does it follow a Poisson process?
# # Is the dynamical model consistent with a Gaussian pdf?

plt.ioff()
mpl.use('qt5agg')

fig = plt.figure(figsize=(8, 12), dpi=100)
gs = gridspec.GridSpec(3, 2, figure=fig)
ax = np.empty((3, 2), dtype='object')
plt.show(block=False)

ax[0, 0] = fig.add_subplot(gs[0, 0])
ax[0, 1] = fig.add_subplot(gs[0, 1])
ax[0, 1].set_xlabel('Distance (km)')
ax[0, 1].set_ylabel('Elevation (km)')

ax[1, 0] = fig.add_subplot(gs[1, 0])
ax[1, 0].set_xlabel('Time (years)')
ax[1, 0].set_ylabel('T (^oC)')

ax[1, 1] = fig.add_subplot(gs[1, 1])
ax[1, 1].set_xlabel('Time (years)')
ax[1, 1].set_ylabel('L (km)')

ax[2, 0] = fig.add_subplot(gs[2, 0])
ax[2, 0].set_ylabel('Bal (m yr^{-1})')
plt.tight_layout()

# #-----------------
# #define parameters
# #-----------------
rho = 910  # kg/m^3
g = 9.81  # m/s^2
n = 3

mu = 0.65  # melt rate in m /yr /degC
gamma = 6.5e-3  # lapse rate
w_lin = 500  # width in m

# climate forcing
sigT = 0.8  # degC
sigP = 1.0  # m/yr

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
tf = 2025  # final time
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
profile = pd.read_csv(r'C:\Users\drotto\Documents\USGS\glacier-diseq\features\Wolverine_zb_prof_NoDeep.csv')
x_gr = profile['x_gr']
zb_gr = profile['zb_gr']
zb = interp1d(x_gr, zb_gr)
zb = zb(x)

# w = interp1(x_fran,width_fran,x)

w = 500 * np.ones(x.size)  # no width data in this version
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

Qp = np.zeros(x.size)  # Qp equals j+1/2 flux
Qm = np.zeros(x.size)  # Qm equals j-1/2 flux
h = np.zeros(x.size)  # zero out height array
dhdt = np.zeros(x.size)  # zero out thickness rate of change array
h[:] = h0  # initialize height array

# pick climate forcing
climate = pd.read_csv(r"C:\Users\drotto\Documents\USGS\glacier-diseq\features\ClimateRandom.csv")
Trand = climate['Trand']
Prand = climate['Prand']

# initialize climate forcing
Tp = sigT * Trand[1:nyrs + 1]
Pp = sigP * Prand[1:nyrs + 1]  # initialize climate forcing
Tp[0:49] = 0


@njit
def sia(h_ave, dhdx, dzdx, rho=rho, g=g, fd=fd, fs=fs):
    Q = -(rho * g)**3 * h_ave**4 * (dhdx + dzdx)**3 * (fd * h_ave + fs / h_ave)
    return Q 

# -----------------------------------------
# begin loop over time
# -----------------------------------------
yr = 0
idx_out = 0
deltout = 5
nouts = round(nts / deltout)
edge_out = np.full(nouts, fill_value=np.nan, dtype='float')
t_out = np.full(nouts, fill_value=np.nan, dtype='float')
T_out = np.full(nouts, fill_value=np.nan, dtype='float')
bal_out = np.full(nouts, fill_value=np.nan, dtype='float')
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
        T_wk = (13.45 + 1.0 * Tp[yr]) * np.ones(x.size) - gamma * (zb + h)
        if yr >= 1000:
            T_wk = T_wk - 0.25 * (yr - 1000) / 1000
        #           if (yr >= 1850) 
        #               T_wk = T_wk + 1.3*(yr-1850)/150

        #           end
        melt = np.core.umath.clip(mu * T_wk, 0, 100000)
        b = P - melt

    # -----------------------------------------
    # begin loop over space
    # -----------------------------------------
    for j in range(0, nxs - 1):  # this is a kloudge -fix sometime
        if j == 0:
            h_ave = (h[0] + h[1]) / 2
            dhdx = (h[1] - h[0]) / delx
            dzdx = (dzbdx[0] + dzbdx[1]) / 2
            Qp[0] = sia(h_ave=h_ave, dhdx=dhdx, dzdx=dzdx)  # flux at plus half grid point
            Qm[0] = 0  # flux at minus half grid point
            dhdt[0] = b[0] - Qp[0] / (delx / 2) - (Qp[0] + Qm[0]) / (2 * w[0]) * dwdx[0]
        elif (h[j] == 0) & (h[j - 1] > 0):  # glacier toe condition
            Qp[j] = 0
            h_ave = h[j - 1] / 2
            dhdx = -h[j - 1] / delx  # correction inserted ght nov-24-04
            dzdx = (dzbdx[j - 1] + dzbdx[j]) / 2
            Qm[j] = sia(h_ave=h_ave, dhdx=dhdx, dzdx=dzdx)
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
            Qp[j] = sia(h_ave=h_ave, dhdx=dhdx, dzdx=dzdx)
            h_ave = (h[j - 1] + h[j]) / 2
            dhdx = (h[j] - h[j - 1]) / delx
            dzdx = (dzbdx[j] + dzbdx[j - 1]) / 2
            Qm[j] = sia(h_ave=h_ave, dhdx=dhdx, dzdx=dzdx)
            dhdt[j] = b[j] - (Qp[j] - Qm[j]) / delx - (Qp[j] + Qm[j]) / (2 * w[j]) * dwdx[j]
        dhdt[nxs - 1] = 0  # enforce no change at boundary

        # ----------------------------------------
        # end loop over space
        # ----------------------------------------
    h = np.core.umath.clip(h + dhdt * delt, 0, 100000)

    # ----------------------------
    # plot glacier every so often
    # save h , and edge
    # ----------------------------
    if t / deltout == np.floor(t / deltout):
        T_out[idx_out] = T_wk[idx_out] - 3.2
        # edge_out[idx_out] = edge
        t_out[idx_out] = t
        edge_out[idx_out] = delx * max(h[h > 10])

        bal = b * w * delx  # mass added in a given cell units are m^3 yr^-1
        bal_out[idx_out] = np.trapz(bal[:np.argmax(
            h[h > 1])])  # should add up all the mass up to the edge, and be zero in equilibrium (nearly zero)
        idx_out = idx_out + 1
    if t / 10 == np.floor(t / 10):
        #         set(gcf,'Units','normalized')

        x1 = x[:edge]
        z0 = zb[:edge]
        z1 = zb[:edge] + h[:edge]
        ax[0, 1].fill_between(x1 / 1000, z0, z1, fc='lightblue')
        ax[0, 1].plot(x1 / 1000, z0, c='black', lw=2, )

        # h3 = text('position',[0.77, 0.92],'string',time,'units','normalized','fontsize',16)
        print('outputting')
        #        hout(idx_out,:) = h

        ax[1, 0].plot(t_out, T_out, c='blue', lw=0.25)

        ax[1, 1].plot(t_out, edge_out / 1000, c='black', lw=1)
        ax[2, 0].plot(t_out, bal_out / (edge_out * 500), c='black', lw=0.25)

        fig.canvas.flush_events()
        fig.canvas.draw()

        # plt.tight_layout()

        # -----------------------------------------
        # end loop over time
        # -----------------------------------------
ax[1, 0].plot(t_out, np.roll(T_out, 15).mean(), c='blue', lw=1.5)
ax[1, 0].set_xlim(1500, 2025)

#             
# 
# plot(t_out, smooth(T_out, 15), 'b', 'linewidth', 1.5)
# axis([1500 2025 - 2 4])
# hold
# off
# xlabel('Time (years)', 'fontsize', 14)
# ylabel('T (^oC)', 'fontsize', 14)
# grid
# on
# text(2027, 4.2, '(c)', 'fontsize', 12)
# 
# subplot(6, 2, 8)
# hold
# off
# plot(t_out, edge_out / 1000, 'k--', 'linewidth', 2)
# axis([1500 2025 7.5 12])
# 
# ylabel('L (km)', 'fontsize', 14)
# grid
# on
# text(2027, 12.2, '(d)', 'fontsize', 12)
# 
# subplot(6, 2, 9)
# hold
# on
# plot(t_out, smooth(bal_out. / (edge_out * 500), 15), 'b', 'linewidth', 1.5)
# axis([1500 2025 - 2 1.5])
# 
# xlabel('Time (years)', 'fontsize', 14)
# ylabel('Bal (m yr^{-1})', 'fontsize', 14)
# text(2027, 1.65, '(e)', 'fontsize', 12)
# 
# subplot(6, 2, 10)
# hold
# on
# idx = find(t_out > 1500)
# 
# plot(t_out(idx), cumsum(bal_out(idx). / (edge_out(idx) * 500)), 'b', 'linewidth', 1.5)
# axis([1500 2025 - 80 30])
# 
# grid
# on
# set(gca, 'box', 'on')
# set(gca, 'YTick', [-80 - 40 0])
# xlabel('Time (years)', 'fontsize', 14)
# ylabel('Cum. bal. (m)', 'fontsize', 14)
# text(2027, 40, '(f)', 'fontsize', 12)
# 
# #          # Now plot the extra stuff on top
# #          load ExptOutput.mat
# #          load ExptOutput_PreLIA.mat
# #          
# #          idx1 = find(t_out>=1850)
# 
# #          subplot(6,2,7)
# hold
# on
# #          plot(t_out(idx1),T_out(idx1),'r','linewidth',0.25)
# axis([1500 2025 - 2 4])
# 
# #          plot(t_out(idx1),smooth(T_out(idx1),15),'r','linewidth',1.5)
# axis([1500 2025 - 2 4])
# 
# #          xlabel('Time (years)','fontsize',14)
# ylabel('T (^oC)', 'fontsize', 14)
# #          grid on
# #          text(2027, 4.2,'(c)','fontsize',12)
# #          
# #          subplot(6,2,8)
# hold
# on
# #          plot(t_out,edge_out/1000,'k','linewidth',2)
# axis([1500 2025 7.5 12])
# 
# #          ylabel('L (km)','fontsize',14)
# #          grid on
# #          text(2027, 12.2,'(d)','fontsize',12)
# #           
# #          subplot(6,2,9)
# hold
# on
# #          plot(t_out(idx1),bal_out(idx1)./(edge_out(idx1)*500),'r','linewidth',0.25)
# axis([1500 2025 - 2 1.5])
# 
# #          plot(t_out(idx1),smooth(bal_out(idx1)./(edge_out(idx1)*500),15),'r','linewidth',1.5)
# axis([1500 2025 - 2 1.5])
# 
# #          xlabel('Time (years)','fontsize',14)
# ylabel('Bal (m yr^{-1})', 'fontsize', 14)
# #          text(2027, 1.65,'(e)','fontsize',12)
# #          
# #          subplot(6,2,10)
# hold
# on
# #          idx = find(t_out>=1500)
# 
# #          wk = cumsum(bal_out(idx)./(edge_out(idx)*500))
# 
# #          plot(t_out(idx1),wk(end-175:end),'r','linewidth',1.5)
# axis([1500 2025 - 80 30])
# 
# #          grid on
# #          set(gca,'box','on')
# #          set(gca,'YTick',[-80 -40 0])
# #          xlabel('Time (years)','fontsize',14)
# ylabel('Cum. bal. (m)', 'fontsize', 14)
# #          text(2027, 40,'(f)','fontsize',12)
#             



