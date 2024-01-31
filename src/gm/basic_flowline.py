# -*- coding: utf-8 -*-
"""
basic_flowline.py

Description.

Author: drotto
Created: 1/31/24 @ 09:42
Project: gm
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class flowline:
    def __init__(
        self,
        L0,
        h0,
        W,
        dzdx,
        mu,
        Tbar,
        sigT,
        Pbar,
        sigP,
        zb,
        t1,
        t0=0,
        delx=200,
        gamma=6.5e-3,
        rho=916.8,
        f_d=1.9e-24,
        f_s=5.7e-20,
        dhdb=None,
    ):
        """
        This code solves the shallow-ice equations for ice flow down a
        flowline with prescribed bed geometry. See Roe, JGlac, 2011, or
        many, many other places for equations.

        You can provide a time series of climate (accumlation and melt-season
        temperature). The code then calculates integrates the shallow-ice
        equations.

        The standard set of parameters are chosen(ish) to be typical of small Alpine
        glaciers around Mt. Baker, WA. The flowline assumes constant width (1D),
        so the glacier is longer than typical around Mt Baker in order to
        emulate a realistic accumulation area.

        This version of the model was just for fun, so the parameters are
        different from those in Roe, JGalc, 11.

        The code uses an explicit numberical scheme, so is not particularly
        stable. If it blows up, try and shorter time step (delt), or a coarser
        spatial resolution (delx). I am a crappy coder, so there are no
        guarantees it is free of bugs.

        I have tuned the mean glacier length primarily by specifying average
        melt-season temperature at z=0, and mean-annual accumulation, Pbar.

        Right now the code loads in an initial glacier profile from the file
        "glac_init_baker.mat". If you change parameters, you may want to
        integrate the glacier in time with climate anomalies set to zero until
        the glacier attains equilibrium, save that profile, and use it as the
        initial conditions for your new integrations (if that makes sense).

        You can change the basal geometry, glacier size, climate, flow
        parameters,


        Feel free to use for anything, and to change anything. Have fun!
        Gerard (groe@uw.edu - although negligible technical support will be offered!).

        "Most numerical models
        solve the shallow-ice equations (which neglect longitudinal
        stresses; e.g. Hutter, 1983) and incorporate a representation
        of basal sliding. For a one-dimensional flowline following the
        longitudinal profile of a glacier"

        "The first equation represents local
        mass conservation, while the second represents the transla-
        tion and deformation of ice associated with shear stresses. In
        combination, the equations have the form of a nonlinear
        diffusion equation in thickness."

        """

        # %%
        # model parameters
        self.ts = t0  # starting time, yrs
        self.tf = t1  # final time, yrs
        self.delt = 0.05  # initial time step in yrs
        self.delx = delx  # grid spacing, m
        self.xmx = len(dzdx) * delx  # the domain size in m
        self.nxs = int(np.round(self.xmx / self.delx)) - 1  # number of grid points
        self.x = np.arange(0, self.xmx, self.delx)  # x array
        # self.nts = int(np.floor(
        #     (self.tf - self.ts) / self.delt))  # number of time steps ('round' used because need nts as integer)

        # glacier parameters
        self.L0 = L0
        self.h0 = h0
        self.W = W
        self.dzdx = dzdx
        self.rho = rho  # kg/m^3
        self.f_s = f_s
        self.f_d = f_d
        self.g = 9.81  # m/s^2
        self.n = 3  # shallow ice param
        self.A = 2.4e-24  # #Pa^-3 s^-1
        self.K = 2 * self.A * (rho * self.g) ** self.n / (self.n + 2)
        self.K = self.K * np.pi * 1e7  # make units m^2/yr

        # glacier geometry
        self.zb = zb
        self.dzbdx = np.gradient(self.zb, self.x)  # slope of glacer bed geometries.

        # climate parameters
        self.Tbar = Tbar  # average temperature at z=0 [^oC]
        self.sigT = sigT  # standard deviation of temperature [^oC]
        self.Pbar = Pbar  # average value of accumulation [m yr^-1]
        self.sigP = sigP  # standard deviation of accumulation [m yr^-1]
        self.gamma = gamma  # lapse rate  [K m^-1]
        self.mu = mu  # melt factor for ablation [m yr^-1 K^-1]
        self.rng = default_rng()

        # find zeros of flux to find glacier length
        # Note oerlemans value is c=10
        # self.b0 = (3 - (4 / 5000) * self.x)  # mass balance in m/yr
        # self.c = 1  # equals 2tau/(rho g) plastic ice sheet constant from paterson
        # #self.c = 10
        # idx = np.where(np.cumsum(self.b0) < 0)  # indices where flux is less than zero
        # idx = np.min(idx)
        # self.L = self.x[idx]  # approximate length of glacier
        self.L = self.L0 * 0.5
        self.c = 10
        idx = len(self.x * 0.5)

        ###########################################
        # plastic glacier initial profile
        ###########################################
        if isinstance(self.h0, np.ndarray):
            self.h = self.h0
        else:
            self.h = np.zeros_like(self.x, dtype="float")  # zero out initial height array
            self.h[0 : idx - 1] = np.sqrt(self.c * (self.L - self.x[0 : idx - 1]))  # plastic ice profile as initial try
            self.h = np.nan_to_num(self.h)

        self.nyrs = self.tf - self.ts
        self.Pout = np.zeros(self.nyrs)
        self.Tout = np.zeros(self.nyrs)

        self.dhdt_out = []
        self.h_out = []
        self.Qp_out = []
        self.Qm_out = []
        self.edge_out = []
        self.t_out = []
        self.P_out = []
        self.T_out = []
        self.b_out = []
        self.dzbdx_out = []
        self.zb_out = []
        self.melt_out = []

    def run(self, tout=50):
        ###########################################
        # begin loop over time
        ###########################################
        self.yr = self.ts - 1  # so the climate init works properly
        self.steps = []
        idx_out = 0
        self.last_t = self.ts
        self.b = np.zeros_like(self.x, dtype="float")

        def step_time(t, _):
            # t = self.delt*(i)  # time in years
            print(t)
            # define climate if it is the start of a new year
            try:
                delt = t - self.steps[-1]
            except:
                delt = t

            if np.floor(t) > self.yr:  # equal to for first step
                self.yr = self.yr + 1
                print(f"yr = {self.yr} | timestep = {t}")
                self.P = np.ones_like(self.x, dtype="float") * (self.Pbar + self.sigP * self.rng.standard_normal(1))
                self.T_wk = (self.Tbar + self.sigT * self.rng.standard_normal(1)) * np.ones_like(
                    self.x
                ) - self.gamma * self.zb  # todo: this should be based on the ice surface?
                # T_wk = (self.Tbar) * \
                #        np.ones_like(self.x, dtype='float') - \
                #        self.gamma * self.zb + \
                #        yr * 0.01

                self.melt = np.clip(self.mu * self.T_wk, 0, None)
                self.b = (self.P - self.melt) * delt

            ###########################################
            # begin loop over space
            ###########################################

            self.Qp = np.zeros_like(self.x, dtype="float")  # Qp equals j+1/2 flux
            self.Qm = np.zeros_like(self.x, dtype="float")  # Qm equals j+1/2 flux
            self.dhdt = np.zeros_like(self.x, dtype="float")  # zero out thickness rate of change
            for j in np.arange(0, self.nxs, 1):
                if j == 0:
                    h_ave = (self.h[0] + self.h[1]) / 2
                    dhdx = (self.h[1] - self.h[0]) / (self.delx / 2)
                    dzdx = (self.dzbdx[0] + self.dzbdx[1]) / 2
                    self.Qp[0] = -self.K * (dhdx + dzdx) ** 3 * h_ave**5 * delt  # flux at plus half grid point
                    self.Qm[0] = 0  # flux at minus half grid point
                    self.dhdt[0] = (self.b[0] - self.Qp[0]) / (self.delx / 2)

                elif (self.h[j] <= 0) & (self.h[j - 1] > 0):  # glacier toe condition
                    h_ave = self.h[j - 1] / 2
                    dhdx = -self.h[j - 1] / self.delx  # correction inserted ght nov-24-04
                    dzdx = (self.dzbdx[j - 1] + self.dzbdx[j]) / 2
                    self.Qm[j] = -self.K * (dhdx + dzdx) ** 3 * h_ave**5 * delt
                    self.Qp[j] = 0
                    self.dhdt[j] = (self.b[j] + self.Qm[j]) / (self.delx / 2)
                    edge = j  # index of glacier toe - used for fancy plotting

                elif (self.h[j] <= 0) & (self.h[j - 1] <= 0):  # beyond glacier toe - no glacier flux
                    self.dhdt[j] = self.b[j]
                    self.Qp[j] = 0
                    self.Qm[j] = 0

                else:  # within the glacier
                    h_ave = (self.h[j + 1] + self.h[j]) / 2
                    dhdx = (self.h[j + 1] - self.h[j]) / self.delx  # correction inserted ght nov-24-04
                    dzdx = (self.dzbdx[j] + self.dzbdx[j + 1]) / 2
                    self.Qp[j] = -self.K * (dhdx + dzdx) ** 3 * h_ave**5 * delt

                    h_ave = (self.h[j - 1] + self.h[j]) / 2
                    dhdx = (self.h[j] - self.h[j - 1]) / self.delx
                    dzdx = (self.dzbdx[j] + self.dzbdx[j - 1]) / 2
                    self.Qm[j] = -self.K * (dhdx + dzdx) ** 3 * h_ave**5 * delt
                    self.dhdt[j] = (self.b[j] - (self.Qp[j] - self.Qm[j])) / self.delx

                # self.dhdt[self.nxs] = 0  # enforce no change at boundary

            # end of loop over space
            self.dhdt = np.nan_to_num(self.dhdt)
            self.h = self.h + self.dhdt
            self.h = np.clip(self.h, a_min=0, a_max=None)

            # end of loop over time
            if len(self.steps) % tout == 0:
                print(f"output saved at time {t}")
                # np.argmin(np.gradient(res.h.to_numpy())[1], axis=1)
                edge = np.argmin(np.gradient(self.h))  # * self.delx
                self.edge_out.append(edge)
                self.dhdt_out.append(self.dhdt)
                self.h_out.append(self.h)
                self.Qp_out.append(self.Qp)
                self.Qm_out.append(self.Qm)
                self.P_out.append(self.P)
                self.T_out.append(self.T_wk)
                self.b_out.append(self.b)
                self.dzbdx_out.append(self.dzbdx)
                self.zb_out.append(self.zb)
                self.melt_out.append(self.melt)

            # to get delt
            self.steps.append(t)

            # diagnostic result for integration step sizing
            # return self.Qp + self.Qm
            return self.dhdt

        # run the model with rk4
        Qp0 = np.zeros_like(self.x, dtype="float")  # initial value for diagnostic value
        rk4 = scipy.integrate.RK45(
            step_time,
            t_bound=self.tf,
            max_step=1,
            t0=self.ts,
            y0=Qp0,
            first_step=0.0025,
            vectorized=True,
        )
        while True:
            try:
                rk4.step()
            except:
                print("Instability :(")
                break

        # collect results
        res = xr.Dataset(
            data_vars=dict(
                edge=(["t"], np.array(self.edge_out)),
                dzbdx=(["t", "x"], np.vstack(self.dzbdx_out)),
                zb=(["t", "x"], np.vstack(self.zb_out)),
                dhdt=(["t", "x"], np.vstack(self.dhdt_out)),
                h=(["t", "x"], np.vstack(self.h_out)),
                Qp=(["t", "x"], np.vstack(self.Qp_out)),
                Qm=(["t", "x"], np.vstack(self.Qm_out)),
                P=(["t", "x"], np.vstack(self.P_out)),
                T=(["t", "x"], np.vstack(self.T_out)),
                b=(["t", "x"], np.vstack(self.b_out)),
                melt=(["t", "x"], np.vstack(self.melt_out)),
            ),
            coords=dict(
                x=("x", self.x),
                t=("t", np.arange(0, rk4.nfev // tout, 1)),
            ),
            attrs=dict(),
        )

        self.res = res

        mpl.rcParams["image.aspect"] = "auto"
        fig, ax = plt.subplots(1, 1, figsize=(4, 6), dpi=150)
        im = ax.imshow(res.h.to_numpy(), norm=mpl.colors.Normalize(vmin=-0.25, vmax=500))
        ax.set_title("h")
        plt.colorbar(im, ax=ax)
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(4, 6), dpi=150)
        im = ax.imshow(res.T.to_numpy(), norm=mpl.colors.Normalize(vmin=-20, vmax=20))
        ax.set_title("T")
        plt.colorbar(im, ax=ax)
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(4, 6), dpi=150)
        im = ax.imshow(res.melt.to_numpy(), norm=mpl.colors.Normalize(vmin=-1, vmax=15))
        ax.set_title("melt")
        plt.colorbar(im, ax=ax)
        plt.show()

        # col 1
        fig, ax = plt.subplots(1, 1, figsize=(4, 6), dpi=150)
        im = ax.imshow(res.Qp.to_numpy(), vmin=0, vmax=100)
        ax.set_title("Qp")
        plt.colorbar(im, ax=ax)
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(4, 6), dpi=150)
        im = ax.imshow(res.Qm.to_numpy(), vmin=0, vmax=100)
        ax.set_title("Qm")
        plt.colorbar(im, ax=ax)
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(4, 6), dpi=150)
        norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=-0.25, vmax=0.25)
        im = ax.imshow(np.gradient(res.h.to_numpy())[0], cmap="bwr_r", norm=norm)
        ax.set_title("gradient(h)[0]")
        plt.colorbar(im, ax=ax, norm=norm)
        plt.show()

        # col 2
        fig, ax = plt.subplots(1, 1, figsize=(4, 6), dpi=150)
        norm = mpl.colors.SymLogNorm(linthresh=0.01, linscale=0.01)
        im = ax.imshow(res.dhdt.to_numpy(), cmap="bwr_r", norm=norm)
        ax.set_title("dhdt")
        plt.colorbar(im, ax=ax, norm=norm)
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(4, 6), dpi=150)
        norm = mpl.colors.SymLogNorm(linthresh=0.01, linscale=1, vmin=-10, vmax=10)
        im = ax.imshow(res.b.to_numpy(), cmap="bwr_r", norm=norm)
        ax.set_title("b")
        plt.colorbar(im, ax=ax, norm=norm)
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(4, 6), dpi=150)
        im = ax.imshow(res.P.to_numpy())
        ax.set_title("P")
        plt.colorbar(im, ax=ax)
        plt.show()

        # col 3
        fig, ax = plt.subplots(1, 1, figsize=(4, 6), dpi=150)
        norm = mpl.colors.TwoSlopeNorm(vcenter=0)
        im = ax.imshow(np.gradient(res.h.to_numpy())[1], cmap="bwr_r", norm=norm)
        ax.set_title("gradient(h)[1]")
        plt.colorbar(im, ax=ax, norm=norm)
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(4, 6), dpi=150)
        edge = res.edge.to_series()
        im = ax.plot(edge, edge.index)
        ax.invert_yaxis()
        ax.set_title("edge")
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(4, 6), dpi=150)
        norm = mpl.colors.SymLogNorm(linthresh=0.01, linscale=0.01)
        im = ax.imshow(res.Qm.to_numpy() - res.Qp.to_numpy(), norm=norm)
        ax.set_title("Qm - Qp")
        plt.colorbar(im, ax=ax, norm=norm)
        plt.show()