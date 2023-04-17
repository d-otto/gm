# -*- coding: utf-8 -*-

import copy
import collections
import copy
import dill
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import scipy.ndimage
import xarray as xr
import numba as nb
from numpy.random import default_rng
from scipy.interpolate import interp1d
from scipy.stats import norm
import logging
from tqdm import tqdm


# %%

# preallocate empty array and assign slice by chrisaycock
def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


# relevant as of Numpy 1.24
# https://github.com/iperov/DeepFaceLab/pull/5618/commits/52dcf152e2b5aedf96b5c8a343cad24585492df6
def fast_clip(array, min_value, max_value):
    return np.minimum(max_value, np.maximum(array, min_value, out=array), out=array)


# %%


class gm1s:
    def __init__(
        self,
        L=None,
        ts=None,
        H=None,
        bt=None,
        tau=None,
        g=-9.81,
        rho=916.8,
        b_p=0,
        dbdt=None,
        mode=None,
    ):
        if isinstance(dbdt, (int, float)) & (dbdt is not None):
            self.mode = "linear"
            self.bt_p = np.full_like(ts, fill_value=dbdt) + bt
            self.b_p = np.full_like(ts, fill_value=dbdt)
            self.bt_eq = bt
        elif isinstance(b_p, (collections.abc.Sequence, np.ndarray)):
            self.mode = "discrete"
            self.bt_p = bt + b_p
            self.bt_eq = bt
            self.b_p = b_p
        elif isinstance(b_p, (int, float)):
            # step change
            # todo: implement sol'n for step change
            self.mode = "discrete"
            b_p = np.full_like(ts, fill_value=b_p)
            self.bt_p = bt + b_p
            self.bt_eq = bt
            self.b_p = b_p

        self.ts = ts
        self.dt = np.diff(ts, prepend=-1)
        self.L_bar = L  # steady state without perturbations (m)
        self.H = H  # m
        self.dbdt = dbdt

        self.beta = self.L_bar / H  # eq. 1
        if tau is None:
            self.tau = -H / bt
        else:
            self.tau = tau

        if self.mode == "linear":
            self.linear()
        elif self.mode == "discrete":
            self.discrete()

    def linear(self):
        self.tau = -self.H / self.bt_p
        self.L_eq = self.tau * self.beta * self.bt_p * (self.ts - self.tau)
        self.L_p = np.zeros_like(self.ts, dtype="float")

        # Christian et al eq. 4
        if self.mode == "linear":
            for i, t in enumerate(self.ts):
                if self.bt_p[i] == 0:
                    self.L_p[i] = 0
                    continue

                self.L_p[i] = (
                    self.tau[i] * self.beta * self.bt_p[i] * (t - self.tau[i] * (1 - np.exp(-t / self.tau[i])))
                )

        self.L = self.L_bar + np.cumsum(self.L_p)

    def discrete(self):
        self.L_p = np.empty_like(self.ts, dtype="float")

        for i, t in enumerate(self.ts):
            # Roe and Baker (2014) eq. 8
            if i == 0:
                self.L_p[i] = self.beta * t * self.b_p[i]
                continue
            self.L_p[i] = (1 - self.dt[i] / self.tau) * self.L_p[i - 1] + self.beta * self.dt[i] * self.b_p[i]
        self.L = self.L_bar + self.L_p
        self.L_eq = self.tau * self.beta * self.bt_p
        self.dL = abs(self.L_p[-1])

    def to_xarray(self):
        import xarray as xr

        ds = xr.Dataset(
            data_vars=dict(
                bp=("t", self.b_p),
                Pp=("t", self.Pp),
                Tp=("t", self.Tp),
            ),
            coords=dict(t=self.ts),
            attrs=dict(),
        )

        return ds

    def to_pandas(self):
        import pandas as pd

        df = pd.DataFrame(
            data=dict(
                bp=self.bp,
                Pp=self.Pp,
                Tp=self.Tp,
            ),
            index=pd.Index(self.t, name="t"),
        )

        return df

    def to_tidy(self):
        import pandas as pd

        df = self.to_pandas().reset_index()
        df = pd.melt(df, id_vars=["t"])

        return df


################################################################################
class gm3s:
    """Real 3-stage linear glacier model

    Assumes Lambda << L_bar and h2_p, h3_p ~= H. Therefore, V1_p ~= h1_p * L_bar,
    V2_p ~= h2_p * lambda, and V3_p ~= (L_p - Lambda)*H ... (see figure)

    Parameters
    ----------
    mode : str, optional
        Calculate anomalies based on mass balance 'b' or length 'l'. Default is 'b'.
    ts : int
        Timeseries of years.
    mu : numeric
        Melt factor [m yr**-1 K**-1]
    Atot : numeric
        Total area of the glacier [m**2]
    ATgt0 : numeric
        Area of the glacier where some melting occurs [m**2]
    Aabl : numeric
        Ablation area [m**2]
    w : numeric
        Characteristic width of the glacier tongue [m].
    H : numeric
        Characteristic ice thickness near the terminus [m]
    gamma : numeric
        Assumed surface lapse rate [K m**-1]
    dzdx : numeric
        Assumed basal slope [no units]
    sigP : numeric
        Std. dev. of accumulation variability [m yr**-1]
    sigT : numeric
        Std. dev. of melt-season temperature variability [m yr**-1]
    sigb : numeric
        Std. dev. of annual-mean mass balance [m yr**-1]. One of `b` or `sigb` is required for `mode='b'`
    T : array-like
        Annual melt season temperature anomaly. One of `T` or `sigT` is required for `mode='l'`.
    P : array-like
        Annual accumulation anomaly. One of `P` or `sigP` is required  for `mode='l'`.
    b : array-like
        Annual mass balance anomaly. One of `b` or `sigb` is required for `mode='b'`.

    Returns
    -------
    linear_1s : object





    """

    def __init__(
        self,
        L,
        H,
        ts,
        mode="b",
        bt=None,
        b_p=None,
        tau=None,
        dzdx=None,
        Atot=None,
        W=None,
        mu=None,
        gamma=None,
        Aabl=None,
        sigT=None,
        sigP=None,
        sigb=None,
        P0=None,
        T0=None,
        T_p=None,
        P_p=None,
        ATgt0=None,
        zb=None,
    ):
        self.ts = ts
        self.dt = np.diff(self.ts + 1, prepend=ts[0])  # works idk why
        # self.dt = np.diff(self.ts)  # works idk why

        if mode == "l":
            # note at this point you could create your own climate time series, using
            # random forcing, trends, oscillations etc.
            # Tp = array of melt-season temperature anomalise
            # Pp = array of accumlation anomalies
            # bp = array of mass balance anomalies
            self.alpha = mu * ATgt0 * self.dt / (W * H)
            if T_p is None:
                T_p = np.zeros_like(ts)
            if P_p is None:
                P_p = np.zeros_like(ts)
            T_p = (T0 - gamma * zb) + T_p
            P_p = P0 + P_p
            if b_p is None:  # is apparently faster for numpy 1.17
                b_p = P_p - T_p * self.alpha

            self.ATgt0 = ATgt0
            self.sigT = sigT
            self.sigP = sigP
            self.sigb = sigb
            self.P0 = P0
            self.Aabl = Aabl
            self.gamma = gamma
            self.mu = mu
        if mode == "b":
            if isinstance(b_p, (collections.abc.Sequence, np.ndarray)):
                pass
            elif isinstance(b_p, (int, float)):
                # step change
                # todo: implement sol'n for step change
                b_p = np.full_like(ts, fill_value=b_p)

        self.mode = mode
        self.bt_p = bt + b_p
        self.bt_eq = bt
        self.L_bar = L  # m
        self.H = H  # m
        self.dzdx_s = dzdx

        self.Atot = Atot
        self.W = W

        # glacier memory [ys]
        # this is the glacier response time  (i.e., memory) based on the above glacier geometry
        # if you like, just pick a different time scale to see what happens.
        # Or also, use the simple, tau = hbar/b_term, if you know the terminus
        # balance rate from, e.g., observations
        if tau is None:
            try:
                self.tau = W * H / (mu * gamma * dzdx * Aabl)
            except:
                try:
                    self.tau = -H / bt
                except:
                    pass
        else:
            self.tau = tau

        # coefficient needed in the model integrations
        # keep fixed - they are intrinsic to 3-stage model
        self.eps = 1 / np.sqrt(3)
        self.K = 1 - self.dt / (self.eps * self.tau)
        self.beta = self.Atot * self.dt / (self.W * self.H)
        self.T_p = T_p
        self.P_p = P_p

    def copy(self):
        return copy.deepcopy(self)

    def to_pandas(self):
        import pandas as pd

        df = pd.DataFrame(
            data=dict(
                L_p=self.L_p,
                L_eq=self.L_eq,
                dL=self.dL,
                bt_p=self.bt_p,
            ),
            index=pd.Index(self.ts, name="t"),
        )
        return df

    def linear(self, bt=None):
        # preeeetty sure this doesn't work at all

        if bt is not None:
            self.b_p = bt
            self.bt_p = self.bt_eq + self.b_p

        # convenience renaming
        tau = self.tau
        L_bar = self.L_bar
        H = self.H
        eps = self.eps
        ts = self.ts

        n_steps = len(ts)
        self.h = np.zeros(n_steps)
        self.F = np.zeros(n_steps)
        self.L = np.zeros(n_steps)
        self.L_p = np.zeros(n_steps)
        self.L_debug = np.zeros(n_steps)

        self.h[0] = 0
        # self.F[0] = 0
        self.L[0] = self.L_bar
        # self.L_p[0] = 0
        # self.L_debug[0] = 0

        for i, t in enumerate(ts):

            self.h[i] = (1 - self.dt[i] / (eps * tau)) * self.h[t - self.dt[i]] + self.bt_p[i]
            self.F[i] = (1 - self.dt[i] / (eps * tau)) * self.F[t - self.dt[i]] + L_bar / (eps * tau) ** 2 * self.h[
                i
            ]  # writing F2 as F
            self.L[i] = (1 - self.dt[i] / (eps * tau)) * self.L[t - self.dt[i]] + self.F[i] / (eps * H)
            self.L_p[i] = self.L[i] - self.L[t - self.dt[i]]

            try:
                self.L_debug = (
                    self.dt * self.L_bar / (eps * self.H) * (self.dt / (eps * tau)) ** 2 * self.bt_p[t - 3]
                )  # if I specified everything right, this should be the same is L[t]
            except:
                pass

    def run(self):
        # convenience renaming
        K = self.K
        eps = self.eps
        tau = self.tau
        beta = self.beta
        L_bar = self.L_bar

        L_p = np.zeros_like(self.ts, dtype="float")

        # L3s(i) = 3 * phi * L3s(i - 1) -
        # 3 * phi ^ 2 * L3s(i - 2)
        # + 1 * phi ^ 3 * L3s(i - 3)...
        # + dt ^ 3 * tau / (eps * tau) ^ 3 * (beta * bp(i - 3))

        if self.mode == "b":
            for i, t in enumerate(self.ts):
                if i <= 3:
                    continue
                L_p[i] = (
                    3 * K[i] * L_p[i - 1]
                    - 3 * K[i] ** 2 * L_p[i - 2]
                    + 1 * K[i] ** 3 * L_p[i - 3]
                    + self.dt[i] ** 3 * tau / (eps * tau) ** 3 * (beta[i] * self.b_p[i - 3])
                )

        if self.mode == "l":
            for i, t in enumerate(self.ts):
                if i <= 3:
                    continue
                L_p[i] = (
                    3 * K[i] * L_p[i - 1]
                    - 3 * K[i] ** 2 * L_p[i - 2]
                    + 1 * K[i] ** 3 * L_p[i - 3]
                    + self.dt[i] ** 3
                    * tau
                    / (eps * tau) ** 3
                    * (beta[i] * self.P_p[i - 3] - self.alpha[i] * self.T_p[i - 3])
                )

        self.L_p = L_p
        self.L = self.L_bar + self.L_p
        self.L_eq = self.tau * self.beta * self.bt_p
        self.dL = abs(L_p[-1])

        return self

    def power_spectrum(self, freq, sig_L_1s):
        P0 = 4 * self.tau * sig_L_1s  # power spectrum in the limit f -> 0 using the variance from the 1s model
        P_spec = (
            P0 * (1 - self.K) ** 6 / (1 - 2 * self.K * np.cos(2 * np.pi * freq * self.dt) + self.K**2) ** 3
        )  # eq. 20
        return P_spec

    def phase(self, freq):
        """Mostly correct?"""

        H = (
            np.exp(-6 * np.pi * 1j * freq * self.dt) / (1 - self.K * np.exp(-2 * np.pi * 1j * freq * self.dt)) ** 3
        )  # eq. 19b
        phase = np.angle(H * 1j, deg=True)

        return phase

    def acf(self, t):
        """Based on the continuous form of the 3-stage equations"""
        eps = self.eps
        tau = self.tau

        acf = np.exp(-t / (eps * tau)) * (1 + t / (eps * tau) + 1 / 3 * (t / (eps * tau)) ** 2)
        return acf


###############################################################################
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


###############################################################################
class flowline2d:
    def __init__(
        self,
        x_gr,
        zb_gr,
        x_geom,
        w_geom,
        xmx,
        sigT,
        sigP,
        T0,
        P0,
        T=None,
        P=None,
        x_init=None,
        h_init=None,
        profile=None,
        t_stab=None,
        temp=None,
        gamma=6.5e-3,
        dpdz=0,
        mu=0.65,
        g=-9.81,
        rho=916.8,
        fd=1.9e-24,
        fs=5.7e-20,
        delx=50,
        delt=0.0125 / 8,
        ts=0,
        tf=2025,
        dt_plot=100,
        rt_plot=False,
        xlim0=None,
        min_thick=1,
        hmb=True,
    ):
        """2d flowline model

        This module demonstrates documentation as specified by the `NumPy
        Documentation HOWTO`_. Docstrings may extend over multiple lines. Sections
        are created with a section header followed by an underline of equal length.

        Parameters
        ----------
        x_gr : array-like
            x dimension of bed topography; m
        zb_gr : array-like
            z dimension of bed topography; m
        x_geom : array-like
            x dimension of glacier width; m
        w_geom : array-like
            width of glacier along flowline; m
        x_init : array-like
            x dimension of initial thickness profile; m
        h_init : array-like
            initial ice thickness profile; m
        T : array-like
            random state for temperature
        P : array-like
            random state for precipitation
        rho : float
            Ice density kg/m^3
        g : float
            Gravity m/s^2
        mu : float
            Melt rate m/yr/degC
        n : float
            Glenn's flow law parameter. Default n = 3
        gamma : float
            Temperature lapse rate degC/km
        sigT : float
            Temperature standard deviation degC
        sigP : float
            Precipitation standard deviation m/yr
        T0 : float
            Baseline temperature degC
        fd : float
            Deformation parameter Pa^-3 s^-2
        fs : float
            Sliding parameter Pa^-3 s^-1 m^2
        xmx : int
            Domain size in m
        delx : int
            Grid spacing in m
        delt : float
            Time step in yrs suitable for 200m yr^-1
        ts : float
            Starting time yr
        tf : float
            Ending time yr
        dt_plot : int
            Plotting interval yr
        rt_plot : bool
            Whether there should be real time plotting while the model is running.
        xlim0 : float
            Left limit for plots (yrs)


        Returns
        -------
        bool
            True if successful, False otherwise.

        """

        # # How well do the equilibrium responses match?
        # # How well does the timescale match?
        # # What is the distribution of trends?
        # # How does the run length work? Does it follow a Poisson process?
        # # Is the dynamical model consistent with a Gaussian pdf?

        if rt_plot:
            plt.ioff()
            mpl.use("qt5agg")

        # #-----------------
        # #define parameters
        # #-----------------

        xmx = delx * round(xmx / delx)  # round to neaest delx
        x = np.arange(0, xmx, delx)  # x array
        nxs = len(x)

        fd = fd * np.pi * 1e7
        fs = fs * np.pi * 1e7  # convert from seconds to years

        # ---------------------------------
        # different glacier bed geometries
        # ---------------------------------
        self.load_profile(profile, x)
        zb = interp1d(x_gr, zb_gr)
        zb = zb(x)
        w = interp1d(x_geom, w_geom)
        w = w(x)
        dzbdx = np.gradient(zb, x)  # slope of glacer bed geometries.
        dwdx = np.gradient(w, x)

        # geometry errors/warnings
        if any(dzbdx == 0):
            logging.warning(f'Bed slope is zero at {(dzbdx == 0).argmax()}.')
        if any(dzbdx[0:2] > 0):
            logging.warning('The slope of the bed at the top of the glacier is positive. This may cause instabilities.')

        # initialize climate forcing
        # tf = tf + 1  # results in common-sense arguments. The last year executed is tf as supplied to the fn.
        self.nts = round(np.floor((tf - ts) / delt))  # number of time steps ('round' used because need nts as integer)
        nyrs = tf - ts + 1
        if T is None:
            T = np.zeros(nyrs)
        if P is None:
            P = np.zeros(nyrs)
        self.Tp = sigT * T
        self.Pp = sigP * P
        if temp is None:
            temp = pd.Series(np.zeros(nyrs), index=np.arange(ts, tf + 1, 1))
        if t_stab:
            self.Tp.iloc[:t_stab] = 0
            self.Pp.iloc[:t_stab] = 0
            temp.iloc[:t_stab] = 0
        self.ts = ts
        self.tf = tf
        self.temp = temp
        self.T = T
        self.P = P
        self.t_stab = t_stab

        # constants
        self.sigT = sigT
        self.sigP = sigP
        self.P0 = P0
        self.T0 = T0
        self.mu = mu
        self.gamma = gamma
        self.rho = rho
        self.g = g
        self.min_thick = min_thick
        self.nxs = nxs
        self.delt = delt
        self.dzbdx = dzbdx
        self.dwdx = dwdx
        self.fd = fd
        self.fs = fs
        self.x = x
        self.zb = zb
        self.nxs = nxs
        self.delx = delx
        self.w = w
        
        # functions
        if callable(dpdz) is False:
            self.dpdz = lambda x: dpdz * x
        else:
            self.dpdz = dpdz

        # option flags
        self.hmb = hmb
        self.dt_plot = dt_plot
        self.rt_plot = rt_plot

        # runtime flags
        self.no_error = True

    def run(self, **kwargs):
        if kwargs:
            self.__dict__.update(kwargs)
        if "profile" in kwargs.keys():
            self.load_profile(self.profile, self.x)

        # output
        nouts = int((self.nts * self.delt) // 1)
        self.edge_idx = np.full(nouts, fill_value=np.nan, dtype="int")
        self.edge = np.full(nouts, fill_value=np.nan, dtype="float")
        self.t = np.full(nouts, fill_value=np.nan, dtype="float")
        self.T = np.full(nouts, fill_value=np.nan, dtype="float")
        
        self.gwb = np.full(nouts, fill_value=np.nan, dtype="float")
        self.ela = np.full(nouts, fill_value=np.nan, dtype="float")
        self.area = np.full(nouts, fill_value=np.nan, dtype="float")
        self.h = np.full((nouts, self.nxs), fill_value=np.nan, dtype="float")
        self.b = np.full((nouts, self.nxs), fill_value=np.nan, dtype="float")
        self.melt = np.full((nouts, self.nxs), fill_value=np.nan, dtype="float")
        self.ela_idx = np.full(nouts, fill_value=np.nan, dtype="int")
        self.F = np.full((nouts, self.nxs), fill_value=np.nan, dtype="float")
        self.P = np.full((nouts, self.nxs), fill_value=np.nan, dtype="float")

        @nb.njit(fastmath={"contract", "arcp", "nsz", "afn", "reassoc"})
        def space_loop(h, b, x, rho, g, nxs, delx, dzbdx, fd, fs, dwdx, w, delt, min_thick):
            Qp = np.zeros(x.size)  # Qp equals j+1/2 flux
            Qm = np.zeros(x.size)  # Qm equals j-1/2 flux
            dhdt = np.zeros(x.size)  # zero out thickness rate of change array
            rho_g_cu = (rho * g) ** 3  # precompute for speed
            dzdx = (dzbdx[:-1] + dzbdx[1:]) / 2  # slope at plus half a grid point
            # -----------------------------------------
            # begin loop over space
            # -----------------------------------------
            # todo: I wonder if this should be evaluated backwards?
            for j in range(0, nxs - 1):  # this is a kloudge -fix sometime
                if j == 0:
                    h_ave = (h[0] + h[1]) / 2
                    dhdx = (h[1] - h[0]) / delx
                    Qp[0] = (
                        rho_g_cu * (dhdx + dzdx[j]) ** 3 * (fd * h_ave**5 + fs * h_ave**3)  # top of glacier qp
                    )  # flux at plus half grid point
                    # Qm[0] = 0  # flux at minus half grid point
                    dhdt[0] = b[0] - Qp[0] / (delx / 2) - (Qp[0] + Qm[0]) / (2 * w[0]) * dwdx[0]
                elif (h[j] <= 0) & (h[j - 1] > 1):  # glacier toe condition
                    # Qp[j] = 0
                    h_ave = h[j - 1] / 2
                    dhdx = -h[j - 1] / delx  # correction inserted ght nov-24-04
                    Qm[j] = rho_g_cu * (dhdx + dzdx[j - 1]) ** 3 * (fd * h_ave**5 + fs * h_ave**3)  # glacier toe qm
                    dhdt[j] = b[j] + Qm[j] / delx - (Qp[j] + Qm[j]) / (2 * w[j]) * dwdx[j]
                elif (h[j] == 0) & (h[j - 1] < 1):  # beyond glacier toe - no glacier flux
                    dhdt[j] = b[j]  # todo: verify that this is actually being evaluated
                    # Qp[j] = 0
                    # Qm[j] = 0
                else:  # within the glacier
                    h_ave = (h[j + 1] + h[j]) / 2
                    dhdx = (h[j + 1] - h[j]) / delx  # correction inserted ght nov-24-04
                    Qp[j] = rho_g_cu * (dhdx + dzdx[j]) ** 3 * (fd * h_ave**5 + fs * h_ave**3)  # Within glacier qp
                    h_ave = (h[j - 1] + h[j]) / 2
                    dhdx = (h[j] - h[j - 1]) / delx
                    Qm[j] = (
                        rho_g_cu * (dhdx + dzdx[j - 1]) ** 3 * (fd * h_ave**5 + fs * h_ave**3)
                    )  # within glacier qm
                    dhdt[j] = b[j] - (Qp[j] - Qm[j]) / delx - (Qp[j] + Qm[j]) / (2 * w[j]) * dwdx[j]
            dhdt[nxs - 1] = 0  # enforce no change at boundary
            # ----------------------------------------
            # end loop over space
            # ----------------------------------------
            # h = fast_clip(h + dhdt * delt, 0, 10000)
            # h = np.minimum(10000, np.maximum(h + dhdt * delt, 0))
            h = np.core.umath.maximum(h + dhdt * delt, 0)
            edge = (
                len(h) - np.searchsorted(h[::-1], min_thick) - 1
            )  # very fast location of the terminus https://stackoverflow.com/questions/16243955/numpy-first-occurrence-of-value-greater-than-existing-value
            F = Qm - Qp
            return h, edge, F

        yr = self.ts - 1  # - 1 because we start the time loop by incramenting the year
        idx_out = 0
        deltout = 1

        if self.rt_plot:
            self.fig, self.ax = self._init_plot()

        h = self.h0  # define initial height
        for i in tqdm(
            range(0, self.nts),
            unit_scale=self.delt,
            unit="yrs",
            bar_format="{desc}: {percentage:2.0f}%|{bar}| {n:.1f}/{total:.1f} [{elapsed}<{remaining}, {rate_fmt}{postfix}",
            ascii=True,
            ncols=100,
        ):
            t = self.delt * i  # time in years

            # define climate every year
            if t == t // 1:
                yr = yr + 1
                
                if self.hmb:
                    T_wk = (self.T0 + self.Tp[yr]) * np.ones(self.x.size) - self.gamma * (
                        self.zb + h
                    )  # adding h to zb = altitude-mass-balance feedback
                    P = (self.P0 + self.Pp[yr]) * np.ones(self.x.size) - self.dpdz(self.zb + h)
                else:
                    T_wk = (self.T0 + self.Tp[yr]) * np.ones(
                        self.x.size
                    ) - self.gamma * self.zb  # adding h to zb = altitude-mass-balance feedback
                    P = (self.P0 + self.Pp[yr]) * np.ones(self.x.size) - self.dpdz(self.zb)
                T_wk = T_wk + self.temp[yr]  # add temperature forcing

                melt = np.maximum(self.mu * T_wk, 0)  # this is apparently faster than clip for numpy 1.17
                b = P - melt

            # loop over space (solve SIA)
            # this is the entire model, really
            h, edge_idx, F = space_loop(
                h,
                b,
                self.x,
                self.rho,
                self.g,
                self.nxs,
                self.delx,
                self.dzbdx,
                self.fd,
                self.fs,
                self.dwdx,
                self.w,
                self.delt,
                self.min_thick,
            )

            if t / deltout == np.floor(t / deltout):
                # Save outputs
                area = np.sum(self.w[:edge_idx]) * self.delx
                bal = b * self.w * self.delx  # mass added in a given cell units are m^3 yr^-1
                # bal[edge+1] =
                self.gwb[idx_out] = bal[
                    :edge_idx
                ].sum()  # should add up all the mass up to the edge, and be zero in equilibrium (nearly zero)
                self.T[idx_out] = self.T0 + self.Tp[yr] + self.temp[yr]  # input temperature
                
                self.t[idx_out] = t + self.ts
                self.edge_idx[idx_out] = edge_idx
                self.edge[idx_out] = edge_idx * self.delx
                self.h[idx_out, :] = h
                self.area[idx_out] = area
                ela_idx = np.abs(b).argmin()
                self.ela_idx[idx_out] = ela_idx
                self.ela[idx_out] = self.zb[ela_idx] + h[ela_idx]
                self.b[idx_out, :] = b
                # b_out[idx_out, edge+1:] = np.nan
                self.P[idx_out, :] = P
                self.melt[idx_out, :] = melt
                self.F[idx_out, :] = F
                idx_out = idx_out + 1

                if self.rt_plot:
                    self._rt_plot(t)

                # -----------------------------------------
                # end loop over time
                # -----------------------------------------

        if self.h[-1, 0] is np.nan:
            self.no_error = False
        else:
            self.no_error = True

        return copy.deepcopy(self)

    def load_profile(self, profile, x):
        # load initial profile
        if profile:
            try:
                h_init = profile.h[-1, :].copy()
                x_init = profile.x.copy()
            except:
                with open(profile, 'rb') as f:
                    last_run = dill.load(f)
                h_init = np.array(last_run.h[-1, :])
                x_init = np.array(last_run.x)

        try:
            h0 = interp1d(x_init, h_init, "linear")
            h0 = h0(x)
        except:
            logging.warning(
                f"A value in x exceeds x_init for interpolation of h_init to h0. Proceeding with extrapolation. x_init.max() = {x_init.max()}, x.max() = {x.max()}"
            )
            h0 = interp1d(x_init, h_init, "linear", fill_value="extrapolate")
            h0 = h0(x)
        self.h0 = h0
        return x_init, h_init

    def plot_full(self, xlim0=None, smooth=20):
        """This is a docstring

        This is the longer portion of the docstring.

        Parameters
        ----------------
        xlim0 : float
            left x-limit for figure (years)

        Returns
        ----------------
        fig : Figure
            It's a figure??

        """
        if xlim0 is None:
            xlim0 = self.ts

        pad = 20
        pedge = int(self.edge_idx[-1]) + pad
        self.pedge = pedge
        x1 = self.x[:pedge]
        z0 = self.zb[:pedge]
        z1 = z0 + self.h[-1, :pedge]

        fig, ax = self._init_plot()
        ax[0, 0].plot(
            self.t,
            scipy.ndimage.uniform_filter1d(self.area / 1e6, smooth, mode="mirror"),
            c="black",
            label=f"MA-{smooth}",
        )
        poly1 = ax[0, 1].fill_between(
            x1 / 1000,
            z0,
            z1,
            fc="lightblue",
            ec="lightblue",
            label=f"{self.tf} profile",
        )
        ax[0, 1].plot(
            x1 / 1000,
            z0,
            c="black",
            lw=2,
        )
        ax[0, 2].hist(
            self.gwb / self.area,
            bins=100,
            density=True,
        )
        ax[0, 2].axvline(x=(self.gwb / self.area).mean(), ls="--", lw=2, c="black", label="Mean")
        ax[0, 2].annotate(
            f"b_s = {np.std(self.gwb / self.area):0.4f}\n" f"mean = {np.mean(self.gwb/self.area):0.4f}",
            xy=(0.05, 0.05),
            xycoords="axes fraction",
        )
        ax[1, 2].hist(
            self.edge / 1000,
            bins=30,
            density=True,
        )
        ax[1, 2].axvline(x=(self.edge / 1000).mean(), ls="--", lw=2, c="black", label="Mean")
        ax[1, 2].annotate(
            f"$\sigma_l$ = {np.std(self.edge / 1000):0.4f}\n" f"mean = {np.mean(self.edge):0.4f}",
            xy=(0.05, 0.05),
            xycoords="axes fraction",
        )
        ax[0, 0].set_xlim(xlim0, self.tf)
        ax[1, 0].plot(
            self.t,
            scipy.ndimage.uniform_filter1d(self.ela, smooth, mode="mirror"),
            c="black",
            label=f"MA-{smooth}",
        )
        ax[1, 0].set_xlim(xlim0, self.tf)
        ax[2, 1].set_xlim(0, x1.max() / 1000 * 1.1)
        # ax[2, 0].plot(self.t, self.T, c="blue", lw=0.25, alpha=0.25)
        ax[2, 0].plot(
            self.t,
            scipy.ndimage.uniform_filter1d(self.T, smooth, mode="mirror"),
            c="black",
            lw=1,
            alpha=0.5,
            label=f"MA-{smooth}",
        )
        ax[2, 0].plot(
            self.t,
            scipy.ndimage.uniform_filter1d(self.T, 300, mode="mirror"),
            c="red",
            lw=1,
            label="MA-300",
        )
        ax[2, 0].set_xlim(xlim0, self.tf)
        ax[2, 1].plot(
            self.t,
            self.edge / 1000,
            c="black",
            lw=2,
            label=f"Length",
        )
        ax[2, 1].set_xlim(xlim0, self.tf)
        ax[2, 2].scatter(
            scipy.ndimage.uniform_filter1d(self.edge / 1000, 100, mode="mirror"),
            scipy.ndimage.uniform_filter1d(self.gwb / self.area, 100, mode="mirror"),
            c=self.t,
            cmap="viridis",
            s=2,
            label="MA-100",
        )
        # ax[2, 2].set_xlim(xlim0, self.tf)
        # ax[3, 0].plot(self.t, self.gwb / self.area, c="blue", lw=0.25)
        ax[3, 0].plot(
            self.t,
            scipy.ndimage.uniform_filter1d(self.gwb / self.area, smooth, mode="mirror"),
            c="black",
            lw=1,
            alpha=0.5,
            label=f"MA-{smooth}",
        )
        ax[3, 0].plot(
            self.t,
            scipy.ndimage.uniform_filter1d(self.gwb / self.area, 300, mode="mirror"),
            c="red",
            lw=1,
            label="MA-300",
        )
        ax[3, 0].set_xlim(xlim0, self.tf)
        ax[3, 1].plot(
            self.t,
            scipy.ndimage.uniform_filter1d(
                np.cumsum(self.gwb / self.area),
                smooth,
                mode="mirror",
            ),
            c="blue",
            lw=2,
            label=f"MA-{smooth}",
        )
        ax[3, 1].set_xlim(xlim0, self.tf)
        scat = ax[3, 2].scatter(
            scipy.ndimage.uniform_filter1d(self.h.mean(axis=1), smooth, mode="mirror"),
            scipy.ndimage.uniform_filter1d(self.edge / 1000, smooth, mode="mirror"),
            c=self.t,
            cmap="viridis",
            s=2,
            label=f"MA-{smooth}",
        )
        fig.colorbar(scat, ax=ax[2:, 2], label="Year")
        for axis in ax.ravel():
            try:
                axis.legend(loc="upper left")
            except:
                pass

        return fig, ax

    def plot(self, smooth=1):
        def sm(d):
            return scipy.ndimage.uniform_filter1d(d, smooth, mode="mirror")
        
        fig, ax = plt.subplots(2, 2, layout='constrained', dpi=200)
        pad = 20
        pedge = int(self.edge_idx[-1]) + pad
        self.pedge = pedge
        x1 = self.x[:pedge]
        z0 = self.zb[:pedge]
        z1 = z0 + self.h[-1, :pedge]
        poly1 = ax[0, 0].fill_between(
            x1 / 1000,
            z0,
            z1,
            fc="lightblue",
            ec="lightblue",
            label=f"yr={self.tf} profile",
        )
        ax[0, 0].plot(
            x1 / 1000,
            z0,
            c="black",
            lw=2,
        )
        ax[0, 1].plot(self.t, sm(self.gwb / self.area), label='Sp. MB', c='black')
        ax01b = ax[0, 1].twinx()
        ax01b.plot(
            self.t,
            self.h.max(axis=1),
            c='grey',
        )
        ax[0, 1].plot(  # just for the legend
            [None],
            [None],
            c='grey',
            label=f"Max H",
        )

        ax[1, 0].plot(
            self.t,
            scipy.ndimage.uniform_filter1d(self.T, 30, mode="mirror"),
            c="black",
            lw=1,
            alpha=0.5,
            label=f"T (MA-30)",
        )
        ax10b = ax[1, 0].twinx()
        ax10b.plot(
            self.t,
            sm(self.ela),
            c='blue',
            ls='--',
            lw=1,
            label=f"ELA",
        )
        ax[1, 0].plot(  # just for the legend
            [None],
            [None],
            c='blue',
            ls='--',
            lw=1,
            label=f"ELA",
        )

        ax[1, 1].plot(
            self.t,
            self.edge / 1000,
            c="black",
            lw=2,
            label=f"Length",
        )
        ax11b = ax[1, 1].twinx()
        ax11b.plot(
            self.t,
            self.area / 1e6,
            c='red',
            ls='--',
            lw=1,
        )
        ax[1, 1].plot(  # just for the legend
            [None],
            [None],
            c='red',
            ls='--',
            lw=1,
            label=f"Area",
        )

        ax01b.grid(None)
        ax10b.grid(None)
        ax11b.grid(None)
        for axis in ax.ravel():
            axis.grid(which='both', axis='both', ls=':', c='grey')
            axis.legend()

        return fig, ax

    def to_pickle(self, fp):
        with open(fp, "wb") as f:
            dill.dump(self, f)

        return None

    def to_pandas(self):
        d = dict(
            T=self.T,
            area=self.area,
            bal=self.gwb,
            edge=self.edge_idx,
            edge_m=self.edge,
            ela=self.ela,
        )
        df = pd.DataFrame(d, index=self.t)
        return df

    def to_xarray(self):
        ds = xr.Dataset(
            data_vars=dict(
                T=(["time"], self.T),
                P=(["time"], self.P),
                edge_idx=(["time"], self.edge_idx),
                edge=(["time"], self.edge),
                gwb=(["time"], self.gwb),
                b=(["time", "x"], self.b),
                ela=(["time"], self.ela),
                h=(["time", "x"], self.h),
                area=(["time"], self.area),
                w=(["x"], self.w),
                zb=(["x"], self.zb),
            ),
            coords=dict(
                time=self.t,
                x=self.x,
            ),
            attrs=dict(
                ts=self.ts,
                tf=self.tf,
                nxs=self.nxs,
                delx=self.delx,
                sigT=self.sigT,
                sigP=self.sigP,
                P0=self.P0,
                T0=self.T0,
                Tref=self.Tref,
                Pref=self.Pref,
                nrun=self.nrun,
                ref_period=self.ref_period,
            ),
        )
        return ds

    def _init_plot(self):
        fig = plt.figure(figsize=(18, 10), dpi=100, layout="constrained")
        gs = gridspec.GridSpec(4, 3, figure=fig, height_ratios=(1, 1, 1, 1))
        ax = np.empty((4, 3), dtype="object")
        plt.show(
            block=False
        )  # for live plotting, though maybe block=True would wait for the plot to open before running?

        ax[0, 0] = fig.add_subplot(gs[0, 0])
        ax[0, 0].set_xlabel("Time (years)")
        ax[0, 0].set_ylabel("Glacier Area ($km^2$)")

        ax[0, 1] = fig.add_subplot(gs[0:2, 1])
        ax[0, 1].set_xlabel("Distance (km)")
        ax[0, 1].set_ylabel("Elevation (m)")

        ax[0, 2] = fig.add_subplot(gs[0, 2])
        ax[0, 2].set_xlabel("Mass balance (m)")
        ax[0, 2].set_ylabel("Probability density")

        ax[1, 0] = fig.add_subplot(gs[1, 0])
        ax[1, 0].set_xlabel("Time (years)")
        ax[1, 0].set_ylabel("Equilibrium Line Altitude (m)")

        ax[1, 2] = fig.add_subplot(gs[1, 2])
        ax[1, 2].set_xlabel("Length (km)")
        ax[1, 2].set_ylabel("Probability density")

        ax[2, 0] = fig.add_subplot(gs[2, 0])
        ax[2, 0].set_ylabel("T ($^o$C)")

        ax[2, 1] = fig.add_subplot(gs[2, 1])
        ax[2, 1].set_ylabel("L (km)")

        ax[2, 2] = fig.add_subplot(gs[2, 2])
        ax[2, 2].set_xlabel("L (km)")
        ax[2, 2].set_ylabel("Bal (m $yr^{-1}$)")

        ax[3, 0] = fig.add_subplot(gs[3, 0])
        ax[3, 0].set_ylabel("Bal (m $yr^{-1}$)")
        ax[3, 0].set_xlabel("Time (years)")

        ax[3, 1] = fig.add_subplot(gs[3, 1])
        ax[3, 1].set_xlabel("Time (years)")
        ax[3, 1].set_ylabel("Cum. bal. (m)")

        ax[3, 2] = fig.add_subplot(gs[3, 2])
        ax[3, 2].set_xlabel("Mean thickness (m)")
        ax[3, 2].set_ylabel("Length (km)")

        for axis in ax.ravel():
            if axis is not None:  # this handles gridspec col/rowspans > 1
                axis.grid(axis="both", alpha=0.5)
                axis.set_axisbelow(True)
        plt.tight_layout()

        return fig, ax

    def _rt_plot(self, t, i):
        if (t / self.dt_plot == np.floor(t / self.dt_plot)) | (
            i == self.nts - 1
        ):  # force plotting on the last time step
            print("outputting")
            pad = 10
            x1 = self.x[: self.edge + pad]
            z0 = self.zb[: self.edge + pad]
            z1 = self.zb[: self.edge + pad] + self.h[: self.edge + pad]

            try:
                self.ax[0, 1].collections[0].remove()  # remove the glacier profile before redrawing
            except:
                pass
            poly = self.ax[0, 1].fill_between(x1 / 1000, z0, z1, fc="lightblue")
            self.ax[0, 1].plot(
                x1 / 1000,
                z0,
                c="black",
                lw=2,
            )
            self.ax[0, 0].plot(
                self.t,
                scipy.ndimage.uniform_filter1d(self.area / 1e6, 20, mode="mirror"),
                c="black",
            )
            self.ax[1, 0].plot(
                self.t,
                scipy.ndimage.uniform_filter1d(self.ela, 20, mode="mirror"),
                c="black",
            )
            self.ax[2, 0].plot(self.t, self.t, c="blue", lw=0.25)
            self.ax[2, 1].plot(
                self.t,
                scipy.ndimage.uniform_filter1d(self.edge, 20, mode="mirror") / 1000,
                c="black",
                lw=2,
            )
            self.ax[3, 0].plot(self.t, self.gwb / self.area, c="blue", lw=0.25)
            self.ax[3, 1].plot(
                self.t,
                # scipy.ndimage.uniform_filter1d(
                #     np.cumsum(self.gwb / self.area), 20, mode="mirror"
                # ),
                np.cumsum(self.gwb / self.area) - np.cumsum((self.gwb / self.area).mean()),
                c="blue",
                lw=2,
            )

            # update the plot
            self.fig.canvas.flush_eveself.nts()
            self.fig.canvas.draw()

        if t == self.tf:
            plt.draw()


def calc_ela(P0, T0, gamma, mu, h=None):
    # this seems to be accurate with elev mb feedback??
    if np.asarray(h).any():  # idk if this part is right
        T0 = T0 - h * gamma
    ela = T0 / gamma - P0 / (mu * gamma)
    return ela


def calc_Leq(A, w, bt, db, L=None):
    if np.ndim(w) != 0:
        w = np.mean(w)
    return A / w * -db / bt


def calc_tau(model):
    H = np.array([model.h[i, (model.ela_idx[i]) : (model.edge_idx[i])].mean() for i in range(len(model.ela_idx))])
    bt = np.array([model.b[i, (model.ela_idx[i] - 10) : (model.edge_idx[i])].mean() for i in range(len(model.ela_idx))])
    tau = -H / bt
    return tau, H, bt
