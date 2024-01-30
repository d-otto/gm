# -*- coding: utf-8 -*-
"""
gm3s.py

Description.

Author: drotto
Created: 1/30/24 @ 09:56
Project: gm
"""

import pandas as pd
import numpy as np
import collections
import numba as nb

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
        ts,  # in units of years
        dt=1,  # in units of years
        H=None,
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
        sigb=1,
        P0=None,
        T0=None,
        T_p=None,
        P_p=None,
        ATgt0=None,
        zb=None,
        beta=None,
    ):
        #todo: is there a dt everywhere there should be a dt?


        self.dt = dt
        self.ts = np.linspace(ts[0], ts[-1], round(len(ts)/dt))
        self.years = np.arange(ts[0], ts[-1] + 1)

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
            if b_p is None:
                b_p = P_p - T_p * self.alpha

            self.ATgt0 = ATgt0
            self.sigT = sigT
            self.sigP = sigP
            self.sigb = sigb
            self.P0 = P0
            self.Aabl = Aabl
            self.gamma = gamma
            self.mu = mu
        elif mode == "b":
            self.sigb = sigb
            if isinstance(b_p, (collections.abc.Sequence, np.ndarray)):
                b_p = b_p * sigb
            elif isinstance(b_p, (int, float)):
                # step change
                # todo: implement sol'n for step change
                b_p = np.full_like(ts, fill_value=b_p)
            elif b_p is None:
                b_p = np.zeros_like(ts)

            b_p = np.interp(self.ts, self.years, b_p)

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
        if beta is None:
            self.beta = self.L_bar / self.H
        else:
            self.beta = beta
        self.T_p = T_p
        self.P_p = P_p

    def copy(self):
        return copy.deepcopy(self)

    def to_pandas(self):
        df = pd.DataFrame(
            data=dict(L_p=self.L_p, L_eq=self.L_eq, dL=self.dL, L=self.L, bt_p=self.bt_p, L_bar=self.L_bar),
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
        dt = self.dt
        ts = self.ts

        L_p = np.zeros_like(self.ts, dtype="float")

        # L3s(i) = 3 * phi * L3s(i - 1) -
        # 3 * phi ^ 2 * L3s(i - 2)
        # + 1 * phi ^ 3 * L3s(i - 3)...
        # + dt ^ 3 * tau / (eps * tau) ^ 3 * (beta * bp(i - 3))

        if self.mode == "b":
            bt_p = self.bt_p
            L_p = calc_L_p_for_b(ts, L_p, K, dt, tau, eps, beta, bt_p)

        elif self.mode == "l":
            for i, t in enumerate(self.ts):
                if i <= 3:
                    continue
                L_p[i] = (
                    3 * K * L_p[i - 1]
                    - 3 * K ** 2 * L_p[i - 2]
                    + 1 * K ** 3 * L_p[i - 3]
                    + self.dt[i] ** 3
                    * tau
                    / (eps * tau) ** 3
                    * (beta[i] * self.P_p[i - 3] - self.alpha[i] * self.T_p[i - 3])
                )

        self.L_p = L_p
        self.L = self.L_bar + self.L_p
        self.L_eq = self.tau * self.beta * self.bt_p + self.L_bar
        self.dL = abs(self.L[0] - self.L_eq)

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

    @property
    def sigL_3s(self):
        # if self.mode=='l':
        #     self.sigL_1s = np.sqrt(self.tau * self.dt / 2 * (self.alpha**2 * self.sigT**2 + self.beta**2 * self.sigP**2))
        # elif self.mode=='b':
        #     self.sigL_1s = self.tau * self.dt / 2 * (self.beta**2 * self.sigb**2)
        # self.P0 = 4 * self.tau * self.sigL_1s**2  # power spectrum in the limit f -> 0 using the variance from the 1s model
        # sigL_3s = np.sqrt((self.P0 * (1 - self.K) * (1 + 4 * self.K**2 + self.K**4))/(2 * self.dt * (1 + self.K)**5))

        sigL_3s = ((3 * self.tau * np.mean(self.dt)) / (16 * self.eps)) ** (1 / 2) * np.mean(self.beta) * self.sigb
        return sigL_3s

    def acf(self, t):
        """Based on the continuous form of the 3-stage equations"""
        eps = self.eps
        tau = self.tau

        acf = np.exp(-t / (eps * tau)) * (1 + t / (eps * tau) + 1 / 3 * (t / (eps * tau)) ** 2)
        return acf



@nb.njit()
def calc_L_p_for_b(ts, L_p, K, dt, tau, eps, beta, bt_p):
    for i, t in enumerate(ts):
        if i < 3:
            continue
        L_p[i] = (
            3 * K * L_p[i - 1]
            - 3 * K ** 2 * L_p[i - 2]
            + 1 * K ** 3 * L_p[i - 3]
            #+ dt ** 3. * tau / (eps * tau) ** 3. * (beta * bt_p[i - 3])
            + dt * beta / eps * (dt/(eps*tau))**2 * bt_p[i-3]
        )
    return L_p