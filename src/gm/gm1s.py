# -*- coding: utf-8 -*-
"""
gm1s.py

Description.

Author: drotto
Created: 1/31/24 @ 09:43
Project: gm
"""

import numpy as np
import collections

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
        bp=0,
        dbdt=None,
        mode=None,
    ):
        
        self.btp = bt + bp
        self.bt_eq = bt
        self.bp = bp

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


    def linear(self):
        self.tau = -self.H / self.btp
        self.Lp_eq = self.tau * self.beta * self.btp * (self.ts - self.tau)  # todo: what is this last term?
        self.Lp = np.zeros_like(self.ts, dtype="float")

        # Christian et al eq. 4
        if self.mode == "linear":
            for i, t in enumerate(self.ts):
                if self.btp[i] == 0:
                    self.Lp[i] = 0
                    continue

                self.Lp[i] = (
                        self.tau[i] * self.beta * self.btp[i] * (t - self.tau[i] * (1 - np.exp(-t / self.tau[i])))
                )

        self.L = self.L_bar + np.cumsum(self.Lp)

    def run(self):
        self.Lp = np.empty_like(self.ts, dtype="float")

        for i, t in enumerate(self.ts):
            # Roe and Baker (2014) eq. 8
            if i == 0:
                self.Lp[i] = self.beta * t * self.bp[i]
                continue
            self.Lp[i] = (1 - self.dt[i] / self.tau) * self.Lp[i - 1] + self.beta * self.dt[i] * self.bp[i]
        self.L = self.L_bar + self.Lp
        self.Lp_eq = self.tau * self.beta * self.btp
        self.dL = abs(self.Lp_eq[-1])
        
        return self

    def to_xarray(self):
        import xarray as xr

        ds = xr.Dataset(
            data_vars=dict(
                bp=("t", self.bp),
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