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
        self.L_eq = self.tau * self.beta * self.bt_p * (self.ts - self.tau)  # todo: what is this last term?
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