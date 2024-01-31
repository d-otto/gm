# -*- coding: utf-8 -*-
"""
test_gm3s.py

Description.

Author: drotto
Created: 1/29/24 @ 10:23
Project: gm
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from gm import gm3s


#%%

def calc_feq(t, tau):
    # Define the analytical solution for feq in response to a constant warming trend
    # (Christain et al., 2018)
    eps = 1 / np.sqrt(3)
    feq = (1
           - (3 * eps * tau) / t * (1 - np.exp(-t / (eps * tau)))
           + np.exp(-t / (eps * tau)) * (t / (2 * eps * tau) + 2))
    return feq

#%%

class TestFeq:

    def calc_feq_error(self, tau):
        t = np.arange(0, 200)
        melt_factor = -0.65
        b_p = np.linspace(0, -1, len(t))  # constant warming trend with final value of -0.65 m/yr
        b_p = b_p * melt_factor

        params_3s = dict(
            dt=0.01,
            Atot=10,
            L=10,
            H=100,
            bt=0,
            b_p=b_p,
            ts=t,
        )
        res = gm3s(tau=tau, **params_3s).run()
        res.Lp = res.Lp
        res.feq = res.Lp / res.Lp_eq
        res.feq = res.feq[::100]  # resample from 20k to 200
        res.feq_analytic = calc_feq(t, tau)  # compare w/ analytical result
        error = (res.feq_analytic - res.feq)[1:]  # first value is nan

        return error
    
    def test_one(self):
        tau = 10
        error = self.calc_feq_error(tau)
        assert np.max(np.abs(error)) < 0.0003 
    
    def test_two(self):
        tau = 25
        error = self.calc_feq_error(tau)
        assert np.max(np.abs(error)) < 0.0001

    def test_three(self):
        tau = 75
        error = self.calc_feq_error(tau)
        assert np.max(np.abs(error)) < 0.00005


class TestStepResponse:
    # test the response to a step change in climate
    def calc_response(self, tau):
        
        dt=0.1
        t = np.arange(0, 5*tau)
        melt_factor = -0.65
        b_p = np.full(len(t), fill_value=-2)  # step change
        b_p = b_p * melt_factor

        params_3s = dict(
            dt=dt,
            Atot=50,
            L=20,
            H=200,
            bt=0,
            b_p=b_p,
            ts=t,
        )
        res = gm3s(tau=tau, **params_3s).run()
        feq = res.Lp / res.Lp_eq
        # resample to t/tau
        # starting with 5*tau*dt steps, resample to a clean 500 steps
        feq = np.interp(np.linspace(0, 5*tau, 500), res.ts, feq)
        
        return feq
    
    def test_one(self):
        # See Fig. 7, Roe and Baker (2014)
        feq = self.calc_response(20)
        assert feq[100] - 0.25 < 0.01 
        
    def test_two(self):
        feq = self.calc_response(40)
        assert feq[100] - 0.25 < 0.01
        
    def test_three(self):
        # H = 200, so bt = -2.5 m/yr
        feq = self.calc_response(80)
        assert feq[100] - 0.25 < 0.01