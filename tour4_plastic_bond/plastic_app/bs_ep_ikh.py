#!/usr/bin/env python

import traits.api as tr
import numpy as np
import bmcs_utils.api as bu

class ConvergenceError(Exception):
    """ Inappropriate argument value (of correct type). """

    def __init__(self, message, state):  # real signature unknown
        self.message = message
        self.state = state
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message} for state {self.state}'

class BS_EP_IKH(bu.Model):

    name = 'elastic-platic'

    E = bu.Float(28000, MAT=True)
    gamma = bu.Float(10, MAT=True)
    K = bu.Float(8, MAT=True)
    tau_bar = bu.Float(9, MAT=True)

    ipw_view = bu.View(
        bu.Item('E', latex='E', minmax=(0.5, 100)),
        bu.Item('gamma', latex=r'\gamma_\mathrm{T}', minmax=(-20, 20)),
        bu.Item('K', minmax=(-20, 20)),
        bu.Item('tau_bar', latex=r'\bar{\tau}', minmax=(0.5, 20)),
    )

    def get_sig_n1(self, s_n1, Sig_n, Eps_n, k_max):
        '''Return mapping iteration:
        This function represents a user subroutine in a finite element
        code or in a lattice model. The input is $s_{n+1}$ and the state variables
        representing the state in the previous solved step $\boldsymbol{\mathcal{E}}_n$.
        The procedure returns the stresses and state variables of
        $\boldsymbol{\mathcal{S}}_{n+1}$ and $\boldsymbol{\mathcal{E}}_{n+1}$
        '''
        Eps_k = np.copy(Eps_n)
        _E_b = self.E
        _K = self.K
        _gamma = self.gamma
        _tau_Y = self.tau_bar  # material parameters
        s_pl_k, z_k, alpha_k = Eps_k  # initialization of trial states
        tau_k = _E_b * (s_n1 - s_pl_k)  # elastic trial step
        Z_k = _K * z_k  # isotropic hardening
        X_k = _gamma * alpha_k  # isotropic hardening
        f_k = np.abs(tau_k - X_k) - Z_k - _tau_Y

        if f_k > 0:  # inelastic step - return mapping
            delta_lambda_k = f_k / (_E_b + _K + _gamma)
            s_pl_k += delta_lambda_k * np.sign(tau_k - X_k)
            z_k += delta_lambda_k  # to save lines n=n+1 is shortend to k
            alpha_k += delta_lambda_k * np.sign(tau_k - X_k)  # to save lines n=n+1 is shortend to k
            tau_k = _E_b * (s_n1 - s_pl_k)

        Eps_k = np.array([s_pl_k, z_k, alpha_k])
        Sig_k = np.array([tau_k, _K*z_k, _gamma*alpha_k])
        return Eps_k, Sig_k, 1

    Eps_names = tr.Property
    @tr.cached_property
    def _get_Eps_names(self):
        return [eps.codename for eps in self.symb.Eps]

    Sig_names = tr.Property
    @tr.cached_property
    def _get_Sig_names(self):
        return [sig.codename for sig in self.symb.Sig]

    state_var_shapes = tr.Property
    @tr.cached_property
    def _get_state_var_shapes(self):
        '''State variables shapes:
        variables are using the codename string in the Cymbol definition
        Since the same string is used in the lambdify method via print_Symbol
        method defined in Cymbol as well'''
        return {eps_name: () for eps_name in self.Eps_names + self.Sig_names}


    def plot_f_state(self, ax, Eps, Sig):
        lower = -self.f_c * 1.05
        upper = self.f_t + 0.05 * self.f_c
        lower_tau = -self.tau_bar * 2
        upper_tau = self.tau_bar * 2
        lower_tau = 0
        upper_tau = 10
        tau_x, tau_y, sig = Sig[:3]
        tau = np.sqrt(tau_x**2 + tau_y**2)
        sig_ts, tau_x_ts  = np.mgrid[lower:upper:201j,lower_tau:upper_tau:201j]
        Sig_ts = np.zeros((len(self.symb.Eps),) + tau_x_ts.shape)
        Eps_ts = np.zeros_like(Sig_ts)
        Sig_ts[0,...] = tau_x_ts
        Sig_ts[2,...] = sig_ts
        Sig_ts[3:,...] = Sig[3:,np.newaxis,np.newaxis]
        Eps_ts[...] = Eps[:,np.newaxis,np.newaxis]
        H_sig_pi = self.symb.get_H_sig_pi_(Sig_ts)
        f_ts = np.array([self.symb.get_f_(Eps_ts, Sig_ts, H_sig_pi)])

        #phi_ts = np.array([self.symb.get_phi_(Eps_ts, Sig_ts)])
        ax.set_title('threshold function');

        omega_N = Eps_ts[-1,:]
        omega_T = Eps_ts[-2,:]
        sig_ts_eff = sig_ts / (1 - H_sig_pi*omega_N)
        tau_x_ts_eff = tau_x_ts / (1 - omega_T)
        ax.contour(sig_ts_eff, tau_x_ts_eff, f_ts[0,...], levels=0, colors=('green',))

        ax.contour(sig_ts, tau_x_ts, f_ts[0,...], levels=0, colors=('red',))
        #ax.contour(sig_ts, tau_x_ts, phi_ts[0, ...])
        ax.plot(sig, tau, marker='H', color='red')
        ax.plot([lower, upper], [0, 0], color='black', lw=0.4)
        ax.plot([0, 0], [lower_tau, upper_tau], color='black', lw=0.4)
        ax.set_ylim(ymin=0, ymax=10)

    def plot_f(self, ax):
        lower = -self.f_c * 1.05
        upper = self.f_t + 0.05 * self.f_c
        lower_tau = -self.tau_bar * 2
        upper_tau = self.tau_bar * 2
        sig_ts, tau_x_ts  = np.mgrid[lower:upper:201j,lower_tau:upper_tau:201j]
        Sig_ts = np.zeros((len(self.symb.Eps),) + tau_x_ts.shape)
        Sig_ts[0,:] = tau_x_ts
        Sig_ts[2,:] = sig_ts
        Eps_ts = np.zeros_like(Sig_ts)
        H_sig_pi = self.symb.get_H_sig_pi_(Sig_ts)
        f_ts = np.array([self.symb.get_f_(Eps_ts, Sig_ts, H_sig_pi)])
        phi_ts = np.array([self.get_phi_(Eps_ts, Sig_ts, H_sig_pi)])
        ax.set_title('threshold function');
        ax.contour(sig_ts, tau_x_ts, f_ts[0,...], levels=0)
        ax.contour(sig_ts, tau_x_ts, phi_ts[0, ...])
        ax.plot([lower, upper], [0, 0], color='black', lw=0.4)
        ax.plot([0, 0], [lower_tau, upper_tau], color='black', lw=0.4)

    def update_plot(self, ax):
        return
        self.plot_f(ax)
