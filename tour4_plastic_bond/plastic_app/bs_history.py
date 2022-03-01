import bmcs_utils.api as bu
import traits.api as tr
from bmcs_utils.api import mpl_align_yaxis
import numpy as np
from scipy.integrate import cumtrapz

class BSHistory(bu.Model):
    name = 'State evolution'

    slider_exp = tr.WeakRef(bu.Model)

    t_slider = bu.Float(0)
    t_max = bu.Float(1.001)

    t_arr = tr.DelegatesTo('slider_exp')
    Sig_arr = tr.DelegatesTo('slider_exp')
    Eps_arr = tr.DelegatesTo('slider_exp')
    s_t = tr.DelegatesTo('slider_exp')

    ipw_view = bu.View(
        bu.Item('t_max', latex=r't_{\max}', readonly=True),
        time_editor=bu.HistoryEditor(var='t_slider', low=0, high_name='t_max', n_steps=50)
    )

    def plot_Sig_Eps(self, axes):
        ax1, ax11, ax33, ax3 = axes
        colors = ['blue', 'red', 'green', 'black', 'magenta']
        t = self.t_arr
        s_pi_, z_, alpha_ = self.Eps_arr.T
        tau_pi_, Z_, X_ = self.Sig_arr.T
        n_step = len(s_pi_)

        idx = np.argmax(self.t_slider < self.t_arr)

        s_t = self.s_t
        s_pi_t = s_pi_
        tau_pi_t = tau_pi_

        ax1.set_title('stress - displacement');
        ax1.plot(t, tau_pi_t, '--', color='darkgreen', label=r'$\tau$')
        ax1.fill_between(t, tau_pi_t, 0, color='limegreen', alpha=0.1)
        ax1.set_ylabel(r'$ \tau')
        ax1.set_xlabel('$t$');
        ax1.plot(t[idx], 0, marker='H', color='red')
        ax1.legend()
        ax11.plot(t, s_t, color='darkgreen', label=r'$s$')
        ax11.plot(t, s_pi_t, color='orange', label=r'$s_\mathrm{pl}$')
        ax11.set_ylabel(r'$s$')
        ax11.legend()
        mpl_align_yaxis(ax1,0,ax11,0)

        ax3.set_title('hardening force - displacement');
        alpha_t = alpha_
        X_t = X_
        ax3.plot(t, Z_, '--', color='darkcyan', label=r'$Z$')
        ax3.fill_between(t, Z_, 0, color='darkcyan', alpha=0.05)
        ax3.plot(t, X_t, '--', color='darkslateblue', label=r'$X$')
        ax3.fill_between(t, X_t, 0, color='darkslateblue', alpha=0.05)
        ax3.set_ylabel(r'$Z, X$')
        ax3.set_xlabel('$t$');
        ax3.plot(t[idx], 0, marker='H', color='red')
        ax3.legend()
        ax33.plot(t, z_, color='darkcyan', label=r'$z$')
        ax33.plot(t, alpha_t, color='darkslateblue', label=r'$\alpha$')
        ax33.set_ylabel(r'$z, \alpha$')
        ax33.legend(loc='lower left')

    def subplots(self, fig):
        ax1, ax3 = fig.subplots(1, 2)
        ax11 = ax1.twinx()
        ax33 = ax3.twinx()
        return ax1, ax11, ax3, ax33

    def update_plot(self, axes):
        self.plot_Sig_Eps(axes)
