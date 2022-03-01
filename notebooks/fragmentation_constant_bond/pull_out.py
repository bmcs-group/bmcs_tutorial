

import bmcs_utils.api as bu
import numpy as np
import sympy as sp
import traits.api as tr

from po_symb import \
    CB_ELF_ELM_Symb, PO_ELF_RLM_Symb, PO_ESF_RLM_Symb, PO_ELF_ELM_Symb

class PullOutAModel(bu.Model, bu.InjectSymbExpr):
    """
    General pullout elastic long fiber and rigid long matrix
    """
    symb_class = PO_ELF_RLM_Symb

    name = "Pull-Out"

    E_f = bu.Float(210000, MAT=True)
    E_m = bu.Float(28000, MAT=True)
    tau = bu.Float(8, MAT=True)
    A_f = bu.Float(100, CS=True)
    A_m = bu.Float(100*100, CS=True)
    p = bu.Float(20, CS=True)
    L_b = bu.Float(300, GEO=True)
    w_max = bu.Float(3, BC=True)

    t = bu.Float(0.0)
    t_max = bu.Float(1.0)

    ipw_view = bu.View(
        bu.Item('E_f', latex=r'E_\mathrm{f}~[\mathrm{MPa}]'),
        bu.Item('E_m', latex=r'E_\mathrm{m}~[\mathrm{MPa}]'),
        bu.Item('tau', latex=r'\tau~[\mathrm{MPa}]'),
        bu.Item('A_f', latex=r'A_\mathrm{f}~[\mathrm{mm}^2]'),
        bu.Item('A_m', latex=r'A_\mathrm{m}~[\mathrm{mm}^2]'),
        bu.Item('p', latex=r'p~[\mathrm{mm}]'),
        bu.Item('L_b', latex=r'L_\mathrm{b}~[\mathrm{mm}]'),
        bu.Item('w_max', latex=r'w_\max~[\mathrm{mm}]'),
        time_editor = bu.HistoryEditor(
            var='t',
            var_max='t_max'
        )
    )

    w_range = tr.Property(depends_on='state_changed')
    """Pull-out range w"""
    @tr.cached_property
    def _get_w_range(self):
        return np.linspace(0, self.w_max, 100)

    def plot_Pw(self, ax):
        w = self.t * self.w_max
        P = 0.001*self.symb.get_P_w_pull(w)
        w_L_b = self.symb.get_w_L_b(w)
        ax.plot(w,P,marker='o', color='blue')
        ax.plot(w_L_b,P,marker='o', color='blue')

        P_range = self.symb.get_P_w_pull(self.w_range)
        w_L_b_range = self.symb.get_w_L_b(self.w_range)
        ax.plot(self.w_range, P_range * 0.001, color='blue', label=r'$w(0)$')
        ax.plot(w_L_b_range, P_range * 0.001, color='blue', linestyle='dashed',
                label=r'$w(L_\mathrm{b})$')
        ax.set_ylabel(r'$P$ [kN]')
        ax.set_xlabel(r'$w$ [mm]')
        ax.legend()

    def subplots(self, fig):
        gs = fig.add_gridspec(2,2, width_ratios=[1., 1.])
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[1,0])
        ax4 = fig.add_subplot(gs[1,1])
        ax44 = ax4.twinx()
        return ax1, ax2, ax3, ax4, ax44

    def plot_fields(self, ax_u, ax_eps, ax_sig, ax_tau):
        L_b = self.L_b
        x_range = np.linspace(-L_b, 0, 100)
        w_max = self.w_max
        w_range = self.w_range
        P_range = self.symb.get_P_w_pull(self.w_range)
        w_argmax = w_range[np.argmax(P_range)]
        w = self.t * w_max
        eps_f_range = self.symb.get_eps_w_f_x(x_range, w)
        sig_f_range = self.symb.get_sig_w_f_x(x_range, w)
        N_f_range = self.A_f * sig_f_range
        u_f_range = self.symb.get_u_w_f_x(x_range, w)
        eps_m_range = self.symb.get_eps_w_m_x(x_range, w)
        sig_m_range = self.symb.get_sig_w_m_x(x_range, w)
        N_m_range = self.A_m * sig_m_range
        u_m_range = self.symb.get_u_w_m_x(x_range, w)
        tau_range = self.symb.get_tau_w_x(x_range, w) * np.ones_like(x_range)
        T_range = self.p * tau_range

        eps_max = np.max(self.symb.get_eps_w_f_x(x_range, w_argmax))
        sig_max = np.max(self.symb.get_sig_w_f_x(x_range, w_argmax))
        N_max = self.A_f * sig_max
        u_max = w_max
        eps_min = np.min(self.symb.get_eps_w_m_x(x_range, w_argmax))
        sig_min = np.min(self.symb.get_sig_w_m_x(x_range, w_argmax))
        N_min = self.A_m * sig_min
        u_min = np.min(self.symb.get_u_w_m_x(x_range, w_argmax))
        tau_max = self.tau
        T_max = self.p * tau_max
        x_min = -L_b
        x_max = 0


        self.plot_filled_var(
            ax_u, x_range, u_f_range,
            color='brown', alpha=0.2,
            ylim=(u_min, u_max), xlim=(x_min,x_max)
            )

        self.plot_filled_var(
            ax_u, x_range, u_m_range,
            xlabel='$x$ [mm]', ylabel='$u$ [mm]',
            color='black', alpha=0.2,
            ylim=(u_min, u_max), xlim=(x_min,x_max)
            )

        self.plot_filled_var(
            ax_eps, x_range, eps_f_range,
            xlabel='$x$ [mm]', ylabel=r'$\varepsilon$ [mm]', color='green',
            ylim=(eps_min, eps_max), xlim=(x_min,x_max)
            )

        self.plot_filled_var(
            ax_eps, x_range, eps_m_range,
            xlabel='$x$ [mm]', ylabel=r'$\varepsilon$ [mm]', color='green',
            ylim=(eps_min, eps_max), xlim=(x_min,x_max)
            )

        self.plot_filled_var(
            ax_sig, x_range, N_f_range,
            xlabel='$x$ [mm]', ylabel=r'$N$ [N]', color='blue',
            ylim=(sig_min, sig_max), xlim=(x_min,x_max)
            )

        self.plot_filled_var(
            ax_sig, x_range, N_m_range,
            xlabel='$x$ [mm]', ylabel=r'$N$ [N]', color='blue',
            ylim=(N_min, N_max), xlim=(x_min,x_max)
            )

        self.plot_filled_var(
            ax_tau, x_range, T_range,
            xlabel='$x$ [mm]', ylabel=r'$T$ [N/mm]', color='red',
            ylim=(0, T_max), xlim=(x_min,x_max)
            )

    def update_plot(self, axes):
        ax_Pw, ax_u, ax_eps, ax_sig, ax_tau = axes
        self.plot_Pw(ax_Pw)
        self.plot_fields(ax_u, ax_eps, ax_sig, ax_tau)

    def plot_filled_var(self, ax, xdata, ydata, xlabel='', ylabel='',
                        color='black', alpha=0.1, ylim=None, xlim=None):
        line, = ax.plot(xdata, ydata, color=color);
        if xlabel:
            ax.set_xlabel(xlabel);
        if ylabel:
            ax.set_ylabel(ylabel)
        if ylim:
            y_min, y_max = ylim
            dy = y_max - y_min
            ax.set_ylim(y_min-0.05*dy, y_max+0.05*dy)
        if xlim:
            x_min, x_max = xlim
            dx = x_max - x_min
            ax.set_xlim(x_min-0.05*dx, x_max+0.05*dx)
        ax.fill_between(xdata, ydata, 0, color=color, alpha=alpha);
        return line

class PO_ELF_RLM(PullOutAModel):
    name='PO_ELF_RLM'
    symb_class = PO_ELF_RLM_Symb

class PO_ESF_RLM(PullOutAModel):
    name='PO_ESF_RLM'
    symb_class = PO_ESF_RLM_Symb

class PO_ELF_ELM(PullOutAModel):
    name='PO_ELF_ELM'
    symb_class = PO_ELF_ELM_Symb

class CB_ELF_ELM(PullOutAModel):
    name='CB_ELF_ELM'
    symb_class = CB_ELF_ELM_Symb


class PullOutAModelExplorer(bu.Model):
    """fix the update behavior"""
    name = 'Pullout Explorer'
    PO_ELF_RLM = bu.Instance(PullOutAModel, ())
    def _PO_ELF_RLM_default(self):
        return PullOutAModel(symb_class=PO_ELF_RLM_Symb)

    PO_ELF_ELM = bu.Instance(PullOutAModel, ())
    def _PO_ELF_ELM_default(self):
        return PullOutAModel(symb_class=PO_ELF_ELM_Symb)

    PO_ESF_RLM = bu.Instance(PullOutAModel, ())
    def _PO_ESF_RLM_default(self):
        return PullOutAModel(symb_class=PO_ESF_RLM_Symb)

    CB_ECF_ECM = bu.Instance(PullOutAModel, ())
    def _CB_ECF_ECM_default(self):
        return PullOutAModel(symb_class=CB_ELF_ELM_Symb)

    tree = ['PO_ELF_RLM', 'PO_ELF_ELM', 'PO_ESF_RLM', 'CB_ECF_ECM']
