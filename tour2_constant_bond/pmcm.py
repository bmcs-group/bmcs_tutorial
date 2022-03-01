# %% md

import numpy as np
from scipy.optimize import newton
import traits.api as tr
import bmcs_utils.api as bu
from ipw_editors.int_range_editor import IntRangeEditor

from pull_out import CB_ELF_ELM

# %%
import warnings  # (*\label{error1}*)

warnings.filterwarnings("error", category=RuntimeWarning)  # (*\label{error2}*)

class ICrackBridgeModel(tr.Interface):
    """
    Record of all material parameters of the composite. The model components
    (PullOutModel, CrackBridgeRespSurf, PMCM) are all linked to the database record
    and access the parameters they require. Some parameters are shared across all
    three components (E_m, Ef, vf), some are specific to a particular type of the
    PulloutModel.
    """
    v_f = bu.Float
    E_c = bu.Float

@tr.provides(ICrackBridgeModel)
class CBMConstantBond2(CB_ELF_ELM):

    v_f = tr.Property(bu.Float, depends_on='state_changed')
    @tr.cached_property
    def _get_v_f(self):
        return self.A_f / (self.A_f + self.A_m)

    E_c = tr.Property(bu.Float, depends_on='state_changed')
    @tr.cached_property
    def _get_E_c(self):
        return self.E_m * (1 - self.v_f) + self.E_f * self.v_f  # [MPa] mixture rule

    ipw_view = bu.View(
        *CB_ELF_ELM.ipw_view.content,
        bu.Item('E_c', latex=r'E_\mathrm{c}~\mathrm{[MPa]}', readonly=True),
        bu.Item('v_f', latex=r'V_\mathrm{f}~\mathrm{[-]}', readonly=True),
        time_editor=CB_ELF_ELM.ipw_view.time_editor
    )

    ## Crack bridge with constant bond
    def get_sig_m(self, z, sig_c):  # matrix stress (*\label{sig_m}*)
        return self.symb.get_sig_P_m_x(z, sig_c * self.A_m)

    def get_eps_f(self, z, sig_c):  # reinforcE_ment strain (*\label{sig_f}*)
        return self.symb.get_eps_P_f_x(z, sig_c * self.A_m)

@tr.provides(ICrackBridgeModel)
class CBMConstantBond(bu.Model):
    """
    Return the matrix stress profile of a crack bridge for a given control slip
    at the loaded end
    """
    E_m = bu.Float(25e3, MAT=True)  # [MPa] matrix modulus
    E_f = bu.Float(180e3, MAT=True)  # [MPa] fiber modulus
    A_c = bu.Float(1000, CS=True)
    A_f = bu.Float(1, CS=True)
    p = bu.Float(3.14, CS=True)
    tau = bu.Float(8.00, MAT=True)

    T = tr.Property(bu.Float, depends_on='state_changed')
    @tr.cached_property
    def _get_T(self):
        return self.p * self.tau / self.A_f

    v_f = tr.Property(bu.Float, depends_on='state_changed')
    @tr.cached_property
    def _get_v_f(self):
        return self.A_f / self.A_c

    A_m = tr.Property(bu.Float, depends_on='state_changed')
    @tr.cached_property
    def _get_A_m(self):
        return self.A_c - self.A_f

    E_c = tr.Property(bu.Float, depends_on='state_changed')
    @tr.cached_property
    def _get_E_c(self):
        return self.E_m * (1 - self.v_f) + self.E_f * self.v_f  # [MPa] mixture rule

    ## Crack bridge with constant bond
    def get_sig_m(self, z, sig_c):  # matrix stress (*\label{sig_m}*)
        T = self.T
        v_f = self.v_f
        E_m = self.E_m
        E_f = self.E_f
        sig_m = np.minimum(z * T * v_f / (1 - v_f), E_m * sig_c / (v_f * E_f + (1 - v_f) * E_m))
        return sig_m

    def get_eps_f(self, z, sig_c):  # reinforcE_ment strain (*\label{sig_f}*)
        v_f = self.v_f
        E_f = self.E_f
        sig_m = self.get_sig_m(z, sig_c)
        eps_f = (sig_c - sig_m * (1 - v_f)) / v_f / E_f
        return eps_f

    sig_c_slider: float = bu.Float(1.0, BC=True)

    ipw_view = bu.View(
        bu.Item('E_m'),
        bu.Item('E_f'),
        bu.Item('E_c', readonly=True),
        bu.Item('tau'),
        bu.Item('A_c'),
        bu.Item('A_f'),
        bu.Item('A_m', readonly=True),
        bu.Item('p'),
        bu.Item('v_f', readonly=True),
        bu.Item('T', readonly=True),
        bu.Item('sig_c_slider', editor=bu.FloatRangeEditor(low=0,high=10))
    )

    @staticmethod
    def subplots(fig):
        ax = fig.subplots(1,1)
        ax1 = ax.twinx()
        return ax, ax1

    def update_plot(self, axes):
        ax, ax1 = axes
        x_range = np.linspace(-100,100,1000)
        z_range = np.abs(x_range)

        sig_m_range = self.get_sig_m(z_range, self.sig_c_slider)
        eps_f_range = self.get_eps_f(z_range, self.sig_c_slider)
        sig_max = 10
        eps_max = 1000 / 100000

        ax.plot(x_range, sig_m_range, color='black')
        ax.fill_between(x_range, sig_m_range, color='gray', alpha=0.1)
        ax.set_ylim(ymin=-0.03*sig_max,ymax=sig_max)
        ax.set_xlabel(r'$z$ [mm]')
        ax.set_ylabel(r'$\sigma_\mathrm{m}$ [MPa]')
        ax1.plot(x_range, eps_f_range, color='blue')
        ax1.fill_between(x_range, eps_f_range, color='blue', alpha=0.1)
        ax1.set_ylim(ymin=-0.03*eps_max,ymax=eps_max)
        ax1.set_ylabel(r'$\varepsilon_\mathrm{m}$ [-]')

class CrackBridgeRespSurface(bu.Model):

    cb = bu.Instance(ICrackBridgeModel)

    def get_sig_m(self, z, sig_c):  # matrix stress (*\label{sig_m}*)
        return self.cb.get_sig_m(z, sig_c)

    def get_eps_f(self, z, sig_c):  # reinforcE_ment strain (*\label{sig_f}*)
        return self.cb.get_eps_f(z, sig_c)


class PMCM(bu.Model):
    name='Fragmentation'
    sig_cu = bu.Float(10.0, MAT=True)  # [MPa] composite strength
    sig_mu = bu.Float(3.0, MAT=True)  # [MPa] matrix strength
    m = bu.Float(4, MAT=True)  # Weibull shape modulus
    ## Crack tracing algorithm
    n_x = bu.Int(200, MAT=True)
    L_x = bu.Float(380, MAT=True)

    crack_bridge = bu.Instance(ICrackBridgeModel)
    def _crack_bridge_default(self):
        return CBMConstantBond()

    # cb_rs = tr.Property
    # @tr.cached_property
    # def _get_cb_rs(self):
    #     return CrackBridgeRespSurface(cb=self.cb)

    K = bu.Int(0)
    K_max = tr.Property(bu.Int, depends_on='state_changed')
    @tr.cached_property
    def _get_K_max(self):
        sig_c_K, eps_c_K, sig_mu_x, x, CS, sig_m_x_K, eps_f_x_K = self.cracking_history
        K_max = len(sig_c_K)-2
        # ceil the index of current crack
        self.K = np.min([self.K, K_max])
        return K_max

    tree = ['crack_bridge']

    ipw_view = bu.View(
        bu.Item('n_x'),
        bu.Item('L_x'),
        bu.Item('sig_cu'),
        bu.Item('sig_mu'),
        bu.Item('m'),
        bu.Item('K', latex=r'i_\mathrm{crack}', editor=IntRangeEditor(high_name='K_max')),
        bu.Item('K_max', latex=r'N_\mathrm{cracks}', readonly=True)
    )

    ## Specimen discretization
    def get_z_x(self, x, XK):  # distance to the closest crack (*\label{get_z_x}*)
        z_grid = np.abs(x[:, np.newaxis] - np.array(XK)[np.newaxis, :])
        return np.amin(z_grid, axis=1)

    def get_sig_c_z(self, sig_mu, z, sig_c_pre):
        # crack initiating load at a material element
        fun = lambda sig_c: sig_mu - self.crack_bridge.get_sig_m(z, sig_c)
        try:  # search for the local crack load level
            return newton(fun, sig_c_pre)
        except (RuntimeWarning, RuntimeError):
            # solution not found (shielded zone) return the ultimate composite strength
            return self.sig_cu

    def get_sig_c_K(self, z_x, x, sig_c_pre, sig_mu_x):
        # crack initiating loads over the whole specimen
        get_sig_c_x = np.vectorize(self.get_sig_c_z)
        sig_c_x = get_sig_c_x(sig_mu_x, z_x, sig_c_pre)
        y_idx = np.argmin(sig_c_x)
        return sig_c_x[y_idx], x[y_idx]

    cracking_history = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_cracking_history(self):
        return self.get_cracking_history()

    def get_cracking_history(self, update_progress=None):
        L_x, n_x = self.L_x, self.n_x
        sig_mu = self.sig_mu
        m = self.m
        x = np.linspace(0, L_x, n_x)  # specimen discretization (*\label{discrete}*)
        sig_mu_x = sig_mu * np.random.weibull(m, size=n_x)  # matrix strength (*\label{m_strength}*)

        E_m = self.crack_bridge.E_m
        E_c = self.crack_bridge.E_c

        XK = []  # recording the crack postions
        sig_c_K = [0.]  # recording the crack initating loads
        eps_c_K = [0.]  # recording the composite strains
        CS = [L_x, L_x / 2]  # crack spacing
        sig_m_x_K = [np.zeros_like(x)]  # stress profiles for crack states
        eps_f_x_K = [np.zeros_like(x)]  # stress profiles for crack states

        idx_0 = np.argmin(sig_mu_x)
        XK.append(x[idx_0])  # position of the first crack
        sig_c_0 = sig_mu_x[idx_0] * E_c / E_m
        sig_c_K.append(sig_c_0)
        eps_c_K.append(sig_mu_x[idx_0] / E_m)

        while True:
            z_x = self.get_z_x(x, XK)  # distances to the nearest crack
            sig_m_x_K.append(self.crack_bridge.get_sig_m(z_x, sig_c_K[-1]))  # matrix stress
            eps_f_x_K.append(self.crack_bridge.get_eps_f(z_x, sig_c_K[-1]))
            sig_c_k, y_i = self.get_sig_c_K(z_x, x, sig_c_K[-1], sig_mu_x)  # identify next crack
            if sig_c_k == self.sig_cu:  # (*\label{no_crack}*)
                break
            if update_progress:  # callback to user interface
                update_progress(sig_c_k)
            XK.append(y_i)  # record crack position
            sig_c_K.append(sig_c_k)  # corresponding composite stress
            eps_c_K.append(  # composite strain - integrate the strain field
                np.trapz(self.crack_bridge.get_eps_f(self.get_z_x(x, XK), sig_c_k), x) / np.amax(x))  # (*\label{imple_avg_strain}*)
            XK_arr = np.hstack([[0], np.sort(np.array(XK)), [L_x]])
            CS.append(np.average(XK_arr[1:] - XK_arr[:-1]))  # crack spacing

        sig_c_K.append(self.sig_cu)  # the ultimate state
        eps_c_K.append(np.trapz(self.crack_bridge.get_eps_f(self.get_z_x(x, XK), self.sig_cu), x) / np.amax(x))
        CS.append(CS[-1])
        if update_progress:
            update_progress(sig_c_k)
        return (
            np.array(sig_c_K), np.array(eps_c_K), sig_mu_x, x, np.array(CS),
            np.array(sig_m_x_K), np.array(eps_f_x_K)
        )

    @staticmethod
    def subplots(fig):
        ax1, ax2 = fig.subplots(1,2)
        ax11 = ax1.twinx()
        ax22 = ax2.twinx()
        return ax1, ax11, ax2, ax22

    def plot(self, axes):
        ax, ax_cs, ax_sig_x, ax_eps_x = axes
        E_cf = self.crack_bridge.v_f * self.crack_bridge.E_f
        sig_c_K, eps_c_K, sig_mu_x, x, CS, sig_m_x_K, eps_f_x_K = self.cracking_history
        eps_c_max = eps_c_K[-1]
        #sig_c_K, eps_c_K, sig_mu_x, x, CS, sig_m_x_K, sig_m_x_K1 = self.cracking_history
        n_c = len(eps_c_K) - 2  # numer of cracks
        ax.plot(eps_c_K, sig_c_K, marker='o', label='%d cracks:' % n_c)
        ax.plot([0, eps_c_max], [0, E_cf * eps_c_max], color='black', linewidth=1, linestyle='dashed');
        ax.set_xlabel(r'$\varepsilon_\mathrm{c}$ [-]');
        ax.set_ylabel(r'$\sigma_\mathrm{c}$ [MPa]')
        ax_sig_x.plot(x, sig_mu_x, color='orange', linewidth=1)
        ax_sig_x.fill_between(x, sig_mu_x, 0, color='orange', alpha=0.1)
        ax_sig_x.set_xlabel(r'$x$ [mm]');
        ax_sig_x.set_ylabel(r'$\sigma$ [MPa]')
        ax.legend()
        eps_c_KK = np.array([eps_c_K[:-1], eps_c_K[1:]]).T.flatten()
        CS_KK = np.array([CS[:-1], CS[:-1]]).T.flatten()
        ax_cs.plot(eps_c_KK, CS_KK, color='gray')
        ax_cs.fill_between(eps_c_KK, CS_KK, color='gray', alpha=0.2)
        ax_cs.set_ylabel(r'$\ell_\mathrm{cs}$ [mm]');

        cr = self.K
        ax_sig_x.plot(x,sig_m_x_K[cr])
#        ax_sig_x.plot(x,sig_m_x_K1[cr], linestyle='dashed')
        ax.plot(eps_c_K[cr],sig_c_K[cr],color='magenta',marker='o')
        ax_eps_x.plot(x, eps_f_x_K[cr], color='gray', linewidth=1)
        ax_eps_x.set_ylabel(r'$\varepsilon_\mathrm{f}$')

    def update_plot(self, axes):
        self.plot(axes)
