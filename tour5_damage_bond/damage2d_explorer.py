
import numpy as np
import sympy as sp
import bmcs_utils.api as bu
from bmcs_cross_section.pullout import MATS1D5BondSlipD

s_x, s_y = sp.symbols('s_x, s_y')
kappa_ = sp.sqrt( s_x**2 + s_y**2 )
get_kappa = sp.lambdify( (s_x, s_y), kappa_, 'numpy' )

def get_tau_s(s_x_n1, s_y_n1, Eps_n, bs, **kw):
    '''Get the stress for the slip in x, y dirctions given the state kappa_n'''
    _, _, kappa_n = Eps_n
    kappa = get_kappa(s_x_n1, s_y_n1)
    # adapt the shape of the state array
    kappa_n_ = np.broadcast_to(kappa_n, kappa.shape)
    kappa_n1 = np.max(np.array([kappa_n_, kappa], dtype=np.float_),axis=0)
    E_b = bs.E_b
    omega_n1 = bs.omega_fn_(kappa_n1)
    tau_x_n1 = (1 - omega_n1) * E_b * s_x_n1
    tau_y_n1 = (1 - omega_n1) * E_b * s_y_n1
    return (
        np.array([s_x_n1, s_y_n1, kappa_n1]),
        np.array([tau_x_n1, tau_y_n1, omega_n1])
    )

def plot_tau_s(ax, Eps_n, s_min, s_max, n_s, bs, **kw):
    n_s_i = complex(0,n_s)
    s_x_n1, s_y_n1 = np.mgrid[s_min:s_max:n_s_i, s_min:s_max:n_s_i]
    Eps_n1, Sig_n1 = get_tau_s(s_x_n1, s_y_n1, Eps_n, bs, **kw)
    s_x_n1, s_y_n1, _ = Eps_n1
    tau_x_n1, tau_y_n1, _ = Sig_n1
    tau_n1 = np.sqrt(tau_x_n1**2 + tau_y_n1**2)
    ax.plot_surface(s_x_n1, s_y_n1, tau_n1, alpha=0.2)
    phi=np.linspace(0,2*np.pi,100)
    _, _, kappa_n = Eps_n
    kappa_0 = bs.omega_fn_.kappa_0
    E_b = bs.E_b
    r = max(kappa_0, kappa_n)
    omega_n = bs.omega_fn_(r)
    f_t = (1-omega_n)*E_b*r
    s0_x, s0_y = r*np.sin(phi), r*np.cos(phi)
    ax.plot(s0_x, s0_y, 0, color='gray')
    ax.plot(s0_x, s0_y, f_t, color='gray')
    ax.set_xlabel(r'$s_x$ [mm]');ax.set_ylabel(r'$s_y$ [mm]');
    ax.set_zlabel(r'$\| \tau \| = \sqrt{\tau_x^2 + \tau_y^2}$ [MPa]');


class Explore(bu.Model):
    name = 'Damage model explorer'
    bs = bu.Instance(MATS1D5BondSlipD, ())

    tree = ['bs']

    def __init__(self, *args, **kw):
        super(Explore, self).__init__(*args, **kw)
        self.reset_i()

    def reset_i(self):
        self.s_x_0, self.s_y_0 = 0, 0
        self.t0 = 0
        self.Sig_record = []
        self.Eps_record = []
        iter_record = []
        self.t_arr = []
        self.s_x_t, self.s_y_t = [], []
        self.Eps_n1 = np.zeros((3,), dtype=np.float_)

    def get_response_i(self):
        n_steps = self.n_steps
        t1 = self.t0 + n_steps + 1
        ti_arr = np.linspace(self.t0, t1, n_steps + 1)
        si_x_t = np.linspace(self.s_x_0, self.s_x_1, n_steps + 1)
        si_y_t = np.linspace(self.s_y_0, self.s_y_1, n_steps + 1)
        for s_x_n1, s_y_n1 in zip(si_x_t, si_y_t):
            self.Eps_n1, self.Sig_n1 = get_tau_s(s_x_n1, s_y_n1, self.Eps_n1, self.bs)
            self.Sig_record.append(self.Sig_n1)
            self.Eps_record.append(self.Eps_n1)
        self.t_arr = np.hstack([self.t_arr, ti_arr])
        self.s_x_t = np.hstack([self.s_x_t, si_x_t])
        self.s_y_t = np.hstack([self.s_y_t, si_y_t])
        self.t0 = t1
        self.s_x_0, self.s_y_0 = self.s_x_1, self.s_y_1
        return

    def plot_Sig_Eps(self, ax1, Sig_arr):
        tau_x, tau_y, kappa = Sig_arr.T
        tau = np.sqrt(tau_x ** 2 + tau_y ** 2)
        ax1.plot3D(self.s_x_t, self.s_y_t, tau, color='orange', lw=3)

    def subplots(self, fig):
        ax_sxy = fig.add_subplot(1, 1, 1, projection='3d')
        return ax_sxy

    def update_plot(self, ax):
        self.get_response_i()
        Sig_arr = np.array(self.Sig_record, dtype=np.float_)
        Eps_arr = np.array(self.Eps_record, dtype=np.float_)
        plot_tau_s(ax, Eps_arr[-1, ...],
                   self.s_min, self.s_max, 500, self.bs)
        self.plot_Sig_Eps(ax, Sig_arr)
        ax.plot(self.s_x_t, self.s_y_t, 0, color='red')

    n_s = bu.Int(500, BC=True)
    s_x_1 = bu.Float(0, BC=True)
    s_y_1 = bu.Float(0, BC=True)
    n_steps = bu.Float(20, BC=True)

    s_min = bu.Float(-0.1, BC=True)
    s_max = bu.Float(0.1, BC=True)

    def run(self, update_progress=lambda t: t):
        try:
            self.get_response_i(update_progress)
        except ValueError:
            print('No convergence reached')
            return

    t = bu.Float(0)
    t_max = bu.Float(1)

    def reset(self):
        self.reset_i()

    ipw_view = bu.View(
        bu.Item('s_max'),
        bu.Item('n_s'),
        bu.Item('s_x_1', editor=bu.FloatRangeEditor(low_name='s_min',high_name='s_max')),
        bu.Item('s_y_1', editor=bu.FloatRangeEditor(low_name='s_min',high_name='s_max')),
        bu.Item('n_steps'),
    )

