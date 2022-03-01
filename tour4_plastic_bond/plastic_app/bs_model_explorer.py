
import bmcs_utils.api as bu
import numpy as np
import traits.api as tr
from .bs_ep_ikh import BS_EP_IKH, ConvergenceError
# from bmcs_matmod.slide.energy_dissipation import EnergyDissipation
from .bs_history import BSHistory

class BSModelExplorer(bu.Model):
    name = 'Explorer'

    tree = ['bs_model', 'history']

    bs_model = bu.Instance(BS_EP_IKH, ())

    # energy_dissipation = bu.Instance(EnergyDissipation, tree=True)
    # '''Viewer to the energy dissipation'''
    # def _energy_dissipation_default(self):
    #     return EnergyDissipation(slider_exp=self)
    #
    history = bu.Instance(BSHistory)
    '''Viewer to the inelastic state evolution'''
    def _history_default(self):
        return BSHistory(slider_exp=self)

    def __init__(self, *args, **kw):
        super(BSModelExplorer, self).__init__(*args, **kw)
        self.reset_i()

    n_Eps = tr.Property()

    def _get_n_Eps(self):
        return 3

    s_1 = bu.Float(0, INC=True)

    n_steps = bu.Int(10, ALG=True)
    k_max = bu.Int(20, ALG=True)

    Sig_arr = tr.Array
    Eps_arr = tr.Array

    Sig_t = tr.Property
    def _get_Sig_t(self):
        return self.Sig_arr

    Eps_t = tr.Property
    def _get_Eps_t(self):
        return self.Eps_arr

    ipw_view = bu.View(
        bu.Item('s_1', latex=r's_1'), #, editor=bu.FloatRangeEditor(low=-4,high=4,n_steps=50)),
        bu.Item('n_steps'),
        bu.Item('t', readonly=True),
        bu.Item('t_max', readonly=True),
        time_editor=bu.ProgressEditor(run_method='run',
                                   reset_method='reset',
                                   interrupt_var='sim_stop',
                                   time_var='t',
                                   time_max='t_max',
                                   )
    )

    def reset_i(self, event=None):
        self.s_0 = 0
        self.t_0 = 0
        self.t = 0
        self.t_max = 1
        self.Sig_arr = np.zeros((0, self.n_Eps))
        self.Eps_arr = np.zeros((0, self.n_Eps))
        self.Sig_record = []
        self.Eps_record = []
        self.iter_record = []
        self.t_arr = []
        self.s_t = []
        self.Eps_n1 = np.zeros((self.n_Eps,), dtype=np.float_)
        self.Sig_n1 = np.zeros((self.n_Eps,), dtype=np.float_)
        self.s_1 = 0

    t = bu.Float(0)
    t_max = bu.Float(1)

    def get_response_i(self, event=None):
        n_steps = self.n_steps
        t_1 = self.t_0 + 1
        self.t_max = t_1
        ti_arr = np.linspace(self.t_0, t_1, n_steps + 1)
        si_t = np.linspace(self.s_0, self.s_1, n_steps + 1) + 1e-9
        for t, s_n1 in zip(ti_arr, si_t):
            try: self.Eps_n1, self.Sig_n1, k = self.bs_model.get_sig_n1(
                    s_n1, self.Sig_n1, self.Eps_n1, self.k_max
                )
            except ConvergenceError as e:
                print(e)
                break
            self.Sig_record.append(self.Sig_n1)
            self.Eps_record.append(self.Eps_n1)
            self.iter_record.append(k)
            self.t = t

        self.Sig_arr = np.array(self.Sig_record, dtype=np.float_)
        self.Eps_arr = np.array(self.Eps_record, dtype=np.float_)
        self.iter_t = np.array(self.iter_record, dtype=np.int_)
        n_i = len(self.iter_t)
        self.t_arr = np.hstack([self.t_arr, ti_arr])[:n_i]
        self.s_t = np.hstack([self.s_t, si_t])[:n_i]
        self.t_0 = t_1
        self.s_0 = self.s_1
        # set the last step index in the response browser
        self.history.t_max = self.t_arr[-1]
        return

    def plot_tau_s(self, ax):
        tau_t = self.Sig_arr.T[0, ...]
        ax.plot(self.s_t, tau_t, color='orange', lw=3)

    def run(self):
        try:
            self.get_response_i()
        except ValueError:
            print('No convergence reached')
            return

    def reset(self):
        self.reset_i()

    def subplots(self, fig):
        ax_tau = fig.subplots(1, 1)
        return ax_tau

    def update_plot(self, axes):
        ax_tau = axes
        self.plot_tau_s(ax_tau)
        ax_tau.set_xlabel(r'$w$ [mm]');
        ax_tau.set_ylabel(r'$\sigma$ [MPa]');

