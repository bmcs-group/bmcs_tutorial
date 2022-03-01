
import bmcs_utils.api as bu
import sympy as sp
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class PO_ESF_RLM_Symb(bu.SymbExpr):

    E_f, A_f = sp.symbols(r'E_\mathrm{f}, A_\mathrm{f}', positive=True)
    E_m, A_m = sp.symbols(r'E_\mathrm{m}, A_\mathrm{m}', positive=True)
    tau, p = sp.symbols(r'\bar{\tau}, p', positive=True)
    C, D = sp.symbols(r'C, D')
    P, w = sp.symbols(r'P, w', positive=True)
    x, a, L_b = sp.symbols(r'x, a, L_b')

    d_sig_f = p * tau / A_f

    sig_f = sp.integrate(d_sig_f, x) + C
    eps_f = sig_f / E_f

    u_f = sp.integrate(eps_f, x) + D

    eq_C = {P - sig_f.subs({x:0}) * A_f}
    C_subs = sp.solve(eq_C,C)

    eqns_D = {u_f.subs(C_subs).subs(x, a)}
    D_subs = sp.solve(eqns_D, D)

    u_f.subs(C_subs).subs(D_subs)
    eqns_a = {eps_f.subs(C_subs).subs(D_subs).subs(x, a)}
    a_subs = sp.solve(eqns_a, a)

    var_subs = {**C_subs,**D_subs,**a_subs}

    u_f_x = u_f.subs(var_subs)

    u_P_f_x = sp.Piecewise((u_f_x, x > var_subs[a]),
                          (0, x <= var_subs[a]))

    eps_P_f_x = sp.diff(u_P_f_x,x)

    sig_P_f_x = E_f * eps_P_f_x

    tau_P_x = sp.simplify(sig_P_f_x.diff(x) * A_f / p)

    P_w_pull = sp.solve(u_f_x.subs({x: 0}) - w, P)[0]

    w_L_b = u_P_f_x.subs(x, -L_b).subs(P, P_w_pull)

    a_w_pull = a_subs[a].subs(P, P_w_pull)

    P_max = p * tau * L_b
    w_argmax = sp.solve(P_max - P_w_pull, w)[0]
    P_w_up_pull = P_w_pull
    b, P_down = sp.symbols(r'b, P_\mathrm{down}')
    sig_down = P_down / A_f
    eps_down = 1 / E_f * sig_down
    w_down = (L_b + b) - sp.Rational(1, 2) * eps_down * b
    P_w_down_pull, P_w_down_push = sp.solve(
        w_down.subs(b, -P_down / p / tau) - w,
        P_down
    )

    P_w_short = sp.Piecewise((0, w <= 0),
                            (P_w_up_pull, w <= w_argmax),
                            (P_w_down_pull, w < L_b),
                            (0, True)
                            )

    w_L_b_a = L_b - P_w_down_pull / p / tau
    w_L_b = sp.Piecewise((0, w <= w_argmax),
                         (w_L_b_a, (w > w_argmax) & (w <= L_b)),
                         (w, True))
    a_w_pull = - (P_w_short / p / tau)
    P_w_pull = P_w_short

    a_w_up = -P_w_up_pull / p / tau
    b_w_down = -P_w_down_pull / p / tau

    u_w_f_up = sp.integrate(-P_w_up_pull / E_f/ A_f / a_w_up * (x - a_w_up),(x, a_w_up, x))
    u_w_f_do = sp.integrate(-P_w_down_pull / E_f/ A_f / b_w_down * (x - b_w_down),(x, b_w_down, x))
    u_w_f_x = sp.Piecewise(
        (0, ((w <= w_argmax) & (x <= a_w_up))),
        (u_w_f_up, ((w <= w_argmax) & (x > a_w_up))),
        (0, ((w > w_argmax) & (x <= b_w_down))),
        (u_w_f_do + w_L_b, ((w > w_argmax) & (x > b_w_down))),
    )
    eps_w_f_x = u_w_f_x.diff(x)
    sig_w_f_x = E_f * eps_w_f_x
    tau_w_x = sig_w_f_x.diff(x)

    eps_w_m_x = x * 1e-9
    sig_w_m_x = x * 1e-9
    u_w_m_x = x * 1e-9 # u_w_f_x * 1e-8

    #-------------------------------------------------------------------------
    # Declaration of the lambdified methods
    #-------------------------------------------------------------------------

    symb_model_params = ['E_f', 'A_f', 'E_m', 'A_m', 'tau', 'p', 'L_b']

    symb_expressions = [
        ('u_w_f_x', ('x','w',)),
        ('u_w_m_x', ('x','w',)),
        ('eps_w_f_x', ('x','w',)),
        ('eps_w_m_x', ('x','w',)),
        ('sig_w_f_x', ('x','w',)),
        ('sig_w_f_x', ('x','w',)),
        ('sig_w_m_x', ('x','w',)),
        ('sig_w_m_x', ('x','w',)),
        ('tau_w_x', ('x','w',)),
        ('w_L_b', ('w',)),
        ('a_w_pull', ('w',)),
        ('P_w_pull', ('w',)),
    ]

