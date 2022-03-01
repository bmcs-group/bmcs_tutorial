
import bmcs_utils.api as bu
import sympy as sp


class PO_ELF_RLM_Symb(bu.SymbExpr):
    """Pullout of elastic Long fiber, fromm rigid long matrix
    """
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

    u_fa_x = sp.Piecewise((u_f_x, x > var_subs[a]),
                          (0, x <= var_subs[a]))

    eps_f_x = sp.diff(u_fa_x,x)

    sig_f_x = E_f * eps_f_x

    tau_x = sp.simplify(sig_f_x.diff(x) * A_f / p)

    u_f_x.subs(x, 0) - w

    Pw_pull = sp.solve(u_f_x.subs({x: 0}) - w, P)[0]

    P_max = p * tau * L_b
    w_argmax = sp.solve(P_max - Pw_pull, w)[0]

    Pw_pull_Lb = sp.Piecewise((Pw_pull, w < w_argmax),
                              (P_max, w >= w_argmax))

    w_L_b = u_fa_x.subs(x, -L_b).subs(P, Pw_pull)

    aw_pull = a_subs[a].subs(P, Pw_pull)

    eps_m_x = eps_f_x * 1e-8
    sig_m_x = sig_f_x * 1e-8
    u_ma_x = u_fa_x * 1e-8
    #-------------------------------------------------------------------------
    # Declaration of the lambdified methods
    #-------------------------------------------------------------------------

    symb_model_params = ['E_f', 'A_f', 'tau', 'p', 'L_b']

    symb_expressions = [
        ('eps_f_x', ('x','P',)),
        ('eps_m_x', ('x','P',)),
        ('sig_f_x', ('x','P',)),
        ('sig_m_x', ('x','P',)),
        ('tau_x', ('x','P',)),
        ('u_fa_x', ('x','P',)),
        ('u_ma_x', ('x','P',)),
        ('w_L_b', ('w',)),
        ('aw_pull', ('w',)),
        ('Pw_pull', ('w',)),
    ]
