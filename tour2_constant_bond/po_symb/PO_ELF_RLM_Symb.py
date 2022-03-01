
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

    u_P_f_x = sp.Piecewise((u_f_x, x > var_subs[a]),
                           (0, x <= var_subs[a]))

    eps_P_f_x = sp.diff(u_P_f_x,x)

    sig_P_f_x = E_f * eps_P_f_x

    tau_P_x = sp.simplify(sig_P_f_x.diff(x) * A_f / p)

    P_w_pull = sp.solve(u_f_x.subs({x: 0}) - w, P)[0]

    w_L_b = u_P_f_x.subs(x, -L_b).subs(P, P_w_pull)

    a_w_pull = a_subs[a].subs(P, P_w_pull)

    sig_w_f_x = sp.Piecewise(
        (0, (x < a_w_pull)),
        (P_w_pull / A_f * (x - a_w_pull), True)
    )

    sig_w_f_x = sig_P_f_x.subs(P,P_w_pull)
    eps_w_f_x = eps_P_f_x.subs(P,P_w_pull)
    u_w_f_x = u_P_f_x.subs(P,P_w_pull)
    tau_w_x = tau_P_x.subs(P,P_w_pull)

    eps_w_m_x = eps_w_f_x * 1e-8
    sig_w_m_x = sig_w_f_x * 1e-8
    u_w_m_x = u_w_f_x * 1e-8

    #-------------------------------------------------------------------------
    # Declaration of the lambdified methods
    #-------------------------------------------------------------------------

    symb_model_params = ['E_f', 'A_f', 'tau', 'p', 'L_b']

    symb_expressions = [
        ('eps_w_f_x', ('x','w',)),
        ('eps_w_m_x', ('x','w',)),
        ('sig_w_f_x', ('x','w',)),
        ('sig_w_f_x', ('x','w',)),
        ('sig_w_m_x', ('x','w',)),
        ('tau_w_x', ('x','w',)),
        ('u_w_f_x', ('x','w',)),
        ('u_w_m_x', ('x','w',)),
        ('w_L_b', ('w',)),
        ('a_w_pull', ('w',)),
        ('P_w_pull', ('w',)),
    ]
