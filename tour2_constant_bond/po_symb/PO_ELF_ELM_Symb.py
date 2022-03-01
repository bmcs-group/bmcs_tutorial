
import bmcs_utils.api as bu
import sympy as sp


class PO_ELF_ELM_Symb(bu.SymbExpr):
    """Pullout of elastic Long fiber, fromm elastic long matrix
    """
    E_m, A_m = sp.symbols(r'E_\mathrm{m}, A_\mathrm{m}', nonnegative=True)
    E_f, A_f = sp.symbols(r'E_\mathrm{f}, A_\mathrm{f}', nonnegative=True)
    tau, p = sp.symbols(r'\bar{\tau}, p', nonnegative=True)
    C, D, E, F = sp.symbols('C, D, E, F')
    P, w = sp.symbols('P, w')
    x, a, L_b = sp.symbols('x, a, L_b')

    d_sig_f = p * tau / A_f
    d_sig_m = -p * tau / A_m

    sig_f = sp.integrate(d_sig_f, x) + C
    sig_m = sp.integrate(d_sig_m, x) + D

    eps_f = sig_f / E_f
    eps_m = sig_m / E_m

    u_f = sp.integrate(eps_f, x) + E
    u_m = sp.integrate(eps_m, x) + F

    eq_C = {P - sig_f.subs({x: 0}) * A_f}
    C_subs = sp.solve(eq_C, C)
    eq_D = {P + sig_m.subs({x: 0}) * A_m}
    D_subs = sp.solve(eq_D, D)

    F_subs = sp.solve({u_m.subs(x, 0) - 0}, F)

    eqns_u_equal = {u_f.subs(C_subs).subs(x, a) - u_m.subs(D_subs).subs(F_subs).subs(x, a)}
    E_subs = sp.solve(eqns_u_equal, E)

    eqns_eps_equal = {eps_f.subs(C_subs).subs(x, a) - eps_m.subs(D_subs).subs(x, a)}
    a_subs = sp.solve(eqns_eps_equal, a)
    var_subs = {**C_subs, **D_subs, **F_subs, **E_subs, **a_subs}

    u_f_x = u_f.subs(var_subs)
    u_m_x = u_m.subs(var_subs)

    u_P_f_x = sp.Piecewise((u_f_x.subs(x, var_subs[a]), x <= var_subs[a]),
                          (u_f_x, x > var_subs[a]))
    u_P_m_x = sp.Piecewise((u_m_x.subs(x, var_subs[a]), x <= var_subs[a]),
                          (u_m_x, x > var_subs[a]))

    eps_P_f_x = sp.diff(u_P_f_x, x)
    eps_P_m_x = sp.diff(u_P_m_x, x)

    sig_P_f_x = E_f * eps_P_f_x
    sig_P_m_x = E_m * eps_P_m_x

    tau_P_x = sig_P_f_x.diff(x) * A_f / p

    Pw_push, P_w_pull = sp.solve(u_f_x.subs({x: 0}) - w, P)

    w_L_b = u_P_f_x.subs(x, -L_b).subs(P, P_w_pull)

    a_w_pull = a_subs[a].subs(P, P_w_pull)

    eps_w_f_x = eps_P_f_x.subs(P, P_w_pull)
    eps_w_m_x = eps_P_m_x.subs(P, P_w_pull)
    sig_w_f_x = sig_P_f_x.subs(P, P_w_pull)
    sig_w_m_x = sig_P_m_x.subs(P, P_w_pull)
    u_w_f_x = u_P_f_x.subs(P, P_w_pull)
    u_w_m_x = u_P_m_x.subs(P, P_w_pull)
    tau_w_x = tau_P_x.subs(P, P_w_pull)

    #-------------------------------------------------------------------------
    # Declaration of the lambdified methods
    #-------------------------------------------------------------------------

    symb_model_params = ['E_f', 'A_f', 'E_m', 'A_m', 'tau', 'p', 'L_b']

    symb_expressions = [
        ('eps_w_f_x', ('x','w',)),
        ('eps_w_m_x', ('x','w',)),
        ('sig_w_f_x', ('x','w',)),
        ('sig_w_m_x', ('x','w',)),
        ('tau_w_x', ('x','w',)),
        ('u_w_f_x', ('x','w',)),
        ('u_w_m_x', ('x','w',)),
        ('w_L_b', ('w',)),
        ('a_w_pull', ('w',)),
        ('P_w_pull', ('w',)),
    ]
