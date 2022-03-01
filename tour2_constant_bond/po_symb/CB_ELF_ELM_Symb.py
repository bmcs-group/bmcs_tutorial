
import bmcs_utils.api as bu
import sympy as sp


class CB_ELF_ELM_Symb(bu.SymbExpr):

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

    eq_C = {P - sig_f.subs({x:0}) * A_f}
    C_subs = sp.solve(eq_C,C)
    eq_D = {sig_m.subs({x:0}) * A_m}
    D_subs = sp.solve(eq_D,D)

    eqns_u_equal = {u_f.subs(C_subs).subs(x,a) - u_m.subs(D_subs).subs(x,a)}
    E_subs = sp.solve(eqns_u_equal,E)

    eqns_eps_equal = {eps_f.subs(C_subs).subs(x,a) - eps_m.subs(D_subs).subs(x,a)}

    a_subs = sp.solve(eqns_eps_equal,a)

    var_subs = {**C_subs,**D_subs,**E_subs,**a_subs}

    u_f_x = sp.simplify( u_f.subs(var_subs) )
    u_m_x = sp.simplify( u_m.subs(var_subs) )

    A_c = A_m + A_f
    E_c = (E_m * A_m + E_f * A_f) / (A_c)
    G = sp.symbols('G')
    u_c_x_ = P / A_c / E_c * x + G
    G_subs = sp.solve( u_c_x_.subs(x, -L_b), G)[0]

    u_P_c_x = sp.simplify(u_c_x_.subs(G, G_subs).subs(var_subs))
    u_c_a = sp.simplify(u_P_c_x.subs(x,var_subs[a]) )

    F_sol = sp.simplify( sp.solve( u_m_x.subs(x, a_subs[a]) - u_c_a, F)[0] )
    u_P_mc_x = u_m_x.subs(F, F_sol)
    u_P_fc_x = u_f_x.subs(F, F_sol)

    u_P_f_x = sp.Piecewise((u_P_c_x, x <= var_subs[a]),
                           (u_P_fc_x, x > var_subs[a])
                        )
    u_P_m_x = sp.Piecewise((u_P_c_x, x <= var_subs[a]),
                           (u_P_mc_x, x > var_subs[a]),
                         )
    eps_P_f_x = sp.diff(u_P_f_x,x)
    eps_P_m_x = sp.diff(u_P_m_x,x)
    sig_P_f_x = E_f * eps_P_f_x
    sig_P_m_x = E_m * eps_P_m_x
    tau_P_x = sig_P_f_x.diff(x) * A_f / p

    w_P_pull_ = u_P_fc_x.subs(x,0)

    P_w_pull_min, P_w_pull_pls = sp.solve(w_P_pull_ - w, P)
    P_w_pull_ = P_w_pull_min

    a_w_pull = a_subs[a].subs(P, P_w_pull_)
    w_argmax1 = sp.solve(sp.Eq(a_w_pull, -L_b), w)[0]

    d_Pw_pull_dw = sp.diff(P_w_pull_, w)

    w_argmax = w_argmax1
    K_c = sp.simplify(d_Pw_pull_dw.subs({w: w_argmax}))

    P_c = P_w_pull_.subs(w, w_argmax)
    P_w_clamped = sp.Piecewise(
        (P_w_pull_, w < w_argmax),
        (P_c + K_c * (w - w_argmax), w >= w_argmax)
    )


    P_w_pull = P_w_clamped
    w_L_b = P_w_pull * 1e-9

    u_w_c_x = u_P_c_x.subs(P, P_w_pull_)
    u_w_fc_x = u_P_fc_x.subs(P, P_w_pull_)
    u_w_mc_x = u_P_mc_x.subs(P, P_w_pull_)

    P_d = P_w_pull_ - P_c
    u_d = P_d / A_f / E_f * (x + L_b)

    u_w_f_x = sp.Piecewise((u_w_c_x, ((w <= w_argmax) & (x <= a_w_pull))),
                           (u_w_fc_x, ((w <= w_argmax) & (x > a_w_pull))),
                           (u_d + u_w_fc_x.subs(w, w_argmax), (w > w_argmax))
                        )
    u_w_m_x = sp.Piecewise((u_w_c_x, ((w <= w_argmax) & (x <= a_w_pull))),
                            (u_w_mc_x, ((w <= w_argmax) & (x > a_w_pull))),
                           (u_w_mc_x.subs(w, w_argmax), (w > w_argmax))
                         )
    eps_w_f_x = u_w_f_x.diff(x)
    sig_w_f_x = E_f * eps_w_f_x
    eps_w_m_x = u_w_m_x.diff(x)
    sig_w_m_x = E_m * eps_w_m_x
    tau_w_x = A_f / p * sig_w_f_x.diff(x)
    tau_w_x = -A_m / p * sig_w_m_x.diff(x)

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
