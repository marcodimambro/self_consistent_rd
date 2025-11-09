import numpy as np
from rd_framework import solve_steady_state
import examples_specs
import plotting

# 1) Build a spec from your params
params = dict(
    D=1.0, beta=1.0, L=100,
    eps_x = -0.01, eps_y = -0.01, eps_xy = 0.2,
    eps_ax = 0.001, eps_bx = -0.02, eps_cx = -0.02,
    eps_ay = -0.02, eps_by = 0.02, eps_cy = 0.03,
    kappa_x=0.5, kappa_y=0.5,
    mu_xy_II = 0.0,
    mu_ac = 0,
    k0_client=0, k0_scaffold=1,
    alpha_ab=0.2, gamma_ab=1.0,
    alpha_ac=0.8, gamma_ac=1.0,
    alpha_bc=0.5, gamma_bc=1.0,
    
    alpha_xy_I=2, gamma_xy_I=1.1,
    alpha_xy_II=3, gamma_xy_II = 1.1,
    
    #alpha_xy =1.1, gamma_xy =1.0,
    #alpha_xz =1.1, gamma_xz =1.1,
    #alpha_yz =2.1, gamma_yz =1.0,
    #mu_xz =0,
    #eps_px = -0.01, eps_py = 0.01,
    #eps_kx = 0.01, eps_ky = -0.01,
    
    N_a = 1.0, N_b = 1.0, N_c = 1.0,
    N_x = 1000.0, N_y = 1000.0
    
)
#spec = examples_specs.spec_complete(params)
#spec = examples_specs.spec_two_triangles(params)
spec = examples_specs.spec_test(params)

# 2) Grid and initial guess
L = params["L"]
dx = 6 * np.sqrt(params['kappa_x']) if params['kappa_x'] > 0 else 1
nx = int(2 * params['L'] / dx + 1)
x = np.linspace(-L, L, nx)
nsp = len(spec.species)
rho0 = np.ones((nsp, nx)) * (1.0 / (2*L))  # simple flat start, replace with your existing initializers
# add noise to avoid symmetry issues
#rho0 += 0.01 * np.random.rand(nsp, nx)

for i, s in enumerate(spec.species):
    #if s.role == 'scaffold':
        rho0[i,:] = 1 * (1 + 0.1 * (-1)*i * np.tanh(x)) 
        
rho_ss, info = solve_steady_state(x, rho0, spec, beta=params["beta"], max_iter=10000, tol=1e-6, verbose=True, interval=100)
#plot
plotting.plot_results(
    x=x,
    rho=rho_ss,
    spec=spec,
    beta=params.get("beta", 1.0),
    quantities=("densities", "fluxes", "potentials", "barriers"),  # pick what you want
    roles= ('client', 'scaffold'),  # or a subset like ("client","
)
#plotting.plot_eq_neq(x=x, rho_eq=rho_ss, rho_neq=rho_ss, spec=spec, show=True, roles=('client','scaffold'))


# 3) Totals: per species and/or group (use 'group:0', 'group:1', ... for groups)

# k0_client = params.get('k0_client',1.0)
# k0_scaffold = params.get('k0_scaffold',1.0)
# if k0_client != 0 and k0_scaffold != 0:
#     totals = {
#         "a": N_a, "b": N_b, "c": N_c,    # if spec has individual species totals
#         "x": N_x, "y": N_y,
#         "p": N_p, "k": N_k,
#         "group:0": N_a + N_b + N_c,                            # if spec has groups defined, totals per group
#         "group:1": N_x + N_y + N_z
#     }
# elif k0_client != 0 and k0_scaffold == 0:
#     totals = {
#         "a": N_a, "b": N_b, "c": N_c,
#         "x": N_x, "y": N_y, "z": N_z,
#         "p": N_p, "k": N_k,
#         "group:0": N_a + N_b + N_c
#     }
# elif k0_client == 0 and k0_scaffold != 0:
#     totals = {
#         "a": N_a, "b": N_b, "c": N_c,
#         "x": N_x, "y": N_y, "z": N_z,
#         "p": N_p, "k": N_k,
#         "group:0": N_x + N_y + N_z
#     }
# else:
#     totals = {
#         "a": N_a, "b": N_b, "c": N_c,
#         "x": N_x, "y": N_y, "z": N_z,
#         "p": N_p, "k": N_k,
#     }

# rho_ss, info = solve_steady_state(x, rho0, spec, totals, beta=params["beta"], max_iter=10000, tol=1e-6, verbose=True, interval=100)
# print(info)
# params['mu_xz'] = 0.2
# spec_neq = examples_specs.spec_two_triangles(params)
# rho_neq, info = solve_steady_state(x, rho_ss, spec_neq, totals, beta=params["beta"], max_iter=10000, tol=1e-6, verbose=True)
# plotting.plot_eq_neq(x=x, rho_eq=rho_ss, rho_neq=rho_neq, spec=spec, show=True, roles=('client','scaffold'))
# # # Plot selected quantities; choose any subset of {'densities','fluxes','potentials','barriers'}
# # plot_results(
# #     x=x,
# #     rho=rho_ss,
# #     spec=spec,
# #     beta=params.get("beta", 1.0),
# #     quantities=("densities", "fluxes", "potentials", "barriers"),  # pick what you want
# #     roles= ('client', 'scaffold'),  # or a subset like ("client","enzyme")
# #     V=None,      # pass precomputed V if you already have it; otherwise it’ll compute it
# #     show=True
# # )
