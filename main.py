import numpy as np
from rd_framework import solve_steady_state
import examples_specs
import plotting


# example usage:

# 1) Build a spec from your params
params = dict(
    D=1.0, beta=1.0, L=100,
    
    eps_x = -0.01, eps_y = -0.01, eps_xy = 0.2,
    eps_ax = 0.001, eps_bx = -0.02, eps_cx = -0.02,
    eps_ay = -0.02, eps_by = 0.02, eps_cy = 0.03,
    
    kappa_x= 0.5 , kappa_y = 0.5,
    
    mu_xy_II = 0.0,
    mu_ac = 0,
    
    k0_client=1, k0_scaffold=0,
    
    alpha_ab=0.2, gamma_ab=1.0,
    alpha_ac=0.8, gamma_ac=1.0,
    alpha_bc=0.5, gamma_bc=1.0,
    
    alpha_xy_I= 5, beta_xy_I=0.5, gamma_xy_I = 1.1,
    alpha_xy_II= 0, beta_xy_II=0.5, gamma_xy_II = 1.1,

    N_a = 10.0, N_b = 10.0, N_c = 10.0,
    N_x = 1000.0, N_y = 1000.0,
    
    #N_p = 1000, N_k = 1000,
    #eps_px = 0.01, eps_kx = -0.01, eps_py = -0.01, eps_ky = 0.01
    
)

spec = examples_specs.spec_test(params)

# 2) Grid and initial guess
L = params["L"]
dx = 6 * np.sqrt(params['kappa_x']) if params['kappa_x'] > 0 else 0.5
nx = int(2 * params['L'] / dx + 1)
x = np.linspace(-L, L, nx)
nsp = len(spec.species)
rho0 = np.ones((nsp, nx)) * (1.0 / (2*L))  # simple flat start 
# add noise to avoid symmetry issues
rho0 += 0.01 * np.random.rand(nsp, nx)

for i, s in enumerate(spec.species):
    #if s.role == 'scaffold':
        rho0[i,:] = 1 * (1 - 2 * (-1)*(i) * np.tanh(x)) # biasing towards a single interface. Not necessary if we want to study random initial conditions
        # add a nucleation seed
        #rho0[i,:] += 0.1 * np.exp(-0.5 * (x / 1) ** 2) 
        
rho_ss, info = solve_steady_state(x, rho0, spec, beta=params["beta"], max_iter=10000, tol=1e-6, verbose=True, interval=100)
print(info)

# plotting.plot_results(
#     x=x,
#     rho=rho_ss,
#     spec=spec,
#     beta=params.get("beta", 1.0),
#     quantities=("densities", "fluxes", "barriers"),  # pick what you want
#     roles= ('client', 'scaffold'),  # or a subset like ("client","
# )

params['mu_ac'] = 1
params['mu_xy_I'] = 0
spec_neq = examples_specs.spec_test(params)
rho_neq, info = solve_steady_state(x, rho_ss, spec_neq, beta=params["beta"], max_iter=10000, tol=1e-6, verbose=True, interval=100)
print(info)

plotting.plot_eq_neq(x=x, rho_eq=rho_ss, rho_neq=rho_neq, spec=spec, show=True, roles=('client','scaffold'))

