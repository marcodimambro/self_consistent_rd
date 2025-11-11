
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Iterable, Optional, Union
from diffusion_drift_operators import compute_fluxes_gummel
import pickle

from rd_framework import (
    ModelSpec,
    compute_potentials,
    _safe_eval_barrier,
)

Array = np.ndarray


def compute_fluxes_gummel_method(x: Array, rho_by_species: Dict[str, Array], V_by_species: Dict[str, Array], D_by_species: Dict[str, float], beta: float) -> Dict[str, Tuple[Array, Array]]:
    """
    Returns dict: species -> (x_edge, J_edge) where x_edge has length nx-1.
    """
    x_edge = 0.5 * (x[:-1] + x[1:])
    fluxes = {}
    for name, rho in rho_by_species.items():
        V = V_by_species[name]
        D = D_by_species[name]
        J = compute_fluxes_gummel(x, V, rho, D, beta)
        fluxes[name] = (x_edge, J)
    return fluxes

def compute_barriers(x: Array, V_by_species: Dict[str, Array], spec: ModelSpec) -> Dict[Tuple[str, int], Array]:
    """
    Compute barrier profiles for each reaction pathway using the same expressions
    defined in the spec. Returns dict keyed by (reaction_name, pathway_index) -> barrier array.
    """
    barriers = {}
    for r in spec.reactions:
        for pi, pw in enumerate(r.pathways):
            b = _safe_eval_barrier(pw.barrier.expr, V_by_species, x, pw.barrier.params)
            barriers[(r.name, pi)] = b
    return barriers

def _species_by_role(spec: ModelSpec) -> Dict[str, List[str]]:
    by_role: Dict[str, List[str]] = {}
    for s in spec.species:
        role = s.role or "unlabeled"
        by_role.setdefault(role, []).append(s.name)
    return by_role

def _collect_species_data(rho: Array, spec: ModelSpec) -> Dict[str, Array]:
    names = [s.name for s in spec.species]
    return {n: rho[i] for i, n in enumerate(names)}

# create a function to extract fluxes, barriers, potentials and densities 
def extract_results( x: Array, rho: Array, spec: ModelSpec, beta: float, V: Optional[Dict[str, Array]] = None) -> Dict[str, Dict]:
    """
    Extract densities, fluxes, potentials, and barriers into a dictionary.

    Parameters
    ----------
    x : grid (nx,)
    rho : densities with shape (n_species, nx)
    spec : ModelSpec
    beta : inverse temperature (drift strength factor in J)
    V : optional potentials dict per species; computed if None

    Returns
    -------
    A dictionary of computed data:
    {
      'densities': {species: (x, rho)} ,
      'fluxes':    {species: (x_edge, J_edge)} ,
      'potentials':{species: (x, V)} ,
      'barriers':  {(rxn, pi): barrier_array}
    }
    """
    out: Dict[str, Dict] = {}

    rho_by_species = _collect_species_data(rho, spec)
    names = [s.name for s in spec.species]
    D_map = {s.name: s.D for s in spec.species}

    out['densities'] = {}
    for n in names:
        out['densities'][n] = (x, rho_by_species[n])

    if V is None:
        V = compute_potentials(x, rho, spec)

    out['potentials'] = {}
    for n in names:
        out['potentials'][n] = (x, V[n])

    fluxes = compute_fluxes_gummel_method(x, rho_by_species, V, D_map, beta)
    out['fluxes'] = {}
    for n in names:
        out['fluxes'][n] = fluxes[n]

    B = compute_barriers(x, V, spec)
    out['barriers'] = B

    return out

def plot_results(
    x: Array,
    rho: Array,
    spec: ModelSpec,
    beta: float,
    quantities: Iterable[str] = ('densities', 'fluxes'),
    roles: Optional[Iterable[str]] = None,
    V: Optional[Dict[str, Array]] = None,
    show: bool = True,
) -> Dict[str, Dict[str, Union[Dict[str, Tuple[Array, Array]], Dict[Tuple[str,int], Array]]]]:
    """
    General plotting function for rd_framework steady-state results.

    Parameters
    ----------
    x : grid (nx,)
    rho : densities with shape (n_species, nx)
    spec : ModelSpec
    beta : inverse temperature (drift strength factor in J)
    quantities : which quantities to plot among {'densities','fluxes','potentials','barriers'}
    roles : optional subset of roles to include; defaults to all roles present in spec
    V : optional potentials dict per species; computed if None
    show : whether to call plt.show()

    Behavior
    --------
    - Creates one figure per (role, quantity). (No subplots.)
    - Densities: plots rho for all species with that role vs x.
    - Fluxes: computes interface fluxes and plots J_edge vs x_edge for those species.
    - Potentials: plots V_i(x) for those species.
    - Barriers: groups all reactions whose SOURCE species has that role; plots each pathway curve.
      Keyed as (reaction_name, pathway_index).

    Returns
    -------
    A dictionary of computed data so you can reuse without re-computation:
    {
      'densities': {role: {species: (x, rho)} } ,
      'fluxes':    {role: {species: (x_edge, J_edge)} } ,
      'potentials':{role: {species: (x, V)} } ,
      'barriers':  {role: {(rxn, pi): barrier_array} }
    }
    """
    quantities = tuple(q.lower() for q in quantities)
    valid = {'densities', 'fluxes', 'potentials', 'barriers'}
    for q in quantities:
        if q not in valid:
            raise ValueError(f"Unknown quantity '{q}'. Valid options: {valid}")
    
    #create a figure quantities x roles
    fig, axs = plt.subplots(len(roles), len(quantities), figsize=(4 * len(quantities), 3 * 2)) # left for clients, right for scaffolds
    if len(quantities) == 1:
        axs = axs.reshape(2,1)

    # prepare species maps
    rho_by_species = _collect_species_data(rho, spec)
    names = [s.name for s in spec.species]
    D_map = {s.name: s.D for s in spec.species}
    by_role = _species_by_role(spec)
    selected_roles = list(roles) if roles is not None else list(by_role.keys())

    # compute potentials if needed
    if (('potentials' in quantities) or ('fluxes' in quantities) or ('barriers' in quantities)) and V is None:
        V = compute_potentials(x, rho, spec)

    out: Dict[str, Dict] = {}
    
    # tab10 colors
    colors = plt.get_cmap('tab10').colors
    # sort the colors in a dict for each role
    color_map: Dict[str, Dict[str, Tuple[float,float,float]]] = {}
    # assign unique colors to species across all roles (never reuse a color)
    available = list(colors)[:]  # copy of tab10 colors
    assigned: Dict[str, Tuple[float, float, float]] = {}
    for role in selected_roles:
        color_map[role] = {}
        for n in by_role.get(role, []):
            if n in assigned:
                color_map[role][n] = assigned[n]
                continue
            if available:
                c = available.pop(0)
            else:
                # palette exhausted: generate a new distinct color deterministically
                idx = len(assigned)
                c = plt.get_cmap('hsv')((idx % 360) / 360.0)[:3]
            assigned[n] = c
            color_map[role][n] = c
    
    # fill in the plots
    for i, qty in enumerate(quantities):
        for j, role in enumerate(selected_roles):
            plt.sca(axs[j,i])
            plt.title(f"{qty.capitalize()} | role={role}")
            if qty == 'densities':
                for n in by_role.get(role, []):
                    plt.plot(x, rho_by_species[n], label=n, color=color_map[role][n])
                plt.xlabel( "x ")
                plt.ylabel( "rho ")
                plt.legend()
                out.setdefault('densities', {}).setdefault(role, {})
                for n in by_role.get(role, []):
                    out['densities'][role][n] = (x, rho_by_species[n])
            elif qty == 'fluxes':
                #fluxes = compute_fluxes(x, rho_by_species, V, D_map, beta)
                fluxes = compute_fluxes_gummel_method(x, rho_by_species, V, D_map, beta)
                for n in by_role.get(role, []):
                    x_edge, J = fluxes[n]
                    plt.plot(x_edge, J, label=n, color=color_map[role][n])
                plt.xlabel( "x (edge) ")
                plt.ylabel( "J ")
                plt.legend()
                out.setdefault('fluxes', {}).setdefault(role, {})
                for n in by_role.get(role, []):
                    out['fluxes'][role][n] = fluxes[n]
            elif qty == 'potentials':
                for n in by_role.get(role, []):
                    plt.plot(x, V[n], label=n, color=color_map[role][n])
                plt.xlabel("x")
                plt.ylabel("V")
                plt.legend()
                out.setdefault('potentials', {}).setdefault(role, {})
                for n in by_role.get(role, []):
                    out['potentials'][role][n] = (x, V[n])
            elif qty == 'barriers':
                B = compute_barriers(x, V, spec)
                role_of = {s.name: (s.role or 'unlabeled') for s in spec.species}
                rxn_src = {r.name: r.source for r in spec.reactions}
                role_barriers: Dict[Tuple[str,int], Array] = {}
                for (rxn, pi), barr in B.items():
                    src = rxn_src[rxn]
                    r_role = role_of.get(src, 'unlabeled')
                    if r_role == role:
                        role_barriers[(rxn, pi)] = barr
                for (rxn, pi), arr in role_barriers.items():
                    plt.plot(x, arr, label=f"{rxn}[path{pi}]")
                plt.xlabel("x")
                plt.ylabel("Barrier")
                plt.legend()
                out.setdefault('barriers', {}).setdefault(role, {})
                out['barriers'][role] = role_barriers
                
    
    plt.tight_layout()
    if show:
        plt.show()

    return out

def create_color_map(spec: ModelSpec, roles: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Tuple[float,float,float]]]:
    """
    Create a color map for species per role using tab10 colors.
    Ensures unique colors across all roles.
    """
    by_role = _species_by_role(spec)
    selected_roles = list(roles) if roles is not None else list(by_role.keys())

    # tab10 colors
    colors = plt.get_cmap('tab10').colors
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:cyan', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:purple']
    # sort the colors in a dict for each role
    color_map: Dict[str, Dict[str, Tuple[float,float,float]]] = {}
    # assign unique colors to species across all roles (never reuse a color)
    available = list(colors)[:]  # copy of tab10 colors
    assigned: Dict[str, Tuple[float, float, float]] = {}
    for role in selected_roles:
        color_map[role] = {}
        for n in by_role.get(role, []):
            if n in assigned:
                color_map[role][n] = assigned[n]
                continue
            if available:
                c = available.pop(0)
            else:
                # palette exhausted: generate a new distinct color deterministically
                idx = len(assigned)
                c = plt.get_cmap('hsv')((idx % 360) / 360.0)[:3]
            assigned[n] = c
            color_map[role][n] = c
    return color_map

def plot_eq_neq(x: Array, rho_eq: Array, rho_neq: Array, spec: ModelSpec, roles: Optional[Iterable[str]] = None, show: bool = True):
    """
    Plot comparison of equilibrium vs non-equilibrium densities per species and fluxes
    """

    rho_eq_map = _collect_species_data(rho_eq, spec)
    rho_neq_map = _collect_species_data(rho_neq, spec)
    by_role = _species_by_role(spec)
    selected_roles = list(roles) if roles is not None else list(by_role.keys())

    color_map = create_color_map(spec, roles=selected_roles)

    fig, axs = plt.subplots(len(selected_roles), 2, figsize=(8, 3 * len(selected_roles)))
    if len(selected_roles) == 1:
        axs = axs.reshape(1,2)

    for j, role in enumerate(selected_roles):
        plt.sca(axs[j,0])
        plt.title(f"Densities | role={role}")
        for n in by_role.get(role, []):
            plt.plot(x, rho_eq_map[n], label=f"{n} (eq)", linestyle='--', color= color_map[role][n])
            plt.plot(x, rho_neq_map[n], label=f"{n} (neq)", linestyle='-', color= color_map[role][n])
        plt.xlabel( "x ")
        plt.ylabel( "rho ")
        plt.legend()

        plt.sca(axs[j,1])
        plt.title(f"Fluxes | role={role}")
        D_map = {s.name: s.D for s in spec.species}
        beta = 1.0  # assuming beta=1 for this comparison
        flux_eq = compute_fluxes_gummel_method(x, rho_eq_map, compute_potentials(x, rho_eq, spec), D_map, beta)
        flux_neq = compute_fluxes_gummel_method(x, rho_neq_map, compute_potentials(x, rho_neq, spec), D_map, beta)
        for n in by_role.get(role, []):
            x_edge_eq, J_eq = flux_eq[n]
            x_edge_neq, J_neq = flux_neq[n]
            plt.plot(x_edge_eq, J_eq, label=f"{n} (eq)", linestyle='--', color= color_map[role][n])
            plt.plot(x_edge_neq, J_neq, label=f"{n} (neq)", linestyle='-', color= color_map[role][n])

        plt.xlabel( "x (edge) ")
        plt.ylabel( "J ")
        plt.legend()

    plt.tight_layout()
    if show:
        plt.show()

    return

# kappa_x_values = [0.5, 1.0]
# results = {}
# for kappa_x in kappa_x_values:
#     print(f"[ kappa_x = {kappa_x} ]")
#     params['kappa_x'] = kappa_x
#     params['kappa_y'] = kappa_x  # keep them equal for simplicity
#     spec = spec_test(params)
#     rho_0, x = rd.initial_guess(params, spec)
    
#     rho_ss, info = rd.solve_steady_state(x, rho_0, spec, beta=params["beta"], max_iter=10000, tol=1e-6, verbose=False)
#     res = plotting.extract_results(x, rho_ss, spec, params['beta'])
#     results[kappa_x] = res

# filename = "kappa_x_experiment_results.pkl"
# with open(filename, 'wb') as f:
#     pickle.dump(results, f)
# {
#       'densities': {species: (x, rho)} ,
#       'fluxes':    {species: (x_edge, J_edge)} ,
#       'potentials':{species: (x, V)} ,
#       'barriers':  {(rxn, pi): barrier_array}
#     }

def plot_kappa_experiment(filename: str, show: bool = True):
    """
    Load data from a kappa experiment pickle file and plot the results.
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        
    kappa_x_values = sorted(data.keys())
    # separate clients and scaffolds and plot densities and fluxes
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for role in ('client', 'scaffold'):
        for quantity in ('densities', 'fluxes'):
            plt.sca(axs[0 if role=='client' else 1, 0 if quantity=='densities' else 1])
            for kappa_x in kappa_x_values:
                res = data[kappa_x]
                items = res[quantity]
                for species, (x, arr) in items.items():
                    # check species role
                    spec_role = None
                    for s in res['densities'].keys():
                        if s == species:
                            spec_role = 'client' if 'a' in s or 'b' in s or 'c' in s else 'scaffold'
                            break
                    if spec_role != role:
                        continue
                    label = f"{species} (kappa_x={kappa_x})"
                    if quantity == 'densities':
                        plt.plot(x, arr, label=label)
                    elif quantity == 'fluxes':
                        x_edge, J_edge = arr
                        plt.plot(x_edge, J_edge, label=label)
            plt.title(f"{quantity.capitalize()} | role={role}")
            plt.xlabel("x" if quantity=='densities' else "x (edge)")
            plt.ylabel("rho" if quantity=='densities' else "J")
            plt.legend()
    
    if show:
        plt.show()    