import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple, Any
from scipy import sparse as sp
import scipy.sparse.linalg as spla
from diffusion_drift_operators import build_scharfetter_gummel_operator, laplacian_1d_nonuniform
import rates
from collections import defaultdict, deque

@dataclass
class Species:
    name: str
    D: float = 1.0
    kappa: float = 0.0
    conserve_mass: Optional[bool] = False 
    norm_row_offset: int = 3
    role: Optional[str] = None # 'client' or 'scaffold'
    total: Optional[float] = None   # <-- initial mass / amount to conserve 

@dataclass
class BarrierFn:
    expr: str   
    params: Dict[str, float] = field(default_factory=dict)

@dataclass
class ReactionPathway:
    barrier: 'BarrierFn'
    mu: float = 0.0

@dataclass
class Reaction:
    name: str
    source: str
    target: str
    k0: float = 1.0
    pathways: List['ReactionPathway'] = field(default_factory=list)

@dataclass
class Geometry:
    type: str = 'grid1d'
    bc: str = 'neumann'

@dataclass
class ModelSpec:
    species: List[Species]                                      # list of species in the model
    eps: Dict[Tuple[str, str], float]                           # interaction parameters between species
    geometry: Geometry = field(default_factory=Geometry)        # geometry of the system
    reactions: List[Reaction] = field(default_factory=list)     # list of reactions
    shift_potentials_min_to_zero: bool = True                   # whether to shift potentials so that their minimum is zero
    intermediate_species: List[str] = field(default_factory=list)  # species that are fast decaying intermediates
    groups: List[List[str]] = field(default_factory=list)       # list of species groups for mass conservation
    auto_groups: bool = True        # <-- build groups from reactions if groups not given

def _safe_eval_barrier(expr:str, V:Dict[str, np.ndarray], x:np.ndarray, params:Dict[str,float]) -> np.ndarray:
    safe_globals = {'np': np, '__builtins__': {}}
    safe_locals = {'V': V, 'x': x}
    safe_locals.update(params)
    out = eval(expr, safe_globals, safe_locals)
    if not isinstance(out, np.ndarray):
        out = np.asarray(out) * np.ones_like(x)
    return out

def _weights_trapz(x: np.ndarray) -> np.ndarray:
    dx = np.diff(x)
    w = np.zeros_like(x)
    if len(x) < 2:
        return np.ones_like(x)
    w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    w[0] = 0.5 * dx[0]
    w[-1] = 0.5 * dx[-1]
    return w

def _species_index(spec: 'ModelSpec') -> Dict[str,int]:
    return {s.name: i for i,s in enumerate(spec.species)}

def compute_potentials(x: np.ndarray, rho: np.ndarray, spec: 'ModelSpec') -> Dict[str, np.ndarray]:
    nx = len(x)
    idx = _species_index(spec)
    V: Dict[str, np.ndarray] = {}
    laps = {}
    for s in spec.species:
        if s.kappa != 0:
            laps[s.name] = laplacian_1d_nonuniform(x) @ rho[idx[s.name]]
        else:
            laps[s.name] = np.zeros(nx)
    for si in spec.species:
        acc = np.zeros(nx, dtype=float)
        for sj in spec.species:
            # interactions are symmetric so we only give one value per pair. this implementation doesn't care about order
            eps_ij = spec.eps.get((si.name, sj.name), 0.0) + spec.eps.get((sj.name, si.name), 0.0)
            if si == sj:
                eps_ij *= 0.5
            if eps_ij != 0.0:
                acc += 2.0 * eps_ij * rho[idx[sj.name]]
        acc -= si.kappa * laps[si.name]
        V[si.name] = acc
    if spec.shift_potentials_min_to_zero:
        m = np.min([np.min(V[s.name]) for s in spec.species])
        for k in V:
            V[k] = V[k] - m
    # if there is an interemediate species, set its potential above others
    for sname in spec.intermediate_species:
        if sname in V:
            V[sname] = V[sname] + 1.5 * np.max([np.max(V[sn]) for sn in V])
    return V

def compute_reaction_rates(x: np.ndarray, V: Dict[str, np.ndarray], rho: np.ndarray, spec: 'ModelSpec', beta: float) -> Dict[Tuple[str,str], np.ndarray]:
    idx = _species_index(spec)
    k: Dict[Tuple[str,str], np.ndarray] = {}
    for r in spec.reactions:
        if len(r.pathways) == 0:
            raise ValueError(f'Reaction {r.name} must specify at least one pathway.')
        if len(r.pathways) > 2:
            raise ValueError(f'Reaction {r.name} currently supports up to 2 pathways (got {len(r.pathways)}).')
        barrier_I = _safe_eval_barrier(r.pathways[0].barrier.expr, V, x, r.pathways[0].barrier.params)
        mu_I = r.pathways[0].mu
        if len(r.pathways) == 2:
            barrier_II = _safe_eval_barrier(r.pathways[1].barrier.expr, V, x, r.pathways[1].barrier.params)
            mu_II = r.pathways[1].mu
        else:
            barrier_II = None
            mu_II = None
        
        kval = rates.compute_rate_2_pathways(k0=r.k0, V_from=V[r.source], V_barrier_I=barrier_I, V_barrier_II=barrier_II, mu_I=mu_I, mu_II=mu_II, beta=beta)
        kval[0] = 0.0; kval[-1] = 0.0  # boundary
        k[(r.source, r.target)] = kval
    return k


def infer_groups_from_reactions(spec: 'ModelSpec') -> List[List[str]]:
    idx = _species_index(spec)

    # mark species that participate in at least one ACTIVE reaction (k0 != 0)
    print('Inferring groups from reactions...')
    involved = {s.name: False for s in spec.species}
    for r in spec.reactions:
        if r.k0 != 0:
            if r.source in involved: involved[r.source] = True
            if r.target in involved: involved[r.target] = True

    # --- requested change: if a species is not involved in any reaction at all, enforce conserve_mass = True
    for s in spec.species:
        if not involved.get(s.name, False):
            s.conserve_mass = True

    # build the set of conserved, non-intermediate species to consider in grouping
    S = {s.name for s in spec.species}

    # build adjacency using only active reactions among S
    adj = defaultdict(set)
    for r in spec.reactions:
        if r.k0 == 0:
            continue
        if r.source in S and r.target in S:
            adj[r.source].add(r.target)
            adj[r.target].add(r.source)
            
    #print adjacency for debugging
    for k in adj:
        print(f'Adjacency for {k}: {adj[k]}')

    # connected components → groups (size > 1)
    seen = set()
    groups = []
    for s in S:
        if s in seen:
            continue
        comp = []
        q = deque([s]); seen.add(s)
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); q.append(v)
        if len(comp) > 1:
            groups.append(sorted(comp, key=lambda n: idx[n]))

    return groups



def finalize_groups(spec: 'ModelSpec') -> None:
    if spec.groups:  # user-specified takes precedence
        return
    if not spec.auto_groups:
        return
    inferred = infer_groups_from_reactions(spec)
    spec.groups = inferred
    
def build_totals_from_spec(spec: 'ModelSpec') -> Dict[str, float]:
    finalize_groups(spec)
    singles = {s.name for s in spec.species if s.conserve_mass}
    for g in spec.groups:
        for name in g:
            if name in singles:
                singles.remove(name)
    totals: Dict[str, float] = {}
    # singles
    for s in spec.species:
        if s.name in singles:
            totals[s.name] = float(0.0 if s.total is None else s.total)
    # groups
    for gi, g in enumerate(spec.groups):
        val = 0.0
        for name in g:
            spc = next(z for z in spec.species if z.name == name)
            val += float(0.0 if spc.total is None else spc.total)
        totals[f'group:{gi}'] = val
    return totals


def assemble_operator(x: np.ndarray, rho: np.ndarray, spec: 'ModelSpec', beta: float):
    nx = len(x)
    idx = _species_index(spec)
    nsp = len(spec.species)
    V = compute_potentials(x, rho, spec)
    A_blocks: List[sp.csr_matrix] = []
    for s in spec.species:
        A = build_scharfetter_gummel_operator(x, V[s.name], D=s.D, beta=beta)
        A_blocks.append(A)
    kmap = compute_reaction_rates(x, V, rho, spec, beta=beta)
    empty = sp.csr_matrix((nx, nx))
    blocks = [[empty for _ in range(nsp)] for __ in range(nsp)]
    for i in range(nsp):
        blocks[i][i] = A_blocks[i].copy()
    out_accum = {s.name: np.zeros(nx) for s in spec.species}
    for (src, tgt), kval in kmap.items():
        i = idx[src]; j = idx[tgt]
        blocks[j][i] = blocks[j][i] + sp.diags(kval, 0, format='csr')
        out_accum[src] += kval
    for s in spec.species:
        i = idx[s.name]
        if np.any(out_accum[s.name] != 0):
            blocks[i][i] = blocks[i][i] - sp.diags(out_accum[s.name], 0, format='csr')
    M = sp.bmat(blocks, format='csr')
    norm_rows: Dict[str,int] = {}
    weights = _weights_trapz(x)
    M_lil = M.tolil()
    for s in spec.species:
        if not s.conserve_mass:
            continue
        row = (idx[s.name] + 1) * nx - s.norm_row_offset
        norm_rows[s.name] = row
        start = idx[s.name] * nx; end = (idx[s.name] + 1) * nx
        M_lil[row, :] = 0
        M_lil[row, start:end] = weights
    for g, group in enumerate(spec.groups):
        last = group[-1]
        row = (idx[last] + 1) * nx - 3
        norm_rows[f'group:{g}'] = row
        M_lil[row, :] = 0
        for sname in group:
            start = idx[sname] * nx; end = (idx[sname] + 1) * nx
            M_lil[row, start:end] = weights
        for sname in group:
            if sname in norm_rows:
                r = norm_rows[sname]
                M_lil[r, :] = 0
    M = M_lil.tocsr()
    return M, V, norm_rows

def print_species(spec: 'ModelSpec') -> None:
    print('Species in the model:')
    for s in spec.species:
        print(f' - {s.name}:\t D={s.D}, \tkappa={s.kappa}, \tconserve_mass={s.conserve_mass}, \trole={s.role}, \ttotal={s.total}, \tgrouped={any(s.name in g for g in spec.groups)}')


def solve_steady_state(x: np.ndarray, rho_init: np.ndarray, spec: 'ModelSpec',
                       totals: Optional[Dict[str, float]] = None,
                       beta: float = 1.0, max_iter:int = 5000, tol: float = 1e-6,
                       verbose: bool = True, interval: int = 100):
    nx = len(x)
    nsp = len(spec.species)
    assert rho_init.shape == (nsp, nx)

    # auto-groups if needed (so assemble_operator sees the same groups)
    finalize_groups(spec)
    # print all initial species info
    print_species(spec)


    # auto-totals if not provided
    if totals is None:
        totals = build_totals_from_spec(spec)

    RHS = np.zeros(nsp * nx)
    for i in range(nsp):
        RHS[i*nx] = 0.0
        RHS[(i+1)*nx - 1] = 0.0

    rho_old = rho_init.copy().astype(float)
    M_prev, _, norm_rows = assemble_operator(x, rho_old, spec, beta=beta)

    def fill_RHS(RHS_vec, norm_rows, totals):
        for s in spec.species:
            if s.name in norm_rows and s.conserve_mass:
                RHS_vec[norm_rows[s.name]] = totals.get(s.name, 0.0)
        for gk, row in norm_rows.items():
            if gk.startswith('group:'):
                RHS_vec[row] = totals.get(gk, 0.0)
        return RHS_vec

    RHS = fill_RHS(RHS, norm_rows, totals)

    epsilon = np.random.rand(max_iter)
    final_info = {'converged': False, 'message': '', 'final_diff': None, 'iter': max_iter, 'final_masses': []}
    diffs = []

    for it in range(max_iter):
        M_cur, _, norm_rows = assemble_operator(x, rho_old, spec, beta=beta)
        RHS = fill_RHS(RHS, norm_rows, totals)
        mix = np.exp(-epsilon[it])
        M_mix = (M_cur + mix * M_prev) / (1.0 + mix)
        try:
            sol = spla.spsolve(M_mix, RHS)
        except Exception as e:
            final_info.update({'converged': False, 'message': f'Linear solve failed at iter {it}: {e}', 'final_diff': np.inf, 'iter': it,
                               'final_masses': [(s.name, float(np.dot(_weights_trapz(x), rho_old[i]))) for i,s in enumerate(spec.species)]
                               })
            return rho_old, final_info
        rho_new = sol.reshape((nsp, nx))
        diff = np.linalg.norm(rho_new - rho_old) / max(1e-12, np.linalg.norm(rho_old))
        diffs.append(diff)
        if verbose and it % interval == 0:
            w = _weights_trapz(x)
            masses = [(s.name, float(np.dot(w, rho_new[i]))) for i,s in enumerate(spec.species)]
            print('Iter {:5d}: diff={:.3e}; '.format(it, diff) + ', '.join([f'N_{n}={m:.3f}' for n,m in masses]))
        if len(diffs) > 50 and np.mean(diffs[-10:]) < tol:
            final_info.update({'converged': True, 'message': 'Converged by rolling mean of last 10 diffs.', 'final_diff': float(np.mean(diffs[-10:])), 'iter': it, 'final_masses': masses})
            return rho_new, final_info
        if np.any(np.isnan(rho_new)) or np.any(rho_new < -1e-8):
            final_info.update({'converged': False, 'message': 'NaN or negative density encountered.', 'final_diff': float(diff), 'iter': it, 'final_masses': masses})
            return rho_old, final_info
        rho_old = rho_new
        M_prev = M_mix

    final_info.update({'converged': False, 'message': 'Max iterations reached.',
                       'final_diff': float(diffs[-1] if diffs else np.inf), 'iter': max_iter,
                       'final_masses': [(s.name, float(np.dot(_weights_trapz(x), rho_old[i]))) for i,s in enumerate(spec.species)]
                       })
    return rho_old, final_info


# def solve_steady_state(x: np.ndarray, rho_init: np.ndarray, spec: 'ModelSpec', totals: Dict[str, float], beta: float = 1.0, max_iter:int = 5000, tol: float = 1e-6, verbose: bool = True, interval: int = 100):
    nx = len(x)
    nsp = len(spec.species)
    assert rho_init.shape == (nsp, nx)
    RHS = np.zeros(nsp * nx)
    for i in range(nsp):
        RHS[i*nx] = 0.0
        RHS[(i+1)*nx - 1] = 0.0
    rho_old = rho_init.copy().astype(float)
    M_prev, _, norm_rows = assemble_operator(x, rho_old, spec, beta=beta)
    def fill_RHS(RHS_vec, norm_rows, totals):
        for s in spec.species:
            if s.name in norm_rows and s.conserve_mass:
                RHS_vec[norm_rows[s.name]] = totals.get(s.name, 0.0)
        for gk, row in norm_rows.items():
            if gk.startswith('group:'):
                RHS_vec[row] = totals.get(gk, 0.0)
        return RHS_vec
    RHS = fill_RHS(RHS, norm_rows, totals)
    epsilon = np.random.rand(max_iter)
    final_info = {'converged': False, 'message': '', 'final_diff': None, 'iter': max_iter, 'final_masses': []}
    diffs = []
    for it in range(max_iter):
        M_cur, _, norm_rows = assemble_operator(x, rho_old, spec, beta=beta)
        RHS = fill_RHS(RHS, norm_rows, totals)
        mix = np.exp(-epsilon[it])
        M_mix = (M_cur + mix * M_prev) / (1.0 + mix)
        try:
            sol = spla.spsolve(M_mix, RHS)
        except Exception as e:
            final_info.update({'converged': False, 'message': f'Linear solve failed at iter {it}: {e}', 'final_diff': np.inf, 'iter': it,
                               'final_masses': [(s.name, float(np.dot(_weights_trapz(x), rho_old[i]))) for i,s in enumerate(spec.species)]
                               })
            return rho_old, final_info
        rho_new = sol.reshape((nsp, nx))
        diff = np.linalg.norm(rho_new - rho_old) / max(1e-12, np.linalg.norm(rho_old))
        diffs.append(diff)
        if verbose and it % interval == 0:
            w = _weights_trapz(x)
            masses = [(s.name, float(np.dot(w, rho_new[i]))) for i,s in enumerate(spec.species)]
            print('Iter {:5d}: diff={:.3e}; '.format(it, diff) + ', '.join([f'N_{n}={m:.3f}' for n,m in masses]))
        if len(diffs) > 50 and np.mean(diffs[-10:]) < tol:
            final_info.update({'converged': True, 'message': 'Converged by rolling mean of last 10 diffs.', 'final_diff': float(np.mean(diffs[-10:])), 'iter': it, 'final_masses': masses})
            return rho_new, final_info
        if np.any(np.isnan(rho_new)) or np.any(rho_new < -1e-8):
            final_info.update({'converged': False, 'message': 'NaN or negative density encountered.', 'final_diff': float(diff), 'iter': it, 'final_masses': masses})
            return rho_old, final_info
        rho_old = rho_new
        M_prev = M_mix
    final_info.update({'converged': False, 'message': 'Max iterations reached.', 
                       'final_diff': float(diffs[-1] if diffs else np.inf), 'iter': max_iter,
                       'final_masses': [(s.name, float(np.dot(_weights_trapz(x), rho_old[i]))) for i,s in enumerate(spec.species)]
                       })

    return rho_old, final_info