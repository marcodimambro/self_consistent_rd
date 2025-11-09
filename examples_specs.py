import numpy as np
from rd_framework import ModelSpec, Species, Reaction, ReactionPathway, BarrierFn

def spec_barducci(params: dict) -> ModelSpec:
    sp = [
        Species('a', D=params.get('D',1.0), kappa=params.get('kappa_a',0.0), role='client'),
        Species('b', D=params.get('D',1.0), kappa=params.get('kappa_b',0.0), role='client'),
        Species('p', D=params.get('D',1.0), kappa=0.0, role='enzyme'),
        Species('k', D=params.get('D',1.0), kappa=0.0, role='enzyme'),
    ]
    eps = {
        ('a','a'): params['eps_a'],
        ('b','b'): params['eps_b'],
        ('a','b'): params['eps_ab'],
        ('b','a'): params['eps_ab'],
        ('a','p'): params['eps_ap'],
        ('a','k'): params['eps_ak'],
        ('b','p'): params['eps_bp'],
        ('b','k'): params['eps_bk'],
        ('p','a'): params['eps_ap'],
        ('k','a'): params['eps_ak'],
        ('p','b'): params['eps_bp'],
        ('k','b'): params['eps_bk'],
    }
    alpha_I = params.get('alpha_ab_I',1.2); gamma_I = params.get('gamma_ab_I',1.0)
    alpha_II = params.get('alpha_ab_II',1.5); gamma_II = params.get('gamma_ab_II',1.0)
    bI = BarrierFn(expr="alpha*V['p'] + gamma*max(V['a'].max(), V['b'].max())", params={'alpha':alpha_I, 'gamma':gamma_I})
    bII = BarrierFn(expr="gamma*max(V['a'].max(), V['b'].max())", params={'gamma':gamma_II})
    rxns = [
        Reaction(name='a_to_b', source='a', target='b', k0=params.get('k0_ab',1.0),
                 pathways=[ReactionPathway(barrier=bI, mu=params.get('mu_ab_I',0.0)),
                           ReactionPathway(barrier=bII, mu=params.get('mu_ab_II',0.0))],
                 enzyme_factor='a'),
        Reaction(name='b_to_a', source='b', target='a', k0=params.get('k0_ab',1.0),
                 pathways=[ReactionPathway(barrier=bI, mu=params.get('mu_ba_I',0.0)),
                           ReactionPathway(barrier=bII, mu=params.get('mu_ba_II',0.0))],
                 enzyme_factor='a'),
    ]
    groups = [['a','b']] if params.get('use_group_ab', True) and params.get('k0_ab',1.0) != 0 else []
    return ModelSpec(species=sp, eps=eps, reactions=rxns, groups=groups)

def spec_scaffold(params: dict) -> ModelSpec:
    sp = [
        Species('a', D=params.get('D',1.0), kappa=params.get('kappa_a',0.0), role='client'),
        Species('b', D=params.get('D',1.0), kappa=params.get('kappa_b',0.0), role='client'),
        Species('p', D=params.get('D',1.0), kappa=0.0, role='enzyme'),
    ]
    eps = {
        ('a','a'): params['eps_a'],
        ('b','b'): params['eps_b'],
        ('a','b'): params['eps_ab'],
        ('b','a'): params['eps_ab'],
        ('p','a'): params['eps_ap'],
        ('p','b'): params['eps_bp'],
        ('a','p'): params['eps_ap'],
        ('b','p'): params['eps_bp'],
    }
    alpha_I = params.get('alpha_ab_I',1.2); gamma_I = params.get('gamma_ab_I',1.0)
    alpha_II = params.get('alpha_ab_II',1.5); gamma_II = params.get('gamma_ab_II',1.0)
    bI = BarrierFn(expr="alpha*V['p'] + gamma*max(V['a'].max(), V['b'].max())", params={'alpha':alpha_I, 'gamma':gamma_I})
    bII = BarrierFn(expr="gamma*max(V['a'].max(), V['b'].max())", params={'gamma':gamma_II})
    rxns = [
        Reaction(name='a_to_b', source='a', target='b', k0=params.get('k0_ab',1.0),
                 pathways=[ReactionPathway(barrier=bI, mu=params.get('mu_ab_I',0.0)),
                           ReactionPathway(barrier=bII, mu=params.get('mu_ab_II',0.0))],
                 enzyme_factor='a'),
        Reaction(name='b_to_a', source='b', target='a', k0=params.get('k0_ab',1.0),
                 pathways=[ReactionPathway(barrier=bI, mu=params.get('mu_ba_I',0.0)),
                           ReactionPathway(barrier=bII, mu=params.get('mu_ba_II',0.0))],
                 enzyme_factor='a'),
    ]
    groups = [['a','b']] if params.get('k0_ab',1.0) != 0 else []
    return ModelSpec(species=sp, eps=eps, reactions=rxns, groups=groups)




def spec_complete(params: dict) -> ModelSpec:
    
    groups = []
    if params.get('k0_client',1.0) == 0 and params.get('k0_client',1.0) == 0:
        conserve_client = True
        conserve_scaffold = True
        totals = {'a': params.get('N_a',1.0), 'b': params.get('N_b',1.0), 'c': params.get('N_c',1.0),
                  'x': params.get('N_x',1500.0), 'y': params.get('N_y',1500.0)
                  }
    elif params.get('k0_client',1.0) == 0:
        conserve_client = True
        totals = {'a': params.get('N_a',1.0), 'b': params.get('N_b',1.0), 'c': params.get('N_c',1.0),
                  'group:1': params.get('N_x',1500.0) + params.get('N_y',1500.0)
                  }
    elif params.get('k0_scaffold',1.0) == 0:
        conserve_scaffold = True
        totals = {'x': params.get('N_x',1500.0), 'y': params.get('N_y',1500.0),
                  'group:0': params.get('N_a',1.0) + params.get('N_b',1.0) + params.get('N_c',1.0)
                  }
    else:
        totals = {
                  'group:0': params.get('N_a',1.0) + params.get('N_b',1.0) + params.get('N_c',1.0),
                  'group:1': params.get('N_x',1500.0) + params.get('N_y',1500.0)
                  }
    sp = [
        Species('a', D=params.get('D',1.0), kappa=0.0, role='client', conserve_mass=conserve_client if 'conserve_client' in locals() else False),
        Species('b', D=params.get('D',1.0), kappa=0.0, role='client', conserve_mass=conserve_client if 'conserve_client' in locals() else False),
        Species('c', D=params.get('D',1.0), kappa=0.0, role='client', conserve_mass=conserve_client if 'conserve_client' in locals() else False),
        Species('x', D=params.get('D',1.0), kappa=params.get('kappa_x',0.0), role='scaffold', conserve_mass=conserve_scaffold if 'conserve_scaffold' in locals() else False),
        Species('y', D=params.get('D',1.0), kappa=params.get('kappa_y',0.0), role='scaffold', conserve_mass=conserve_scaffold if 'conserve_scaffold' in locals() else False),
        #Species('p', D=params.get('D',1.0), kappa=0.0, role='scaffold', conserve_mass= True),
        #Species('k', D=params.get('D',1.0), kappa=0.0, role='scaffold', conserve_mass= True),
    ]
    eps = {
        ('x','x'): params['eps_x'], 
        ('y','y'): params['eps_y'],
        ('x','y'): params['eps_xy'], 
        ('a','x'): params['eps_ax'], 
        ('b','x'): params['eps_bx'], 
        ('c','x'): params['eps_cx'], 
        ('x','a'): params['eps_ax'], 
        ('x','b'): params['eps_bx'], 
        ('x','c'): params['eps_cx'], 
        ('p', 'x'): params['eps_px'],
        ('k', 'x'): params['eps_kx'],
        ('p', 'y'): params['eps_py'],
        ('k', 'y'): params['eps_ky'],
    }
    ab = BarrierFn(expr="alpha * V['x'] + gamma* np.array([V['a'].max(), V['b'].max()]).max()", params={'alpha': params.get('alpha_ab',1.0), 'gamma': params.get('gamma_ab',1.0)})
    ac = BarrierFn(expr="alpha * V['x'] + gamma* np.array([V['a'].max(), V['c'].max()]).max()", params={'alpha': params.get('alpha_ac',1.0), 'gamma': params.get('gamma_ac',1.0)})
    bc = BarrierFn(expr="alpha * V['x'] + gamma* np.array([V['b'].max(), V['c'].max()]).max()", params={'alpha': params.get('alpha_bc',1.0), 'gamma': params.get('gamma_bc',1.0)})
    xyI = BarrierFn(expr="alpha * V['x'] + gamma* np.array([V['x'].max(), V['y'].max()]).max()", params={'alpha': params.get('alpha_xy_I',1.2), 'gamma': params.get('gamma_xy_I',1.0)})
    xyII = BarrierFn(expr="alpha * V['y'] + gamma* np.array([V['x'].max(), V['y'].max()]).max()", params={'alpha': params.get('alpha_xy_II',1.5), 'gamma': params.get('gamma_xy_II',1.0)})
    
    rxns = []
    if params.get('k0_client',1.0) :
        rxns.extend([Reaction(name='a_to_b', source='a', target='b', k0=params.get('k0_client',1.0), pathways=[ReactionPathway(barrier=ab, mu =params.get('mu_ab',0.0))]),
                    Reaction(name='b_to_a', source='b', target='a', k0=params.get('k0_client',1.0), pathways=[ReactionPathway(barrier=ab, mu =params.get('mu_ba',0.0))]),
                    Reaction(name='a_to_c', source='a', target='c', k0=params.get('k0_client',1.0), pathways=[ReactionPathway(barrier=ac, mu =params.get('mu_ac',0.0))]),
                    Reaction(name='c_to_a', source='c', target='a', k0=params.get('k0_client',1.0), pathways=[ReactionPathway(barrier=ac, mu =params.get('mu_ca',0.0))]),
                    Reaction(name='b_to_c', source='b', target='c', k0=params.get('k0_client',1.0), pathways=[ReactionPathway(barrier=bc, mu =params.get('mu_bc',0.0))]),
                    Reaction(name='c_to_b', source='c', target='b', k0=params.get('k0_client',1.0), pathways=[ReactionPathway(barrier=bc, mu =params.get('mu_cb',0.0))]),
            ])
    # if params.get('k0_scaffold',1.0):
    #     rxns.extend([Reaction(name='x_to_y', source='x', target='y', k0=params.get('k0_scaffold',1.0), pathways=[ReactionPathway(barrier=xyI, mu=params.get('mu_xy_I',0.0)), ReactionPathway(barrier=xyII, mu=params.get('mu_xy_II',0.0))]),
    #                  Reaction(name='y_to_x', source='y', target='x', k0=params.get('k0_scaffold',1.0), pathways=[ReactionPathway(barrier=xyI, mu=params.get('mu_yx_I',0.0)), ReactionPathway(barrier=xyII, mu=params.get('mu_yx_II',0.0))]),
    #         ])
    if params.get('k0_scaffold',1.0):
        rxns.extend([Reaction(name='x_to_y', source='x', target='y', k0=params.get('k0_scaffold',1.0), pathways=[ReactionPathway(barrier=xyI, mu=params.get('mu_xy_I',0.0))]),
                     Reaction(name='y_to_x', source='y', target='x', k0=params.get('k0_scaffold',1.0), pathways=[ReactionPathway(barrier=xyI, mu=params.get('mu_yx_I',0.0))]),
                     
            
            ])
    
    return ModelSpec(species=sp, eps=eps, reactions=rxns, groups=groups)



def spec_two_triangles(params: dict) -> ModelSpec:
    
    groups = []
    if params.get('k0_client',1.0) != 0:
        groups.append(['a','b','c'])
    else: # conserve mass
        conserve_client = True        
    if params.get('k0_scaffold',1.0) != 0:
        groups.append(['x','y', 'z'])
    else:
        conserve_scaffold = True
    sp = [
        Species('a', D=params.get('D',1.0), kappa=0.0, role='client', conserve_mass=conserve_client if 'conserve_client' in locals() else False),
        Species('b', D=params.get('D',1.0), kappa=0.0, role='client', conserve_mass=conserve_client if 'conserve_client' in locals() else False),
        Species('c', D=params.get('D',1.0), kappa=0.0, role='client', conserve_mass=conserve_client if 'conserve_client' in locals() else False),
        Species('x', D=params.get('D',1.0), kappa=params.get('kappa_x',0.0), role='scaffold', conserve_mass=conserve_scaffold if 'conserve_scaffold' in locals() else False),
        Species('y', D=params.get('D',1.0), kappa=params.get('kappa_y',0.0), role='scaffold', conserve_mass=conserve_scaffold if 'conserve_scaffold' in locals() else False),
        #Species('p', D=params.get('D',1.0), kappa=0.0, role='scaffold', conserve_mass= True),
        #Species('k', D=params.get('D',1.0), kappa=0.0, role='scaffold', conserve_mass= True),
        Species('z', D=params.get('D',1.0), kappa=params.get('kappa_z',0.0), role='scaffold', conserve_mass=conserve_scaffold if 'conserve_scaffold' in locals() else False),
    ]
    eps = {
        ('x','x'): params['eps_x'], 
        ('y','y'): params['eps_y'],
        ('x','y'): params['eps_xy'],
        ('y', 'y'): params['eps_y'],
        ('a','x'): params['eps_ax'], 
        ('b','x'): params['eps_bx'], 
        ('c','x'): params['eps_cx'], 
        ('a','y'): params['eps_ay'],
        ('b','y'): params['eps_by'],
        ('c','y'): params['eps_cy'],
        ('p', 'x'): params['eps_px'],
        ('k', 'x'): params['eps_kx'],
        ('p', 'y'): params['eps_py'],
        ('k', 'y'): params['eps_ky'],
    }
    ab = BarrierFn(expr="alpha * V['x'] + gamma* np.array([V['a'].max(), V['b'].max()]).max()", params={'alpha': params.get('alpha_ab',1.0), 'gamma': params.get('gamma_ab',1.0)})
    ac = BarrierFn(expr="alpha * V['x'] + gamma* np.array([V['a'].max(), V['c'].max()]).max()", params={'alpha': params.get('alpha_ac',1.0), 'gamma': params.get('gamma_ac',1.0)})
    bc = BarrierFn(expr="alpha * V['x'] + gamma* np.array([V['b'].max(), V['c'].max()]).max()", params={'alpha': params.get('alpha_bc',1.0), 'gamma': params.get('gamma_bc',1.0)})
    xy = BarrierFn(expr="alpha * (V['x'] +V['y'])/2  + gamma* np.array([V['x'].max(), V['y'].max()]).max()", params={'alpha': params.get('alpha_xy',1.2), 'gamma': params.get('gamma_xy',1.0)})
    xz = BarrierFn(expr="alpha * V['x'] + gamma* np.array([V['x'].max(), V['z'].max()]).max()", params={'alpha': params.get('alpha_xz',1.2), 'gamma': params.get('gamma_xz',1.0)})
    yz = BarrierFn(expr="alpha * V['y']+  gamma* np.array([V['y'].max(), V['z'].max()]).max()", params={'alpha': params.get('alpha_yz',1.2), 'gamma': params.get('gamma_yz',1.0)})
    
    rxns = []
    if params.get('k0_client',1.0) :
        rxns.extend([Reaction(name='a_to_b', source='a', target='b', k0=params.get('k0_client',1.0), pathways=[ReactionPathway(barrier=ab, mu =params.get('mu_ab',0.0))]),
                    Reaction(name='b_to_a', source='b', target='a', k0=params.get('k0_client',1.0), pathways=[ReactionPathway(barrier=ab, mu =params.get('mu_ba',0.0))]),
                    Reaction(name='a_to_c', source='a', target='c', k0=params.get('k0_client',1.0), pathways=[ReactionPathway(barrier=ac, mu =params.get('mu_ac',0.0))]),
                    Reaction(name='c_to_a', source='c', target='a', k0=params.get('k0_client',1.0), pathways=[ReactionPathway(barrier=ac, mu =params.get('mu_ca',0.0))]),
                    Reaction(name='b_to_c', source='b', target='c', k0=params.get('k0_client',1.0), pathways=[ReactionPathway(barrier=bc, mu =params.get('mu_bc',0.0))]),
                    Reaction(name='c_to_b', source='c', target='b', k0=params.get('k0_client',1.0), pathways=[ReactionPathway(barrier=bc, mu =params.get('mu_cb',0.0))]),
            ])
    # if params.get('k0_scaffold',1.0):
    #     rxns.extend([Reaction(name='x_to_y', source='x', target='y', k0=params.get('k0_scaffold',1.0), pathways=[ReactionPathway(barrier=xyI, mu=params.get('mu_xy_I',0.0)), ReactionPathway(barrier=xyII, mu=params.get('mu_xy_II',0.0))]),
    #                  Reaction(name='y_to_x', source='y', target='x', k0=params.get('k0_scaffold',1.0), pathways=[ReactionPathway(barrier=xyI, mu=params.get('mu_yx_I',0.0)), ReactionPathway(barrier=xyII, mu=params.get('mu_yx_II',0.0))]),
    #         ])
    if params.get('k0_scaffold',1.0):
        rxns.extend([Reaction(name='x_to_y', source='x', target='y', k0=params.get('k0_scaffold',1.0), pathways=[ReactionPathway(barrier=xy, mu=params.get('mu_xy',0.0))]),
                     Reaction(name='y_to_x', source='y', target='x', k0=params.get('k0_scaffold',1.0), pathways=[ReactionPathway(barrier=xy, mu=params.get('mu_yx',0.0))]),
                     Reaction(name='x_to_z', source='x', target='z', k0=params.get('k0_scaffold',1.0), pathways=[ReactionPathway(barrier=xz, mu=params.get('mu_xz',0.0))]),
                        Reaction(name='z_to_x', source='z', target='x', k0=params.get('k0_scaffold',1.0), pathways=[ReactionPathway(barrier=xz, mu=params.get('mu_zx',0.0))]),
                     Reaction(name='y_to_z', source='y', target='z', k0=params.get('k0_scaffold',1.0), pathways=[ReactionPathway(barrier=yz, mu=params.get('mu_yz',0.0))]),
                        Reaction(name='z_to_y', source='z', target='y', k0=params.get('k0_scaffold',1.0), pathways=[ReactionPathway(barrier=yz, mu=params.get('mu_zy',0.0))]),
            ])
    
    return ModelSpec(species=sp, eps=eps, reactions=rxns, groups=groups, intermediate_species=['z'], shift_potentials_min_to_zero=True)













def spec_test(params: dict) -> ModelSpec:
    
    sp = [
        Species('a', D=params.get('D',1.0), kappa=0.0, role='client', total =params.get('N_a',1.0)),
        Species('b', D=params.get('D',1.0), kappa=0.0, role='client', total =params.get('N_b',1.0)),
        Species('c', D=params.get('D',1.0), kappa=0.0, role='client', total =params.get('N_c',1.0)),
        Species('x', D=params.get('D',1.0), kappa=params.get('kappa_x',0.0), role='scaffold', total =params.get('N_x',1500.0)),
        Species('y', D=params.get('D',1.0), kappa=params.get('kappa_y',0.0), role='scaffold', total =params.get('N_y',1500.0)),
        #Species('p', D=params.get('D',1.0), kappa=0.0, role='scaffold', conserve_mass= True),
        #Species('k', D=params.get('D',1.0), kappa=0.0, role='scaffold', conserve_mass= True),
    ]
    eps = {
        ('x','x'): params['eps_x'], 
        ('y','y'): params['eps_y'],
        ('x','y'): params['eps_xy'], 
        ('a','x'): params['eps_ax'], 
        ('b','x'): params['eps_bx'], 
        ('c','x'): params['eps_cx'], 
        ('a','y'): params['eps_ay'], 
        ('b','y'): params['eps_by'], 
        ('c','y'): params['eps_cy'], 
        #('p', 'x'): params['eps_px'],
        #('k', 'x'): params['eps_kx'],
        #('p', 'y'): params['eps_py'],
        #('k', 'y'): params['eps_ky'],
    }
    ab = BarrierFn(expr="alpha * V['x'] + gamma* np.array([V['a'].max(), V['b'].max()]).max()", params={'alpha': params.get('alpha_ab',1.0), 'gamma': params.get('gamma_ab',1.0)})
    ac = BarrierFn(expr="alpha * V['x'] + gamma* np.array([V['a'].max(), V['c'].max()]).max()", params={'alpha': params.get('alpha_ac',1.0), 'gamma': params.get('gamma_ac',1.0)})
    bc = BarrierFn(expr="alpha * V['x'] + gamma* np.array([V['b'].max(), V['c'].max()]).max()", params={'alpha': params.get('alpha_bc',1.0), 'gamma': params.get('gamma_bc',1.0)})
    xyI = BarrierFn(expr="alpha * V['x'] + gamma* np.array([V['x'].max(), V['y'].max()]).max()", params={'alpha': params.get('alpha_xy_I',1.2), 'gamma': params.get('gamma_xy_I',1.0)})
    xyII = BarrierFn(expr="alpha * V['y'] + gamma* np.array([V['x'].max(), V['y'].max()]).max()", params={'alpha': params.get('alpha_xy_II',1.5), 'gamma': params.get('gamma_xy_II',1.0)})
    
    
    rxns = [Reaction(name='a_to_b', source='a', target='b', k0=params.get('k0_client',1.0), pathways=[ReactionPathway(barrier=ab, mu =params.get('mu_ab',0.0))]),
            Reaction(name='b_to_a', source='b', target='a', k0=params.get('k0_client',1.0), pathways=[ReactionPathway(barrier=ab, mu =params.get('mu_ba',0.0))]),
            Reaction(name='a_to_c', source='a', target='c', k0=params.get('k0_client',1.0), pathways=[ReactionPathway(barrier=ac, mu =params.get('mu_ac',0.0))]),
            Reaction(name='c_to_a', source='c', target='a', k0=params.get('k0_client',1.0), pathways=[ReactionPathway(barrier=ac, mu =params.get('mu_ca',0.0))]),
            Reaction(name='b_to_c', source='b', target='c', k0=params.get('k0_client',1.0), pathways=[ReactionPathway(barrier=bc, mu =params.get('mu_bc',0.0))]),
            Reaction(name='c_to_b', source='c', target='b', k0=params.get('k0_client',1.0), pathways=[ReactionPathway(barrier=bc, mu =params.get('mu_cb',0.0))]),
            Reaction(name='x_to_y', source='x', target='y', k0=params.get('k0_scaffold',1.0), pathways=[ReactionPathway(barrier=xyI, mu=params.get('mu_xy_I',0.0)), ReactionPathway(barrier=xyII, mu=params.get('mu_xy_II',0.0))]),
            Reaction(name='y_to_x', source='y', target='x', k0=params.get('k0_scaffold',1.0), pathways=[ReactionPathway(barrier=xyI, mu=params.get('mu_yx_I',0.0)), ReactionPathway(barrier=xyII, mu=params.get('mu_yx_II',0.0))]),
            ]
            
    return ModelSpec(species=sp, eps=eps, reactions=rxns)

