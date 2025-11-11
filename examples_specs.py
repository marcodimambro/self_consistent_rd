import numpy as np
from rd_framework import ModelSpec, Species, Reaction, ReactionPathway, BarrierFn



def spec_test(params: dict) -> ModelSpec:
    
    sp = [
        Species('a', D=params.get('D',1.0), kappa=0.0, role='client', total =params.get('N_a',1.0)),
        Species('b', D=params.get('D',1.0), kappa=0.0, role='client', total =params.get('N_b',1.0)),
        Species('c', D=params.get('D',1.0), kappa=0.0, role='client', total =params.get('N_c',1.0)),
        Species('x', D=params.get('D',1.0), kappa=params.get('kappa_x',0.0), role='scaffold', total =params.get('N_x',1500.0)),
        Species('y', D=params.get('D',1.0), kappa=params.get('kappa_y',0.0), role='scaffold', total =params.get('N_y',1500.0)),
        #Species('p', D=params.get('D',1.0), kappa=0.0, role='scaffold', total = params.get('N_p',1.0)),
        #Species('k', D=params.get('D',1.0), kappa=0.0, role='scaffold', total = params.get('N_k',1.0))
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
    
    # here define functional shape of barriers
    ab = BarrierFn(expr="alpha * V['x'] + gamma* np.array([V['a'].max(), V['b'].max()]).max()", params={'alpha': params.get('alpha_ab',1.0), 'gamma': params.get('gamma_ab',1.0)})
    ac = BarrierFn(expr="alpha * V['x'] + gamma* np.array([V['a'].max(), V['c'].max()]).max()", params={'alpha': params.get('alpha_ac',1.0), 'gamma': params.get('gamma_ac',1.0)})
    bc = BarrierFn(expr="alpha * V['x'] + gamma* np.array([V['b'].max(), V['c'].max()]).max()", params={'alpha': params.get('alpha_bc',1.0), 'gamma': params.get('gamma_bc',1.0)})
    xyI = BarrierFn(expr="alpha * ((beta)*V['x']+(1 - beta) * V['y']) + gamma* np.array([V['x'].max(), V['y'].max()]).max()", params={'alpha': params.get('alpha_xy_I',1.2), 'beta': params.get('beta_xy_I',0.5), 'gamma': params.get('gamma_xy_I',1.0)})
    xyII = BarrierFn(expr="alpha *((beta)*V['x']+(1 - beta) * V['y']) + gamma* np.array([V['x'].max(), V['y'].max()]).max()", params={'alpha': params.get('alpha_xy_II',1.5), 'beta': params.get('beta_xy_II',0.5), 'gamma': params.get('gamma_xy_II',1.0)})
    #yI = BarrierFn(expr="alpha * (V['a']) + gamma* np.array([V['x'].max(), V['y'].max()]).max()", params={'alpha': params.get('alpha_xy_I',1.2), 'beta': params.get('beta_xy_I',0.5), 'gamma': params.get('gamma_xy_I',1.0)})
    #yII = BarrierFn(expr="alpha *(V['b']) + gamma* np.array([V['x'].max(), V['y'].max()]).max()", params={'alpha': params.get('alpha_xy_II',1.5), 'beta': params.get('beta_xy_II',0.5), 'gamma': params.get('gamma_xy_II',1.0)})


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


def spec_complete(params: dict) -> ModelSpec:
    
    sp = [
        Species('a', D=params.get('D',1.0), kappa=0.0, role='client', total =params.get('N_a',1.0)),
        Species('b', D=params.get('D',1.0), kappa=0.0, role='client', total =params.get('N_b',1.0)),
        Species('c', D=params.get('D',1.0), kappa=0.0, role='client', total =params.get('N_c',1.0)),
        Species('x', D=params.get('D',1.0), kappa=params.get('kappa_x',0.0), role='scaffold', total =params.get('N_x',1500.0)),
        Species('y', D=params.get('D',1.0), kappa=params.get('kappa_y',0.0), role='scaffold', total =params.get('N_y',1500.0)),
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
        ('c','y'): params['eps_cy']
    }
    
    # here define functional shape of barriers
    ab = BarrierFn(expr="alpha * V['x'] + gamma* np.array([V['a'].max(), V['b'].max()]).max()", params={'alpha': params.get('alpha_ab',1.0), 'gamma': params.get('gamma_ab',1.0)})
    ac = BarrierFn(expr="alpha * V['x'] + gamma* np.array([V['a'].max(), V['c'].max()]).max()", params={'alpha': params.get('alpha_ac',1.0), 'gamma': params.get('gamma_ac',1.0)})
    bc = BarrierFn(expr="alpha * V['x'] + gamma* np.array([V['b'].max(), V['c'].max()]).max()", params={'alpha': params.get('alpha_bc',1.0), 'gamma': params.get('gamma_bc',1.0)})
    xyI = BarrierFn(expr="alpha * ((beta)*V['x']+(1 - beta) * V['y']) + gamma* np.array([V['x'].max(), V['y'].max()]).max()", params={'alpha': params.get('alpha_xy_I',1.2), 'beta': params.get('beta_xy_I',0.5), 'gamma': params.get('gamma_xy_I',1.0)})
    xyII = BarrierFn(expr="alpha *((beta)*V['x']+(1 - beta) * V['y']) + gamma* np.array([V['x'].max(), V['y'].max()]).max()", params={'alpha': params.get('alpha_xy_II',1.5), 'beta': params.get('beta_xy_II',0.5), 'gamma': params.get('gamma_xy_II',1.0)})
    # xyI = BarrierFn(expr="alpha * (V['a']) + gamma* np.array([V['x'].max(), V['y'].max()]).max()", params={'alpha': params.get('alpha_xy_I',1.2), 'beta': params.get('beta_xy_I',0.5), 'gamma': params.get('gamma_xy_I',1.0)})
    # xyII = BarrierFn(expr="alpha *(V['b']) + gamma* np.array([V['x'].max(), V['y'].max()]).max()", params={'alpha': params.get('alpha_xy_II',1.5), 'beta': params.get('beta_xy_II',0.5), 'gamma': params.get('gamma_xy_II',1.0)})


    rxns = [Reaction(name='a_to_b', source='a', target='b', k0=params.get('k0_ab',1.0), pathways=[ReactionPathway(barrier=ab, mu =params.get('mu_ab',0.0))]),
            Reaction(name='b_to_a', source='b', target='a', k0=params.get('k0_ab',1.0), pathways=[ReactionPathway(barrier=ab, mu =params.get('mu_ba',0.0))]),
            Reaction(name='a_to_c', source='a', target='c', k0=params.get('k0_ac',1.0), pathways=[ReactionPathway(barrier=ac, mu =params.get('mu_ac',0.0))]),
            Reaction(name='c_to_a', source='c', target='a', k0=params.get('k0_ac',1.0), pathways=[ReactionPathway(barrier=ac, mu =params.get('mu_ca',0.0))]),
            Reaction(name='b_to_c', source='b', target='c', k0=params.get('k0_bc',1.0), pathways=[ReactionPathway(barrier=bc, mu =params.get('mu_bc',0.0))]),
            Reaction(name='c_to_b', source='c', target='b', k0=params.get('k0_bc',1.0), pathways=[ReactionPathway(barrier=bc, mu =params.get('mu_cb',0.0))]),
            Reaction(name='x_to_y', source='x', target='y', k0=params.get('k0_scaffold',1.0), pathways=[ReactionPathway(barrier=xyI, mu=params.get('mu_xy_I',0.0)), ReactionPathway(barrier=xyII, mu=params.get('mu_xy_II',0.0))]),
            Reaction(name='y_to_x', source='y', target='x', k0=params.get('k0_scaffold',1.0), pathways=[ReactionPathway(barrier=xyI, mu=params.get('mu_yx_I',0.0)), ReactionPathway(barrier=xyII, mu=params.get('mu_yx_II',0.0))]),
            ]
            
    return ModelSpec(species=sp, eps=eps, reactions=rxns)

