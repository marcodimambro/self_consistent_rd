# self_consistent_rd


Reaction-diffusion formalism for general chemical reaction networks (CRNs) undergoing phase separation. 
A model is defined as a "ModelSpec" class object with entries:
- species: list of chemical states 
- eps: dictionary of energetic interactions (both self- and cross-)
- reactions: list of reactions between states

Reactions are assumed to be activation processes and it is necessary to specify the functional form of the energetic barrier between two states (see examples_spec.py for an example).

The set-up can sustain an arbitrary number of states and different CNR topologies. The adjacency matrix and corresponding normalization contraints are derived from the reactions in the system. 