import numpy as np  
import freud
import gsd, gsd.hoomd 
import hoomd 
import time

def initialize_snapshot_rand_walk(num_pol, num_mon, density=0.85, bond_length=1.0, seed=1234):
    ''' 
    Create a HOOMD snapshot of a cubic box with the number density given by input parameters. Configure particles using a random walk. 

    '''
    rng = np.random.default_rng(seed)

    N = num_pol * num_mon
    L = np.cbrt(N / density)

    #replace chain loop with vectorized random walk
    positions = np.empty((N, 3))
    starts = rng.uniform(-L/2, L/2, size=(num_pol, 3))
    deltas = rng.normal(size=(num_pol, num_mon - 1, 3))
    deltas *= bond_length / np.linalg.norm(deltas, axis=2, keepdims=True)

    displacements = np.cumsum(deltas, axis=1)

    positions_view = positions.reshape(num_pol, num_mon, 3)
    positions_view[:, 0, :] = starts
    positions_view[:, 1:, :] = starts[:, None, :] + displacements

    #pbc
    positions += L/2
    positions %= L
    positions -= L/2

    # bonds (vectorized)
    indices = np.arange(N).reshape(num_pol, num_mon)
    bonds = np.column_stack([
        indices[:, :-1].ravel(),
        indices[:, 1:].ravel()
    ])

    frame = gsd.hoomd.Frame()
    frame.particles.types = ['A']
    frame.particles.N = N
    frame.particles.position = positions
    frame.bonds.N = len(bonds)
    frame.bonds.group = bonds
    frame.bonds.types = ['b']
    frame.configuration.box = [L, L, L, 0, 0, 0]

    return frame

def check_bond_length_equilibration(snap,num_mon,num_pol,max_bond_length=1.1,min_bond_length=0.95):
    '''
    Check the bond distances.
    
    '''
    frame_ds = []
    for j in range(num_pol):
        idx = j*num_mon
        d1 = snap.particles.position[idx:idx+num_mon-1] - snap.particles.position[idx+1:idx+num_mon]
        L = snap.configuration.box[0]
        d1 -= L*np.round(d1/L)
        bond_l = np.linalg.norm(d1,axis=1)
        frame_ds.append(bond_l)
    max_frame_bond_l = np.max(np.array(frame_ds))
    min_frame_bond_l = np.min(np.array(frame_ds))
    print("max: ",max_frame_bond_l," min: ",min_frame_bond_l)
    if max_frame_bond_l <= max_bond_length and min_frame_bond_l >= min_bond_length:
        print("Bonds relaxed.")
        return True
    if max_frame_bond_l > max_bond_length or min_frame_bond_l < min_bond_length:
        return False

def check_inter_particle_distance(snap,minimum_distance=0.95):
    '''
    Check particle separations.
    
    '''
    positions = snap.particles.position
    box = snap.configuration.box
    aq = freud.locality.AABBQuery(box,positions)
    aq_query = aq.query(
        query_points=positions,
        query_args=dict(r_min=0.0, r_max=minimum_distance, exclude_ii=True),
    )
    nlist = aq_query.toNeighborList()
    if len(nlist)==0:
        print("Inter-particle separation reached.")
        return True
    else:
        return False
