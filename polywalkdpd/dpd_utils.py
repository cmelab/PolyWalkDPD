import numpy as np  
import freud
import gsd, gsd.hoomd 
import hoomd 
import time

def initialize_snapshot_rand_walk(num_pol, num_mon, density=0.85, bond_length=1.0, buffer=0.1):
    '''
    Create a HOOMD snapshot of a cubic box with the number density given by input parameters.
    Configure particles using a naiive random walk.
    
    '''    
    N = num_pol * num_mon
    L = np.cbrt(N / density)  # Calculate box size based on density
    positions = np.zeros((N, 3))
    for i in range(num_pol):
        start = i * num_mon
        positions[start] = np.random.uniform(low=(-L/2),high=(L/2),size=3)
        for j in range(num_mon - 1):
            delta = np.random.uniform(low=(-bond_length/2),high=(bond_length/2),size=3)
            delta /= np.linalg.norm(delta)*bond_length
            positions[start+j+1] = positions[start+j] + delta
    positions = pbc(positions,[L,L,L])
    bonds = []
    for i in range(num_pol):
        start = i * num_mon
        for j in range(num_mon - 1):
            bonds.append([start + j, start + j + 1])
    bonds = np.array(bonds)
    frame = gsd.hoomd.Frame()
    frame.particles.types = ['A']
    frame.particles.N = N
    frame.particles.position = positions
    frame.bonds.N = len(bonds)
    frame.bonds.group = bonds
    frame.bonds.types = ['b']
    frame.configuration.box = [L, L, L, 0, 0, 0]
    return frame

def pbc(d,box):
    '''
    periodic boundary conditions
    
    '''
    for i in range(3):
        a = d[:,i]
        pos_max = np.max(a)
        pos_min = np.min(a)
        while pos_max > box[i]/2 or pos_min < -box[i]/2:
            a[a < -box[i]/2] += box[i]
            a[a >  box[i]/2] -= box[i]
            pos_max = np.max(a)
            pos_min = np.min(a)
    return d

def check_bond_length_equilibration(snap,num_mon,num_pol,max_bond_length=1.1,min_bond_length=0.95):
    '''
    Check the bond distances.
    
    '''
    frame_ds = []
    for j in range(num_pol):
        idx = j*num_mon
        d1 = snap.particles.position[idx:idx+num_mon-1] - snap.particles.position[idx+1:idx+num_mon]
        bond_l = np.linalg.norm(pbc(d1,snap.configuration.box),axis=1)
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