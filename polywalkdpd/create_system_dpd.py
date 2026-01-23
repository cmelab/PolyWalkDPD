import numpy as np  
import freud
import gsd, gsd.hoomd 
import hoomd 
import time

from dpd_utils import pbc,initialize_snapshot_rand_walk,check_bond_length_equilibration,check_inter_particle_distance


def create_polymer_system_dpd(num_pol,num_mon,density,k=20000,bond_l=1.0,r_cut=1.15,kT=1.0,A=1000,gamma=800,dt=0.001,particle_spacing=1.1):
    
    '''
    Initialize a polymer system in a cubic box using a random walk and a HOOMD simulation with DPD forces.

    ----------
    Parameters
    ----------
    num_pol : int, required
        number of polymers in system
    num_mon : int, required
        length of polymers in system
    density : float, required
        number density to initalize the system
    k : int, default 20000
        spring constant for harmonic bonds
    bond_l : float, default 1.0
        harmonic bond rest length
    r_cut : float, default 1.15
        cutoff pair distance for neighbor list
    kT : float, default 1.0
        temperature of thermostat
    A : float, default 1000
        DPD force parameter
    gamma : float, default 800
        DPD drag parameter (mass/time)
    dt : float, default 0.001
        timestep for HOOMD simulation
    particle_spacing : float, default 1.1
        condition for ending the soft push simulation

    -------
    Returns
    -------
    
    positions : list
        returns list of particle positions
        
    '''
    print(num_pol*num_mon)
    print(f"\nRunning with A={A}, gamma={gamma}, k={k}, "
          f"num_pol={num_pol}, num_mon={num_mon}")
    start_time = time.perf_counter()
    frame = initialize_snapshot_rand_walk(num_mon=num_mon,num_pol=num_pol,bond_length=bond_l,density=density)
    build_stop = time.perf_counter()
    print("Total build time: ", build_stop-start_time)
    harmonic = hoomd.md.bond.Harmonic()
    harmonic.params["b"] = dict(r0=bond_l, k=k)
    integrator = hoomd.md.Integrator(dt=dt)
    integrator.forces.append(harmonic)
    simulation = hoomd.Simulation(device=hoomd.device.auto_select(), seed=np.random.randint(65535))# TODO seed
    simulation.operations.integrator = integrator 
    simulation.create_state_from_snapshot(frame)
    const_vol = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    integrator.methods.append(const_vol)
    nlist = hoomd.md.nlist.Cell(buffer=0.4)
    simulation.operations.nlist = nlist
    DPD = hoomd.md.pair.DPD(nlist, default_r_cut=r_cut, kT=kT)
    DPD.params[('A', 'A')] = dict(A=A, gamma=gamma)
    integrator.forces.append(DPD)
    
    simulation.run(0)
    simulation.run(1000)
    snap=simulation.state.get_snapshot()
    N = num_pol*num_mon
    time_factor = N/90000
    
    while not check_bond_length_equilibration(snap,num_mon, num_pol,max_bond_length=particle_spacing): 
        check_time = time.perf_counter()
        if (check_time-start_time) > 60*time_factor:
            return num_pol*num_mon, 0
        simulation.run(1000)
        snap=simulation.state.get_snapshot()

    while not check_inter_particle_distance(snap,minimum_distance=0.95):
        check_time = time.perf_counter()
        if (check_time-start_time) > 60*time_factor:
            return num_pol*num_mon, 0
        simulation.run(1000)
        snap=simulation.state.get_snapshot()
        
    end_time = time.perf_counter()
    print("Total build and simulation time:", end_time - start_time)
    return snap.particles.position