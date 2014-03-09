"""
Routine for evolving a binary system without massloss.
"""

import argparse

from numpy import linspace

from amuse.units import units
from amuse.datamodel.particles import Particles
from amuse.community.hermite0.interface import Hermite
from amuse.units.nbody_system import nbody_to_si
from amuse.units.quantities import AdaptingVectorQuantity

from hdf5support import write_to_hdf5

import matplotlib.pyplot as plt

def main():
    data = evolve_system()
    write_to_hdf5(args.write, data)

    plot_energy(data['times'], data['energies'])
    #plot_positions(data)
    

def evolve_system():
    """
    Integrates the orbit of a binary system.
    """
    #for now define here, but use argparse later.
    n_steps = 1000
    t_end = 10000 | units.yr
    masses = [10, 1] | units.MSun
    positions = [[1,0,0], [0,0,0]] | units.AU
    velocities = [[0, 1000, 0], [0, -1000, 0]] | units.kms
 
    bodies = binary_system(masses, positions, velocities)

    integrator = Hermite(nbody_to_si(bodies.total_mass(), 10.0 | units.AU))
    integrator.particles.add_particles(bodies)

    positions_in_time = AdaptingVectorQuantity()
    velocities_in_time = AdaptingVectorQuantity()
    energy_in_time = AdaptingVectorQuantity()
    
    times = linspace(1|units.yr, t_end, n_steps)
    for t in times:
        print(t.value_in(units.yr))
        integrator.evolve_model(t)
        positions_in_time.append(integrator.particles.position)
        velocities_in_time.append(integrator.particles.velocity)
        energy_in_time.append(integrator.get_total_energy())

    integrator.stop()

    data = dict(positions=positions_in_time,
                velocities=velocities_in_time,
                energies=energy_in_time,
                times=times)
    return data 

def binary_system(masses, positions, velocities):
    """
    Sets up a Particles instance representing a binary system.

    Parameters
    ----------
    masses: (i.e: [10, 1] | units.MSun)
    positions: (i.e: [[1,0,0], [0,0,0]] | units.AU)
    velocities: (i.e: [[0, 10, 0], [0, -10, 0]] | units.kms)

    Returns
    -------
    amuse.datamodel.particles.Particles instance
    """

    bodies = Particles(2)
    bodies.mass = masses 
    bodies.position = positions 
    bodies.velocity = velocities

    return bodies
    

def plot_energy(x, y):
    """
    x: time
    y: energy
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel('time')
    ax.set_ylabel('total energy')
    ax.plot(x.value_in(units.yr), y.value_in(units.J))

    #ax.set_xlim(10,30)
    #ax.set_ylim(10,30)
    #ax1.set_xticklabels([]) #removes ticklables
    #ax1.set_xticks()    

    plt.savefig('total_energy.png', bbox_inches='tight', dpi=150)
    fig.clf()
    plt.close()

def plot_positions(data):
    """
    pos1
    """

    data['positions']
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.plot(x1.value_in(units.AU), y1.value_in(units.AU))
    ax.plot(x2.value_in(units.AU), y2.value_in(units.AU))

    #ax.set_xlim(10,30)
    #ax.set_ylim(10,30)
    #ax1.set_xticklabels([]) #removes ticklables
    #ax1.set_xticks()    

    plt.savefig('total_energy.png', bbox_inches='tight', dpi=150)
    fig.clf()
    plt.close()

def plot_velocities():
    pass

def get_options():
    parser = argparse.ArgumentParser()

    ############# ARGUMENTS ##############
    #parser.add_argument('argname', type=str, nargs='+',
    #               help='variable length')

    #parser.add_argument('-v','--verbose',\
    #               action='store_true',\
    #               help='Increases verbosity.')
    #parser.add_argument('-o','--outputdir', type=str, default='images',\
    #               help='output directory for images.')
    parser.add_argument('-w','--write', type=str, default='binary.hdf5',\
                   help='Filepath to write hdf5 file to.')
    #parser.add_argument('-t','--test',\
    #               action='store_true',\
    #               help='Use test sequence [2,2,2,1,2].')
    #parser.add_argument('-f','--full',\
    #               action='store_true',\
    #               help='print full arrays.')
    #parser.add_argument('-o','--initial', type=int, default=10,\
    #               help='length of initial set')
    #parser.add_argument('-m','--max_turns', type=int, default=100000,\
    #               help='maximum number of turns')
    #parser.add_argument('t_init', type=str, nargs='+',
    #               help='length of initial set')
    #parser.add_argument('-o','--outputdir', type=str, default='images',\
    #               help='output directory for images.')
    #####################################

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_options()
    main()

    





