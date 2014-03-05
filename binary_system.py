"""
Routine for evolving a binary system without massloss.
"""

import argparse

from numpy import linspace

from amuse.units import units
from amuse.datamodel.particles import Particles
from amuse.community.hermite0.interface import Hermite
from amuse.units.nbody_system import nbody_to_si

def main():
    """
    Integrates the orbit of a binary system.
    
    """

    #for now define here, but use argparse later.
    n_steps = 1000
    t_end = 100 | units.yr
    masses = [10, 1] | units.MSun
    positions = [[1,0,0], [0,0,0]] | units.AU
    velocities = [[0, 10, 0], [0, -10, 0]] | units.kms
 
    bodies = binary_system(masses, positions, velocities)

    integrator = Hermite(nbody_to_si(bodies.total_mass(), 10.0 | units.AU))
    integrator.particles.add_particles(bodies)

    times = linspace(1|units.yr, t_end, n_steps)
    for t in times:
        integrator.evolve_model(t)
   
    return 0 

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
    
def plot_sma_vs_timestep():

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel('delta T/Period')
    ax.set_ylabel('A/A0')
    ax.set_title('Semi-major-axis vs timestep')
    ax.plot(x, y)

    #ax.set_xlim(10,30)
    #ax.set_ylim(10,30)
    #ax1.set_xticklabels([]) #removes ticklables
    #ax1.set_xticks()    

    plt.savefig('test.png', bbox_inches='tight', dpi=150)
    fig.clf()
    plt.close()


def plot_energy():

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel('time')
    ax.set_ylabel('total energy')
    ax.plot(x, y)

    #ax.set_xlim(10,30)
    #ax.set_ylim(10,30)
    #ax1.set_xticklabels([]) #removes ticklables
    #ax1.set_xticks()    

    plt.savefig('positions.png', bbox_inches='tight', dpi=150)
    fig.clf()
    plt.close()

def plot_positions():
    pass

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

    





