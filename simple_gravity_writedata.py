import argparse

import numpy

from amuse.units import units
from amuse.datamodel.particles import Particles
from amuse.community.hermite0.interface import Hermite
from amuse.units.nbody_system import nbody_to_si

from systems import sun_and_earth

from hdf5utils import HDF5HandlerAmuse

def main():
    """
    Minimalistic stellar dynamics simulation to test HDF5Handler.

    """
    bodies = sun_and_earth()
    evolve_system(bodies)


def evolve_system(particles):
    """
    Evolves the system using the Hermite integrator.

    Parameters
    ----------
    particles: amuse.datamodel.particles.Particles instance

    """
    times = numpy.linspace(1, args.time, args.steps) |units.yr

    integrator = Hermite(nbody_to_si(particles.total_mass(), 1 | units.AU))
    integrator.particles.add_particles(particles)

    with HDF5HandlerAmuse(args.filename) as h:
        for i, t in enumerate(times):
            integrator.evolve_model(t)
            h.append(integrator.particles.position, 'positie')
            h.append(integrator.particles.velocity, 'snelheid')

    integrator.stop()


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f','--filename', metavar="HDF5 FILENAME")
    parser.add_argument('-s','--steps', metavar="NR_OF_STEPS", type=int,
                        default=50)
    parser.add_argument('-t','--time', metavar="END_TIME", type=float,
                        default=50, help="in years")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    print(args)
    main()








    





