import argparse

import numpy

from amuse.units import units
from amuse.datamodel.particles import Particles
from amuse.community.hermite0.interface import Hermite
from amuse.units.nbody_system import nbody_to_si

def main():
    """
    Minimalistic stellar dynamics simulation to test hdf5 writer.
    """
    bodies = system_of_sun_and_earth()
    evolve_system(bodies)


def evolve_system(particles):
    """
    Integrates the orbit of a system.
    """

    times = numpy.linspace(1, args.time, args.steps) |units.yr

    integrator = Hermite(nbody_to_si(particles.total_mass(), 1 | units.AU))
    integrator.particles.add_particles(particles)

    for t in times:
        integrator.evolve_model(t)

    integrator.stop()


def system_of_sun_and_earth():
    """
    Sets up a two-body system representing the sun and earth.
 
    Parameters
    ----------
    None

    Returns
    -------
    Particles instance representing the sun and earth.
    """

    stars = Particles(2)

    sun = stars[0]
    sun.mass = 1.0 | units.MSun
    sun.position = (0.0, 0.0, 0.0) | units.m
    sun.velocity = (0.0, 0.0, 0.0) | (units.m / units.s)
    sun.radius = 1.0 | units.RSun

    earth = stars[1]
    earth.mass = 5.9736e24 | units.kg
    earth.radius = 6371.0 | units.km
    earth.position = (149.5e6, 0.0, 0.0) | units.km
    earth.velocity = (0.0, 29800, 0.0) | (units.m / units.s)

    return stars

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s','--steps', metavar="NR_OF_STEPS", type=int,
                        default=50)
    parser.add_argument('-t','--time', metavar="END_TIME", type=float,
                        default=50, help="in years")

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = get_arguments()
    main()

































    





