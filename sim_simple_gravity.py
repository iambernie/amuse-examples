import argparse

import numpy

from amuse.units import units as u
from amuse.datamodel.particles import Particles
from amuse.community.hermite0.interface import Hermite
from amuse.units.nbody_system import nbody_to_si

from ext.systems import sun_and_earth

def main():
    """
    Minimalistic stellar dynamics simulation.
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
    times = numpy.linspace(0.0001, args.time, args.steps) |u.yr

    integrator = Hermite(nbody_to_si(particles.total_mass(), 1 | u.AU))
    integrator.particles.add_particles(particles)

    if args.dt is not None:
        integrator.set_dt_param(args.dt)

    for t in times:
        integrator.evolve_model(t)
        seperation = (integrator.particles[0].position - integrator.particles[1].position).length()
        #print(integrator.get_time_step().in_(u.day), integrator.get_time().in_(u.yr), t.in_(u.yr), seperation.in_(u.AU) )

    integrator.stop()


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s','--steps', metavar="NR_OF_STEPS", type=int,
                        default=50)
    parser.add_argument('-t','--time', metavar="END_TIME", type=float,
                        default=50, help="in years")
    parser.add_argument('--dt',  type=float,
                        default=None, help="set dt param")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    print(args)
    main()

