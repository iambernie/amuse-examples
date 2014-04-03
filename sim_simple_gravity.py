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

    intr = Hermite(nbody_to_si(particles.total_mass(), 1 | u.AU))
    intr.particles.add_particles(particles)

    energy_begin = intr.get_total_energy()

    if args.dt is not None:
        intr.set_dt_param(args.dt)

    for t in times:
        intr.evolve_model(t)
        energy_error = (intr.get_total_energy() - energy_begin)/energy_begin
        print(intr.get_time_step().in_(u.day), intr.get_time().in_(u.yr))
        print(energy_error)
      

    print("energy error:{}".format(energy_error))

    intr.stop()


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

