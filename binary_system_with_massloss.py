import argparse
import numpy

from amuse.units import units as u
from amuse.community.hermite0.interface import Hermite
from amuse.units.nbody_system import nbody_to_si

from systems import twobodies_circular
from hdf5utils import HDF5HandlerAmuse

#TODO: PEP8-ify

def main():
    """
    Simulates a binary system with linear massloss.

    """

    init_twobodies = [twobodies_circular(i |u.MSun, 0.1|u.MSun, 1|u.AU) for i in range(1, 6)]

    with HDF5HandlerAmuse(args.filename) as datahandler:
        for bodies in init_twobodies:
            print(bodies)
            masslosstimes = (numpy.linspace(0.1, 1, 10)) * bodies[0].period0

            for time in masslosstimes:
                print(time.in_(u.yr))
                mass_sequence, time_sequence = massloss_evolution(time, bodies[0].mass)

                print("totalsteps {}".format(len(mass_sequence)))
                assert len(time_sequence) == len(mass_sequence)

                evolve_system_with_massloss(bodies, mass_sequence, time_sequence, datahandler)


def massloss_evolution(endtime, initmass, mf=0.5, step=1|u.day):
    """
    initmass : initial mass 
    mf : fraction of initmass at endtime
    endtime : time over which massloss should occur

    """
    assert endtime > 2*step
    timerange = numpy.arange(0, endtime.number, step.value_in(endtime.unit)) | endtime.unit
    massrange = numpy.linspace(initmass.number, mf*initmass.number, len(timerange)) | initmass.unit
    return massrange, timerange


def evolve_system_with_massloss(particles, mass_sequence, time_sequence, datahandler):
    """

    Parameters
    ----------
    particles: a two-body system 
    mass_sequence: sequence of masses

    """
    setname = "/"+str(numpy.random.randint(0, 200))+"/"

    integrator = Hermite(nbody_to_si(particles.total_mass(), 1 | u.AU))
    integrator.particles.add_particles(particles)

    for mass, time in zip(mass_sequence, time_sequence):
        integrator.evolve_model(time)
        integrator.particles[0].mass = mass 
        datahandler.append(integrator.particles.position, setname+"position")
        datahandler.append(integrator.particles.velocity, setname+"velocity")
        datahandler.append(integrator.particles.mass, setname+"mass")

        #This is because HDF5Handler doesn't support objects w/o .shape yet.
        datahandler.append([time.number]| time.unit, setname+"time")
     
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


