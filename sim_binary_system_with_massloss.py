#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy

from amuse.units import units as u
from amuse.community.hermite0.interface import Hermite
from amuse.units.nbody_system import nbody_to_si

from ext.systems import twobodies_circular
from ext.hdf5utils import HDF5HandlerAmuse
from ext.misc import semimajoraxis_from_binary

#TODO: PEP8-ify

def main():
    """
    Simulates a binary system with linear massloss.

    Writes data to hdf5 using the following structure:

    /
    ├── system_00
    │   ├── sequence_00
    │   │   ├── center_of_mass
    │   │   ├── center_of_mass_velocity
    │   │   ├── kinetic_energy
    │   │   ├── mass
    │   │   ├── position
    │   │   ├── potential_energy
    │   │   ├── total_energy
    │   │   └── velocity
    │   ├── sequence_01
    │   ├── sequence_02
    │   └── sequence_03
    ├── system_01
    ├── system_02
    └── system_03

    """
    M = args.maxmass
    m = args.minmass
    o = args.orbitmass
    t = args.deltat
    res = args.resolution
    step = args.stepsize |u.day

    assert M>m

    init_twobodies = [twobodies_circular(i|u.MSun, o|u.MSun, 1|u.AU) for i in range(m, M)]

    with HDF5HandlerAmuse(args.filename) as datahandler:

        syscount = 0
        for bodies in init_twobodies:
            sys_str = "system_"+str(syscount).zfill(2) 
            syscount += 1 

            masslosstimes = (numpy.linspace(0.1, t, res)) * bodies[0].period0

            seqcount = 0
            for time in masslosstimes:
                seq_str = "sequence_"+str(seqcount).zfill(2)
                h5path = "/"+sys_str+"/"+seq_str+"/"
                seqcount += 1

                mass_sequence, time_sequence = massloss_evolution(time, bodies[0].mass, step=step)
                assert len(time_sequence) == len(mass_sequence)
                print(h5path, time.in_(u.yr), "totalsteps {}".format(len(mass_sequence)))

                evolve_system_with_massloss(bodies, mass_sequence, time_sequence, datahandler, h5path)
                datahandler.file.flush()


def massloss_evolution(endtime, initmass, mf=0.5, step=2.0|u.day):
    """
    initmass : initial mass 
    mf : fraction of initmass at endtime
    endtime : time over which massloss should occur
    step: timestep

    """
    assert endtime > 2*step
    timerange = numpy.arange(0, endtime.number, step.value_in(endtime.unit)) | endtime.unit
    massrange = numpy.linspace(initmass.number, mf*initmass.number, len(timerange)) | initmass.unit
    return massrange, timerange


def evolve_system_with_massloss(particles, mass_sequence, time_sequence, datahandler, h5path):
    """

    Parameters
    ----------
    particles: a two-body system 
    mass_sequence: sequence of masses
    time_sequence: sequence of times
    datahandler: HDF5HandlerAmuse context manager
    h5path: unix-style path for this simulation's dataset

    """
    integrator = Hermite(nbody_to_si(particles.total_mass(), 1 | u.AU))
    integrator.particles.add_particles(particles)

    datahandler.append(particles[0].period0, h5path+"period0")

    for mass, time in zip(mass_sequence, time_sequence):
        integrator.evolve_model(time)
        integrator.particles[0].mass = mass 

        datahandler.append(integrator.particles.center_of_mass(), h5path+"center_of_mass")
        datahandler.append(integrator.particles.center_of_mass_velocity(), h5path+"center_of_mass_velocity")
        datahandler.append(integrator.particles.position, h5path+"position")
        datahandler.append(integrator.particles.velocity, h5path+"velocity")
        datahandler.append(integrator.particles.mass, h5path+"mass")
        datahandler.append(integrator.particles.kinetic_energy(), h5path+"kinetic_energy")
        datahandler.append(integrator.particles.potential_energy(), h5path+"potential_energy")
        datahandler.append(integrator.get_total_energy(), h5path+"total_energy")
        datahandler.append(time, h5path+"time")
        datahandler.append(semimajoraxis_from_binary(integrator.particles), h5path+"sma")
     
    integrator.stop()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--filename', required=True, metavar="HDF5 FILENAME")
    parser.add_argument('-r','--resolution', type=int, default=4)
    parser.add_argument('--maxmass', type=int, default=3, help="Max mass of central body in MSun")
    parser.add_argument('--minmass', type=int, default=1, help="Min mass of central body in MSun")
    parser.add_argument('--orbitmass', type=float, default=1, help="Mass of the orbiting body in MSun")
    parser.add_argument('-s','--stepsize', type=float, metavar="IN_DAYS", default=2)
    parser.add_argument('-t','--deltat', type=float,  default=3)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    print(args)
    main()


