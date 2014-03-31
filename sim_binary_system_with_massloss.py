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
from ext.misc import eccentricity_from_binary

#TODO: PEP8-ify

def main():
    """
    Simulates a binary system with linear massloss.

    Writes data to hdf5 using the following structure:

    /
    ├── system_00
    │   ├── sequence_00
    │   │   ├── CM_position
    │   │   ├── CM_velocity
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
    #TODO: lose all initial parameters such as M,m,o,seperation, t. Replace by mass-loss index.
    M = args.maxmass
    m = args.minmass
    o = args.orbitmass
    seperation = args.seperation
    t = args.deltat
    res = args.resolution
    step = args.stepsize |u.day

    assert M>m

    init_twobodies = [twobodies_circular(i|u.MSun, o|u.MSun, seperation|u.AU) 
                      for i in range(m, M)]

    with HDF5HandlerAmuse(args.filename) as datahandler:

        for syscount, bodies in enumerate(init_twobodies):
            sys_str = "system_"+str(syscount).zfill(2) 

            masslosstimes = (numpy.linspace(0.1, t, res)) * bodies[0].period0

            for seqcount, time in enumerate(masslosstimes):
                seq_str = "sequence_"+str(seqcount).zfill(2)
                h5path = "/"+sys_str+"/"+seq_str+"/"

                mass_seq, time_seq = massloss_evolution(time, bodies[0].mass, 
                                                        step=step)

                print(h5path, time.in_(u.yr), 
                      "totalsteps {}".format(len(mass_seq)))

                evolve_system_with_massloss(bodies, mass_seq, time_seq, 
                                            datahandler, h5path)

                datahandler.file.flush()


def massloss_evolution(endtime, initmass, mf=0.5, step=2.0|u.day):
    """
    initmass : initial mass 
    mf : fraction of initmass at endtime
    endtime : time over which massloss should occur
    step: timestep

    """
    assert endtime > 2*step

    timerange = numpy.arange(0, endtime.number, 
                             step.value_in(endtime.unit)) | endtime.unit

    massrange = numpy.linspace(initmass.number, mf*initmass.number, 
                               len(timerange)) | initmass.unit

    assert len(timerange) == len(massrange)
    return massrange, timerange


def evolve_system_with_massloss(particles, mass_sequence, time_sequence, 
                                datahandler, h5path):
    """

    Parameters
    ----------
    particles: a two-body system 
    mass_sequence: sequence of masses
    time_sequence: sequence of times
    datahandler: HDF5HandlerAmuse context manager
    h5path: unix-style path for this simulation's dataset

    """
    h = datahandler

    intr = Hermite(nbody_to_si(particles.total_mass(), 1 | u.AU))
    intr.particles.add_particles(particles)

    h.append(particles[0].period0, h5path+"period0")

    for mass, time in zip(mass_sequence, time_sequence):

        intr.particles.move_to_center() 
        intr.evolve_model(time)
        intr.particles[0].mass = mass 

        h.append(intr.particles.center_of_mass(), h5path+"CM_position")
        h.append(intr.particles.center_of_mass_velocity(), h5path+"CM_velocity")
        h.append(intr.particles.position, h5path+"position")
        h.append(intr.particles.velocity, h5path+"velocity")
        h.append(intr.particles.mass, h5path+"mass")
        h.append(intr.particles.kinetic_energy(), h5path+"kinetic_energy")
        h.append(intr.particles.potential_energy(), h5path+"potential_energy")
        h.append(intr.get_total_energy(), h5path+"total_energy")
        h.append(time, h5path+"time")
        h.append(semimajoraxis_from_binary(intr.particles), h5path+"sma")
        h.append(eccentricity_from_binary(intr.particles), h5path+"eccentricity")
     
    intr.stop()


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f','--filename', required=True, 
                        help="Filepath for hdf5 file.")

    parser.add_argument('-r','--resolution', type=int, default=4, metavar="4")

    parser.add_argument('--maxmass', type=int, default=3, metavar="3",
                        help="Max mass of central body in MSun")

    parser.add_argument('--minmass', type=int, default=1, metavar="1",
                        help="Min mass of central body in MSun")

    parser.add_argument('--orbitmass', type=float, default=0.1, metavar="0.1",
                        help="Mass of the orbiting body in MSun")

    parser.add_argument('--seperation', type=float, default=1, metavar="1",
                        help="Initial seperation in AU")

    parser.add_argument('-s','--stepsize', type=float, metavar="2", 
                        default=2, help="Update mass ever -s days.")

    parser.add_argument('-t','--deltat', type=float,  default=3)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    print(args)
    main()


