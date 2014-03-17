#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy

from amuse.units import constants
from amuse.units import units as u
from amuse.community.hermite0.interface import Hermite
from amuse.units.nbody_system import nbody_to_si

from systems import twobodies_circular
from hdf5utils import HDF5HandlerAmuse

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
    m = args.maxratio
    res = args.resolution
    step = args.stepsize|u.day

    init_twobodies = [twobodies_circular(i|u.MSun, 1|u.MSun, 1|u.AU) for i in range(1, m)]

    with HDF5HandlerAmuse(args.filename) as datahandler:

        syscount = 0
        for bodies in init_twobodies:
            sys_str = "system_"+str(syscount).zfill(2) 
            syscount += 1 

            masslosstimes = (numpy.linspace(0.1, 3, res)) * bodies[0].period0

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

def semimajoraxis_from_binary(binary, G=constants.G):

    if len(binary) != 2:
        raise Exception("Expects binary")

    else:
        total_mass = binary.mass.sum()
        position = binary[1].position - binary[0].position
        velocity = binary[1].velocity - binary[0].velocity

    specific_energy = (1.0/2.0)*velocity.lengths_squared() - G*total_mass/position.lengths()
    semimajor_axis = -G*total_mass/(2.0*specific_energy)
    return semimajor_axis


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

    for mass, time in zip(mass_sequence, time_sequence):
        integrator.evolve_model(time)
        integrator.particles[0].mass = mass 

        datahandler.append(integrator.particles.center_of_mass(), h5path+"center_of_mass")
        datahandler.append(integrator.particles.center_of_mass_velocity(), h5path+"center_of_mass_velocity")
        datahandler.append(integrator.particles.position, h5path+"position")
        datahandler.append(integrator.particles.velocity, h5path+"velocity")
        datahandler.append(integrator.particles.mass, h5path+"mass")

        #These need to be casted to VectorQuantities because HDF5Handler doesn't support objects w/o .shape yet.
        datahandler.append([time.number] | time.unit, h5path+"time")

        sma = semimajoraxis_from_binary(integrator.particles)

        datahandler.append([sma.number] | sma.unit, h5path+"sma")

        #FIXME: these raise IOErrors, find out why and where.
        #K = integrator.particles.kinetic_energy()
        #U = integrator.particles.potential_energy()
        #E = integrator.get_total_energy()
        #datahandler.append([K]| K.unit , h5path+"kinetic_energy")
        #datahandler.append([U]| U.unit , h5path+"potential_energy")
        #datahandler.append([E]| E.unit , h5path+"total_energy")
     
    integrator.stop()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--filename', metavar="HDF5 FILENAME")
    parser.add_argument('-r','--resolution', type=int, default=4)
    parser.add_argument('-m','--maxratio', type=int, default=3)
    parser.add_argument('-s','--stepsize', type=float, metavar="IN_DAYS", default=2)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    print(args)
    main()


