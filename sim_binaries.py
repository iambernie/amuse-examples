#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy
import time as Time

from amuse.units import units 
from amuse.units import constants 
from amuse.units.nbody_system import nbody_to_si
from amuse.community.mercury.interface import Mercury
from amuse.community.hermite0.interface import Hermite
from amuse.community.ph4.interface import ph4
from amuse.community.huayno.interface import Huayno
from amuse.community.smalln.interface import SmallN

from ext.misc import MassState
from ext.misc import new_binary_from_elements
from ext.misc import orbital_elements
from ext.hdf5utils import HDF5HandlerAmuse
from ext import progressbar


def simulations():
    """
    Simulates a binary system with different starting eccentricities.

    Writes data to hdf5 using the following structure:

    """
    M = args.centralmass | units.MSun
    m = args.orbitmass | units.MSun
    mdot = args.mdot | (units.MSun/units.yr)
    seperation = args.seperation | units.AU
    datapoints = args.datapoints
    massupdateinterval = args.updateinterval | units.yr
    endtime = args.endtime | units.yr

    integrators = dict(Mercury=Mercury, Hermite=Hermite, ph4=ph4, Huayno=Huayno,
                       SmallN=SmallN)

    eccentricities = numpy.arange(0.1, 1, 0.1)
    twobodies = [new_binary_from_elements(M, m, seperation, eccentricity=e)
                                          for e in eccentricities]

    for name in args.integrators:
        if name in integrators:
            intr = integrators[name]
        else:
            continue

        for i, bodies in enumerate(twobodies):
            datahandler.prefix = '/'+name+'/'+str(i).zfill(2)+'/'
            state = MassState(massupdateinterval, endtime, bodies[0].mass, 
                              mdot, datapoints, name=datahandler.prefix)
            evolve_system(intr, bodies, state, datahandler)
            datahandler.file.flush()


def evolve_system(integrator, particles, state, datahandler):
    """
    Parameters
    ----------
    particles: a two-body system 
    state: MassState instance
    datahandler: HDF5HandlerAmuse context manager

    """
    intr = integrator(nbody_to_si(particles.total_mass(), 1 |units.AU))
    intr.particles.add_particles(particles)

    time, mass = next(state.time_and_mass)
    savepoint = next(state.savepoint)

    while state.stop is False:
        #Race Condition
        if savepoint < time:
            intr.evolve_model(savepoint)
            store_data(intr, state, datahandler)

            try:
                savepoint = next(state.savepoint)
            except StopIteration:
                pass

        elif time < savepoint:
            intr.evolve_model(time)
            intr.particles[0].mass = mass

            try:
                time, mass = next(state.time_and_mass)
            except StopIteration:
                pass

        elif time == savepoint:
            intr.evolve_model(time)
            intr.particles[0].mass = mass
            store_data(intr, state, datahandler)

            try:
                time, mass = next(state.time_and_mass)
            except StopIteration:
                pass

            try:
                savepoint = next(state.savepoint)
            except StopIteration:
                pass

    intr.stop()

def store_data(intr, state, datahandler):
    """
    Set up which parameters to store in the hdf5 file here.
    """
    h = datahandler
    p = intr.particles

    h.append(intr.get_time().in_(units.yr), "time")
    h.append(Time.time(), "walltime")
    h.append(p.center_of_mass(), "CM_position")
    h.append(p.center_of_mass_velocity(), "CM_velocity")
    h.append(p.position, "position")
    h.append(p.velocity, "velocity")
    h.append(p.mass, "mass")
    h.append(p.kinetic_energy(), "kinetic_energy")
    h.append(p.potential_energy(), "potential_energy")
    h.append(intr.get_total_energy(), "total_energy")

    currentprefix = datahandler.prefix        
    datahandler.prefix = currentprefix+"p0"+"/"
    a, e, i, w, W, f = orbital_elements(p)
    mu = p.mass.sum() 
    period = ((2*numpy.pi)**2*a**3/(constants.G*mu)).sqrt()
    mean_motion = 2*numpy.pi / period
    massloss_index = state.mdot / (mean_motion*mu)
    h.append(a, "sma")
    h.append(e, "eccentricity")
    h.append(f, "true_anomaly")
    h.append(period, "period")
    h.append(massloss_index, "massloss_index")

    datahandler.prefix = currentprefix


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f','--filename', required=True, 
                        help="Filepath for hdf5 file.")

    parser.add_argument('-M', '--centralmass', type=float, default=1,
                        help="Mass of central body in MSun")

    parser.add_argument('-m', '--orbitmass', type=float, default=0.001,
                        help="Mass of the orbiting body in MSun")

    parser.add_argument('--mdot', type=float, default=5e-5,
                        help="Masslossrate in MSun/Yr")

    parser.add_argument('--seperation', type=float, default=200,
                        help="Initial seperation in AU")

    parser.add_argument('-u', '--updateinterval', type=float, default=1,
                        help="Mass update interval in yearsr.")

    parser.add_argument('--endtime', type=float, default=1e4,
                        help="endtime in years.")

    parser.add_argument('--datapoints', type=int,  default=1000,
                        help="Number of datapoints.")

    parser.add_argument('--integrators', default=['SmallN'], nargs='+',
                        help="Integrators to use.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    print(args)
    
    with HDF5HandlerAmuse(args.filename) as datahandler:
        simulations()


