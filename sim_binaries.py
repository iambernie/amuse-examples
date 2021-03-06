#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import argparse
import numpy
import time as Time

from amuse.units import units 
from amuse.units import constants 
from amuse.units.nbody_system import nbody_to_si
from amuse.community.hermite0.interface import Hermite
from amuse.community.ph4.interface import ph4
from amuse.community.huayno.interface import Huayno
from amuse.community.smalln.interface import SmallN

from ext.hdf5utils import HDF5HandlerAmuse
from ext.misc import MassState
from ext.misc import new_binary_from_elements
from ext.misc import orbital_elements
from ext.misc import args_quantify
from ext.misc import args_integrators

def main():
    integrators = '_'.join([intr.__name__[:2] for intr in args.integrators])

    if not args.filename:
        subst = (args.centralmass.number, args.orbitmass.number, 
                 args.mdot.number, args.sma.number, args.interval.number, 
                 args.endtime.number, args.datapoints, integrators)
        name = 'binaries_M{}_m{}_mdot{}_s{}_i{}_t{}_p{}_{}.hdf5'.format(*subst)
        filename = args.directory+name
    else:
        filename = args.filename


    with HDF5HandlerAmuse(filename) as datahandler:
        eccentricities = numpy.arange(0, 1, 0.1)
        commit = subprocess.check_output(["git", "rev-parse","HEAD"])

        datahandler.file.attrs['commit'] = commit
        datahandler.file.attrs['eccentricities'] = str(eccentricities)
        datahandler.file.attrs['M'] = str(args.centralmass)
        datahandler.file.attrs['m'] = str(args.orbitmass)
        datahandler.file.attrs['mdot'] = str(args.mdot)
        datahandler.file.attrs['sma'] = str(args.sma)
        datahandler.file.attrs['interval'] = str(args.interval)
        datahandler.file.attrs['endtime'] = str(args.endtime)
        datahandler.file.attrs['datapoints'] = args.datapoints
        datahandler.file.attrs['integrators'] = integrators

        simulations(datahandler, eccentricities)


def simulations(datahandler, eccentricities):
    """ Simulates a binary system with different starting eccentricities. """

    twobodies = [new_binary_from_elements(args.centralmass, args.orbitmass, 
                                          args.sma, eccentricity=e) 
                                          for e in eccentricities]

    for intr in args.integrators:
        for i, bodies in enumerate(twobodies):
            datahandler.prefix = '/'+intr.__name__+'/'+str(i).zfill(4)+'/'
            state = MassState(args.interval, args.endtime, bodies[0].mass, 
                              args.mdot, args.datapoints, name=datahandler.prefix)
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
    intr = integrator(nbody_to_si(particles.total_mass(), 10 |units.AU))
    intr.particles.add_particles(particles)

    time, mass = next(state.time_and_mass)
    savepoint = next(state.savepoint)

    while state.stop is False:
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
    h.append(p.total_angular_momentum(), "angular_momentum")
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
    h.append(i, "inclination")
    h.append(w, "argument_of_periapsis")
    h.append(W, "longitude_of_ascending_node")
    h.append(f, "true_anomaly")
    h.append(period, "period")
    h.append(massloss_index, "massloss_index")

    datahandler.prefix = currentprefix


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f','--filename',
                        help="Filepath for hdf5 file.")

    parser.add_argument('-d','--directory', default='data/',
                        help="Target directory for hdf5 file.")

    parser.add_argument('-M', '--centralmass', type=float, 
                        default=1 | units.MSun, 
                        action=args_quantify(units.MSun),
                        help="Mass of central body in MSun")

    parser.add_argument('-m', '--orbitmass', type=float, 
                        default=0.001 | units.MSun,
                        action=args_quantify(units.MSun),
                        help="Mass of the orbiting body in MSun")

    parser.add_argument('--mdot', type=float, 
                        default=5e-5 | (units.MSun/units.yr),
                        action=args_quantify(units.MSun/units.yr),
                        help="Masslossrate in MSun/yr")

    parser.add_argument('--sma', type=float, 
                        default=200 | units.AU,
                        action=args_quantify(units.AU),
                        help="Initial semimajoraxis in AU")

    parser.add_argument('-i', '--interval', type=float, 
                        default=1 | units.yr,
                        action=args_quantify(units.yr),
                        help="Mass update interval in years.")

    parser.add_argument('--endtime', type=float, 
                        default=1.5e4 | units.yr,
                        action=args_quantify(units.yr),
                        help="endtime in years.")

    parser.add_argument('--datapoints', type=int,  default=1000,
                        help="Number of datapoints.")

    parser.add_argument('--integrators', nargs='+',
                        default=[SmallN, ph4, Hermite, Huayno], 
                        action=args_integrators(),
                        help="Integrators to use. Valid integrators:\
                        Hermite, SmallN, ph4, Huayno")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    print(args)
    main()
    


