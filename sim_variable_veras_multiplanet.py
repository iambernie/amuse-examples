#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import argparse
import numpy
import time as Time

from amuse.units import constants 
from amuse.units import units 
from amuse.units.nbody_system import nbody_to_si

from amuse.community.hermite0.interface import Hermite
from amuse.community.ph4.interface import ph4
from amuse.community.huayno.interface import Huayno
from amuse.community.smalln.interface import SmallN

from ext.hdf5utils import HDF5HandlerAmuse
from ext.misc import orbital_elements
from ext.misc import VariableMassState
from ext.misc import args_quantify
from ext.misc import args_integrators

from ext import systems

def main():
    integrators = '_'.join([intr.__name__[:2] for intr in args.integrators])

    assert args.mdot*args.endtime < args.centralmass

    if args.logspace:
        space = 'logspace'
        etas = numpy.logspace(*args.etas)
    elif args.linspace:
        space = 'linspace'
        etas = numpy.linspace(*args.etas)
    else:
        space = 'arange'
        etas = numpy.arange(*args.etas)

    if not args.filename:
        spacing = "_".join([str(i) for i in args.etas])
        subst = (args.centralmass.number, args.orbitmass.number,
                 args.mdot.number, args.smainner.number, spacing, space, args.endtime.number,
                 args.datapoints, integrators)
        name = 'variable_multiplanet_M{}_m{}_mdot{}_sma_in{}__eta{}_{}_t{}_p{}_{}.hdf5'.format(*subst)
        filename = args.directory+name
    else:
        filename = args.filename

    with HDF5HandlerAmuse(filename) as datahandler:
        commit = subprocess.check_output(["git", "rev-parse","HEAD"])

        datahandler.file.attrs['commit'] = commit
        datahandler.file.attrs['M'] = str(args.centralmass)
        datahandler.file.attrs['m'] = str(args.orbitmass)
        datahandler.file.attrs['mdot'] = str(args.mdot)
        datahandler.file.attrs['sma'] = str(args.smainner)
        datahandler.file.attrs['space'] = space
        datahandler.file.attrs['etas'] = spacing
        datahandler.file.attrs['endtime'] = str(args.endtime)
        datahandler.file.attrs['datapoints'] = args.datapoints
        datahandler.file.attrs['integrators'] = integrators

        simulations(datahandler, etas)


def simulations(datahandler, etas):
    """
    Set up the simulations here.

    Essentially:
        - decide on a name to use as prefix for each simulation
        - create a MassState() instance. (i.e. define a massloss prescription)
        - call evolve_system()

    """
    body1 = dict(mass=args.orbitmass, elements=dict(semimajor_axis=args.smainner, eccentricity=0))
    body2 = dict(mass=args.orbitmass, elements=dict(semimajor_axis=args.smaouter, eccentricity=0.5))
    threebody = systems.nbodies(args.centralmass, body1, body2)

    for intr in args.integrators:
        for i, eta in enumerate(etas):
            datahandler.prefix = intr.__name__+"/sim_"+str(i).zfill(4)+"/"
            datahandler.append(eta, "eta")
            state = VariableMassState(args.endtime, threebody[0].mass, args.mdot, 
                              datapoints=args.datapoints, eta=eta, name=datahandler.prefix)
            evolve_system(intr, threebody, state, datahandler)
            datahandler.file.flush()

def evolve_system(integrator, particles, state, datahandler):
    """
    Iteratively calls integrator.evolve_model().

    Parameters
    ----------
    particles: a two-body system 
    state: MassState instance
    datahandler: HDF5HandlerAmuse context manager

    """
    intr = integrator(nbody_to_si(particles.total_mass(), 1 |units.AU))
    intr.particles.add_particles(particles)

    state.intr = intr

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
    h.append(p.kinetic_energy(), "kinetic_energy")
    h.append(p.potential_energy(), "potential_energy")
    h.append(intr.get_total_energy(), "total_energy")

    p0 = p.copy_to_new_particles()
    p1 = p.copy_to_new_particles()
    p0.remove_particle(p0[2])
    p1.remove_particle(p1[1])

    for i, pset in enumerate([p0, p1]):
        currentprefix = datahandler.prefix        
        datahandler.prefix = currentprefix+"p"+str(i)+"/"

        a, e, i, w, W, f = orbital_elements(pset)
        mu = pset.mass.sum() #NOT the stand. grav. parameter 
        period = ((2*numpy.pi)**2*a**3/(constants.G*mu)).sqrt()
        mean_motion = 2*numpy.pi / period
        massloss_index = state.mdot / (mean_motion*mu)
        h.append(a, "sma")
        h.append(e, "eccentricity")
        h.append(i, "inclination")
        h.append(f, "true_anomaly")
        h.append(w, "argument_of_periapsis")
        h.append(W, "longitude_of_ascending_node")
        h.append(period, "period")
        h.append(massloss_index, "massloss_index")
        datahandler.prefix = currentprefix


def get_arguments():
    parser = argparse.ArgumentParser()


    parser.add_argument('-f','--filename',
                        help="Filepath for hdf5 file.")

    parser.add_argument('-d','--directory', default='data/',
                        help="Target directory for hdf5 file.")

    parser.add_argument('--mdot', type=float,
                        default=1.244e-5 | (units.MSun/units.yr),
                        action=args_quantify(units.MSun/units.yr),
                        help="Masslossrate in MSun/yr")

    parser.add_argument('-M', '--centralmass', type=float,
                        default=7.659 | units.MSun,
                        action=args_quantify(units.MSun),
                        help="Mass of the orbiting body in MSun")

    parser.add_argument('-m', '--orbitmass', type=float,
                        default=0.001 | units.MSun,
                        action=args_quantify(units.MSun),
                        help="Mass of the orbiting body in MSun")

    parser.add_argument('--smainner', type=float,
                        default=10 | units.AU,
                        action=args_quantify(units.AU),
                        help="Initial semimajoraxis in AU (inner body)")

    parser.add_argument('--smaouter', type=float,
                        default=30 | units.AU,
                        action=args_quantify(units.AU),
                        help="Initial semimajoraxis in AU (outer body)")

    parser.add_argument('--endtime', type=float,
                        default=5e5 | units.yr,
                        action=args_quantify(units.yr),
                        help="endtime in years.")

    parser.add_argument('--integrators', nargs='+',
                        default=[SmallN, ph4, Hermite, Huayno],
                        action=args_integrators(),
                        help="Integrators to use. Valid integrators:\
                        Hermite, SmallN, ph4, Huayno")

    parser.add_argument('--datapoints', type=int, default=500,
                        help="Number of datapoints.")

    parser.add_argument('--logspace', action='store_true', 
                        help="Use numpy.logspace in stead of numpy.arange \
                              to set timesteps.")

    parser.add_argument('--linspace', action='store_true', 
                        help="Use numpy.linspace in stead of numpy.arange \
                              to set timesteps.")

    parser.add_argument('--etas', type=float, default=[0.1, 2, 0.1], 
                        nargs=3, help="Supply numpy.arange(START, STOP, STEP) \
                        arguments to create an ndarray of etas. \
                        e.g.: --etas 10 100 10")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_arguments()
    print(args)
    main()



