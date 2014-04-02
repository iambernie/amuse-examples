#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy

from amuse.units import constants 
from amuse.units import units 
from amuse.community.hermite0.interface import Hermite
from amuse.units.nbody_system import nbody_to_si

from ext.systems import twobodies_circular
from ext.hdf5utils import HDF5HandlerAmuse

from ext.misc import orbital_elements

class State(object):
    def __init__(self, startmass, endtime, mdot, starttime=0|units.yr, steps=None, dm=1e-5|units.MSun, dt=2|units.day):

        self.mass = startmass
        self.endtime = endtime
        self.time = starttime 
        self.mdot = mdot  # 1e-5 |(units.MSun/units.yr)
        self.intr_dt

    def next_time(self):
        self.time += dt
        return self.time

    def next_mass(self):
        nextmass = self.mass + self.dm
        return nextmass 

    def mdot(self):
        return dm/dt

    @property
    def stop(self):
        if next_mass :
            return False

def simulations(datahandler):
    """
    Set up the simulations here.
    """
    endtime = args.endtime
    steps_mass = args.stepsmass
    steps_time = args.stepstime 
    mdot = args.mdot
    #dm = 1e-5 |units.MSun
    #dt = 1 |units.yr

    mass1 = 1 |units.MSun
    mass2 = args.orbitmass |units.MSun

    twobody = twobodies_circular(mass1, mass2, 10|units.AU)
    state = State(mass1, endtime, mdot, steps_time=steps_time, steps_mass=steps_mass)

    h5path = "/test/"
    datahandler.prefix = h5path

    evolve_system(twobody, state, datahandler)
    #datahandler.file.flush()


def evolve_system(particles, state, datahandler):
    """

    Parameters
    ----------
    particles: a two-body system 
    datahandler: HDF5HandlerAmuse context manager

    """
    h = datahandler

    intr = Hermite(nbody_to_si(particles.total_mass(), 1 | units.AU))
    intr.particles.add_particles(particles)
    p = intr.particles

    mdot = state.mdot

    while state.stop is False:

        p.move_to_center() 

        a, e, i, w, W, f = orbital_elements(p)
        mu = constants.G*p.mass.sum() #standard grav. param.
        period = (2*numpy.pi)**2*a**3/mu
        mean_motion = 2*numpy.pi / period
        massloss_index = mdot / (mean_motion*mu)

        h.append(time, "time")
        h.append(a, "sma")
        h.append(e, "eccentricity")
        h.append(f, "true_anomaly")
        h.append(w, "argument_of_periapsis")
        h.append(period, "period0")
        h.append(massloss_index, "massloss_index")

        h.append(p.center_of_mass(), "CM_position")
        h.append(p.center_of_mass_velocity(), "CM_velocity")
        h.append(p.position, "position")
        h.append(p.velocity, "velocity")
        h.append(p.mass, "mass")
        h.append(p.kinetic_energy(), "kinetic_energy")
        h.append(p.potential_energy(), "potential_energy")
        h.append(intr.get_total_energy(), "total_energy")

        time = state.next_time()
        intr.evolve_model(time)

        nextmass = state.mass
        if nextmass != p[0].mass:
            p[0].mass = nextmass 

    intr.stop()


def massloss_index(massloss_rate, mean_motion, total_mass):
    return massloss_rate/(mean_motion*total_mass)

def mean_motion(period):
    return 2*numpy.pi/period


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f','--filename', required=True, 
                        help="Filepath for hdf5 file.")

    parser.add_argument('--stepsmass', type=int, default=200, metavar="DEFAULT: 200",
                        help="Max mass of central body in MSun")

    parser.add_argument('--stepstime', type=int, default=400, metavar="DEFAULT: 400",
                        help="Min mass of central body in MSun")

    parser.add_argument('--orbitmass', type=float, default=0.01, metavar="DEFAULT: 0.01",
                        help="Mass of the orbiting body in MSun")

    parser.add_argument('--seperation', type=float, default=1, metavar="DEFAULT: 1",
                        help="Initial seperation in AU")

    parser.add_argument('--endtime', type=float,  default=1000, metavar="DEFAULT: 1000")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    print(args)

    with HDF5HandlerAmuse(args.filename) as datahandler:
        simulations(datahandler)


