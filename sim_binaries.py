#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy

from amuse.units import constants 
from amuse.units import units 
from amuse.community.hermite0.interface import Hermite
from amuse.units.nbody_system import nbody_to_si
from amuse.datamodel.particles import Particles

from ext.hdf5utils import HDF5HandlerAmuse
from ext.misc import orbital_elements
from ext.misc import new_binary_from_elements

class State(object):
    def __init__(self, mdot, endtime, dt_param):
        self.time = None 
        self.endtime = endtime
        self.mdot = mdot  
        self.dt_param = dt_param
        self.next_mass = None
        self.stop = False
        self.timestep = None

    def next_time(self):
        if self.timestep is not None:
            self.time += self.timestep

            if self.time > self.endtime:
                self.stop = True
            return self.time

        else:
            raise Exception("No timestep defined")

    def massloss(self):
        return self.mdot * self.timestep 


def simulations(datahandler):
    """
    Set up the simulations here.
    """
    endtime = 5e5 |units.yr
    mdot = 1.244e-5 |(units.MSun/units.yr)
    mass1 = 7.659 |units.MSun
    mass2 = 0.001 |units.MSun 

    assert mdot * endtime < mass1

    threebody = Particles()
    twobody1 = new_binary_from_elements(mass1, mass2, 10|units.AU)
    twobody2 = new_binary_from_elements(mass1, mass2, 30|units.AU, eccentricity=0.5)
    threebody.add_particles(twobody1)
    threebody.add_particle(twobody2[1])
    print(threebody)

    for i, dt_param in enumerate(args.dtparams):
        datahandler.prefix = "sim_"+str(i).zfill(2)+"/"

        state = State(mdot, endtime, dt_param)
        evolve_system(threebody, state, datahandler)
        datahandler.file.flush()


def evolve_system(particles, state, datahandler):
    """
    Parameters
    ----------
    particles: a two-body system 
    state: State instance
    datahandler: HDF5HandlerAmuse context manager

    """
    intr = Hermite(nbody_to_si(particles.total_mass(), 1 | units.AU))
    intr.particles.add_particles(particles)

    intr.set_dt_param(state.dt_param)
    state.timestep = intr.get_time_step()
    state.time = intr.get_time()

    energy_begin = intr.get_total_energy()

    counter = 0
    while state.stop is False:
        intr.particles.move_to_center() 

        intr.evolve_model(state.next_time())
        intr.particles[0].mass -= state.massloss()

        if counter%args.storestep == 0: #then store data
            storedata(intr, state, datahandler, energy_begin)
            print(intr.get_time().in_(units.yr), intr.get_time_step().in_(units.day))

        counter += 1

    intr.stop()

def storedata(intr, state, datahandler, energy_begin):
    h = datahandler
    p = intr.particles
    h.append(intr.get_time().in_(units.yr), "time")
    h.append(p.center_of_mass(), "CM_position")
    h.append(p.center_of_mass_velocity(), "CM_velocity")
    h.append(p.position, "position")
    h.append(p.velocity, "velocity")
    h.append(p.mass, "mass")
    h.append(p.kinetic_energy(), "kinetic_energy")
    h.append(p.potential_energy(), "potential_energy")
    h.append(intr.get_total_energy(), "total_energy")
    energy_error = abs((intr.get_total_energy() - energy_begin)/energy_begin)
    h.append(energy_error, "energy_error")

    set1 = Particles()
    set2 = Particles()

    set1.add_particle(p[0])
    set1.add_particle(p[1])

    set2.add_particle(p[0])
    set2.add_particle(p[2])

    for i, pset in enumerate([set1, set2]):
        currentprefix = datahandler.prefix
        datahandler.prefix = currentprefix+"p"+str(i)+"/"

        a, e, i, w, W, f = orbital_elements(pset)
        mu = constants.G*pset.mass.sum() #standard grav. param.
        period = (2*numpy.pi)**2*a**3/mu
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

    parser.add_argument('--mdot', type=float, default=1e-5,
                        help="Masslossrate in MSun/Yr")

    parser.add_argument('--orbitmass', type=float, default=0.01,
                        help="Mass of the orbiting body in MSun")

    parser.add_argument('--endtime', type=float,  default=1000,
                        help="Endtime in yr")

    parser.add_argument('--storestep', type=int,  default=100,
                        help="Store data at every 'storestep'.")

    parser.add_argument('--dtparams', type=float, default=[0.1, 0.2, 0.3], nargs='+',  
                        help="Use like this: --dtparam 0.1 0.2 ")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_arguments()
    print(args)

    with HDF5HandlerAmuse(args.filename) as datahandler:
        simulations(datahandler)


