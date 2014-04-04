#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy

from amuse.units import constants 
from amuse.units import units 
from amuse.units.nbody_system import nbody_to_si
#from amuse.community.mercury.interface import Mercury
from amuse.community.hermite0.interface import Hermite

from ext.hdf5utils import HDF5HandlerAmuse
from ext.misc import orbital_elements
from ext.systems import veras_multiplanet

class MassState(object):
    """
 
    Mass is updated at every dt_external regardless of the internal timestep.

    starttime                                                             endtime
    ==============================================================================
    |      |      |      |      |      |      |      |      |      |      |      |
    ==============================================================================

      dt_i   dt_i   dt_i   dt_i   dt_i   dt_i   dt_i   dt_i   dt_i  
     -----> -----> -----> -----> -----> -----> -----> -----> -----> 

            dt_e                dt_e               dt_e            
     -------------------> -------------------> ------------------->


    starttime                                                             endtime
    ==============================================================================
    |           |        |      |   | | |     |        |          |              | 
    ==============================================================================
            dt_e                dt_e               dt_e            
     -------------------> -------------------> ------------------->

    """
    def __init__(self, timestep, endtime, startmass, mdot, storestep=1000, dt_param=None):
        self.time = 0 |units.yr
        self.endtime = endtime
        self.dt_internal = dt_param
        self.dt_external = timestep 
        self.mass = startmass
        self.mdot = mdot  
        self.stop = False
        self.counter = 0
        self.storestep = storestep

        if mdot*endtime > startmass:
            raise Exception("mdot*endtime = negative endmass.")

    def advance(self):
        self.counter += 1
        self.time += self.dt_external
        self.mass -= self.mdot * self.dt_external #assume mdot > 0

        if self.time > self.endtime:
            self.stop = True

    @property
    def store(self):
        if self.counter % self.storestep == 0:
            return True
        else:
            return False

        


def simulations(datahandler):
    """
    Set up the simulations here.
    """
    datahandler.file.attrs['info'] = "some meta data about simulations in this hdf5file."

    threebody = veras_multiplanet()

    mdot = 1.244e-5 | (units.MSun/units.yr)
    endtime = 5e5 |units.yr
    timesteps = args.timesteps |units.yr
    storestep = args.storestep

    for i, timestep in enumerate(timesteps):
        datahandler.prefix = "sim_"+str(i).zfill(2)+"/"

        # For now, store timestep in a h5py.Dataset f['prefix']['timestep'].
        # But later, store it as metadata in f['prefix'].attrs['meta']
        # Since this is a parameter that doesn't change during 
        # the simulation.
        datahandler.append(timestep, "timestep")

        state = MassState(timestep, endtime, threebody[0].mass, mdot, storestep=storestep)

        evolve_system(threebody, state, datahandler)

        datahandler.file.flush()


def evolve_system(particles, state, datahandler):
    """
    Parameters
    ----------
    particles: a two-body system 
    state: MassState instance
    datahandler: HDF5HandlerAmuse context manager

    """
    intr = Hermite(nbody_to_si(particles.total_mass(), 1 |units.AU))
    intr.particles.add_particles(particles)

    if state.time is None:
        state.time = intr.get_time()

    while state.stop is False:

        intr.evolve_model(state.time)
        intr.particles[0].mass = state.mass

        if state.store is True: 
            storedata(intr, state, datahandler)
            print(state.time)

        state.advance()

    intr.stop()


def storedata(intr, state, datahandler):
    """
    Set up which parameters to store in the hdf5 file here.
    """
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

    p0 = p.copy_to_new_particles()
    p1 = p.copy_to_new_particles()
    p0.remove_particle(p0[2])
    p1.remove_particle(p1[1])

    for i, pset in enumerate([p0, p1]):
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

    parser.add_argument('--storestep', type=int,  default=1000,
                        help="Store data at every 'storestep'.")

    parser.add_argument('--dtparams', type=float, default=[0.1, 0.2, 0.3], nargs='+',  
                        help="Use like this: --dtparam 0.1 0.2 ")

    parser.add_argument('--timesteps', type=float, default=[0.5, 1, 2], nargs='+',  
                        help="Update mass every 'timestep', where timestep is in years.\
                        Use like this: --timesteps 0.5 1.0 2.0 ")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_arguments()
    print(args)

    with HDF5HandlerAmuse(args.filename) as datahandler:
        simulations(datahandler)


