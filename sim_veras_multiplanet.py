#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy
import time as Time

from amuse.units import constants 
from amuse.units import units 
from amuse.units.nbody_system import nbody_to_si

from amuse.community.mercury.interface import Mercury
from amuse.community.hermite0.interface import Hermite
from amuse.community.ph4.interface import ph4
from amuse.community.huayno.interface import Huayno
from amuse.community.smalln.interface import SmallN

from ext.hdf5utils import HDF5HandlerAmuse
from ext.misc import orbital_elements
from ext.systems import veras_multiplanet
from ext import progressbar as pb

class MassState(object):
    """
 
    Mass is updated at every dt_external (i.e. self.timestep) regardless of the 
    internal timestep of the integrator.

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
    def __init__(self, timestep, endtime, startmass, mdot, datapoints=200, 
                 dt_param=None, name="None"):

        if mdot*endtime > startmass:
            raise Exception("mdot*endtime = negative endmass.")

        self.name = name

        self.starttime = 0 |units.yr
        self.mdot = mdot  
       
        self.timestep = timestep
        self.endtime = endtime
        self.startmass = startmass
        
        self.time_and_mass = self.update()
        self.savepoint = self.savepoints(datapoints)

        self.stopmass_evo = False
        self.stopsave = False

    def update(self):
        mass = self.startmass
        mdot = self.mdot
        time = self.starttime 
        timestep = self.timestep
        endtime = self.endtime
        
        while time != endtime:
            if time + timestep >= endtime:
                mass -= self.mdot *(endtime - time)
                time = endtime
                self.stopmass_evo = True
                yield time, mass

            else:
                time += timestep
                mass -= self.mdot * timestep
                yield time, mass

             
    def savepoints(self, datapoints, verbose=True):
        """ 
        Generator to yield the points in time at which data needs to be saved.

        """
        unit = units.yr
         
        start, stop = self.starttime.value_in(unit), self.endtime.value_in(unit)
        checkpoints = numpy.linspace(start, stop, datapoints)|unit

        pbar = pb.ProgressBar(widgets=self.drawwidget(self.name), 
                              maxval=len(checkpoints)).start()

        for i, cp in enumerate(checkpoints):
            pbar.update(i)
            yield cp

        pbar.finish()
        self.stopsave = True

    def drawwidget(self, proces_discription):
        """ Formats the progressbar. """
        widgets = [proces_discription.ljust(20), pb.Percentage(), ' ',
                   pb.Bar(marker='#',left='[',right=']'),
                   ' ', pb.ETA()]
        return widgets


    @property
    def stop(self):
        if self.stopmass_evo is True and self.stopsave is True:
            return True
        else:
            return False


def simulations(datahandler):
    """
    Set up the simulations here.

    Essentially:
        - decide on a name to use as prefix for each simulation
        - create a MassState() instance. (i.e. define a massloss prescription)
        - call evolve_system()

    """

    datahandler.file.attrs['info'] = "some meta data about simulations in this hdf5file."

    threebody = veras_multiplanet()

    mdot = args.mdot | (units.MSun/units.yr)
    endtime = args.endtime |units.yr
    timesteps = numpy.arange(args.timesteps[0], args.timesteps[1], args.timesteps[2]) |units.yr
    datapoints = args.datapoints

    for intr in [Hermite, SmallN, Huayno]:
        for i, timestep in enumerate(timesteps):
            datahandler.prefix = intr.__name__+"/sim_"+str(i).zfill(2)+"/"
            datahandler.append(timestep, "timestep")
            state = MassState(timestep, endtime, threebody[0].mass, mdot, 
                              datapoints=datapoints, name=datahandler.prefix)
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

    if hasattr(intr, "set_dt_param") and args.dtparam is not None:
        intr.set_dt_param(args.dtparam)
        datahandler.append(args.dtparam, "dtparam")

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

    p0 = p.copy_to_new_particles()
    p1 = p.copy_to_new_particles()
    p0.remove_particle(p0[2])
    p1.remove_particle(p1[1])

    for i, pset in enumerate([p0, p1]):
        currentprefix = datahandler.prefix
        datahandler.prefix = currentprefix+"p"+str(i)+"/"

        a, e, i, w, W, f = orbital_elements(pset)
        mu = constants.G*pset.mass.sum() #standard grav. param.
        period = ((2*numpy.pi)**2*a**3/mu).sqrt()
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

    parser.add_argument('--mdot', type=float, default=1.244e-5,
                        help="Masslossrate in MSun/Yr")

    parser.add_argument('--orbitmass', type=float, default=0.001,
                        help="Mass of the orbiting body in MSun")

    parser.add_argument('--endtime', type=float,  default=5e5,
                        help="Endtime in yr")

    parser.add_argument('--datapoints', type=int,  default=200,
                        help="Number of datapoints.")

    parser.add_argument('--dtparam', type=float, default=None,  
                        help="set_dt_param")

    parser.add_argument('--integrator', default='Hermite', choices=['Hermite','Mercury'],
                        help="Evolve with this integrator.")#TODO: implement choice of integrator

    parser.add_argument('--timesteps', type=float, default=[50, 100, 10], nargs=3,  
                        help="Supply numpy.arange(START, STOP, STEP) arguments to \
                        create an ndarray of timesteps. e.g.: --timesteps 10 100 10")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_arguments()
    print(args)

    with HDF5HandlerAmuse(args.filename) as datahandler:
        simulations(datahandler)


