#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import numpy
import argparse

from amuse.community.mercury.interface import Mercury
from amuse.community.hermite0.interface import Hermite
from amuse.community.ph4.interface import ph4
from amuse.community.huayno.interface import Huayno
from amuse.community.smalln.interface import SmallN

from amuse.units import units, constants, core
from amuse.datamodel.particles import Particles
from amuse.ext import orbital_elements as orbital_elements_amuse
import progressbar as pb

def args_quantify(unit):
    class QuantifyAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            quantity = values | unit
            setattr(namespace, self.dest, quantity)
    return QuantifyAction

def args_integrators():
    valid_integrators = dict(Mercury=Mercury, Hermite=Hermite, ph4=ph4, Huayno=Huayno,
                             SmallN=SmallN)

    class IntegratorsAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
           
            integrators = []

            for name in values: 
                if name in valid_integrators:
                    integrators.append(valid_integrators[name])
                else:
                    raise Exception('Invalid integrator: {}'.format(name))
                
            setattr(namespace, self.dest, integrators)
    return IntegratorsAction
    


class Parameters(object):
    def __init__(self, hdf5filename):
        self.hdf5filename = hdf5filename
        self.hdf5file = h5py.File(hdf5filename, 'r')

    def available_integrators(self):
        return self.hdf5file.keys()


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
    def __init__(self, timestep, endtime, startmass, mdot, datapoints=200,
                 name="None"):

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
                mass -= mdot *(endtime - time)
                time = endtime
                self.stopmass_evo = True
                yield time, mass

            else:
                time += timestep
                mass -= mdot * timestep
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


class VariableMassState(MassState):
    def __init__(self, endtime, startmass, mdot, datapoints=200,
                 eta=0.1, name="None"):

        if mdot*endtime > startmass:
            raise Exception("mdot*endtime = negative endmass.")

        self.eta = eta
        self.intr = None

        self.name = name

        self.starttime = 0 |units.yr
        self.mdot = mdot

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
        eta = self.eta
        endtime = self.endtime
        intr = self.intr

        while time != endtime:
            #sma = semimajoraxis_from_binary(intr.particles)
            #period = ((2*numpy.pi)**2*sma**3/(constants.G*total_mass)).sqrt()
            total_mass = intr.particles.mass.sum()
            timestep = total_mass/mdot*eta
             
            if time + timestep >= endtime:
                mass -= mdot *(endtime - time)
                time = endtime
                self.stopmass_evo = True
                yield time, mass

            else:
                
                time += timestep
                mass -= mdot * timestep
                yield time, mass
 
      


def semimajoraxis_from_binary(binary, G=constants.G):
    """ Calculates the semimajoraxis of a binary system. """

    if len(binary) != 2:
        raise Exception("Expects binary")

    else:
        total_mass = binary.mass.sum()
        position = binary[1].position - binary[0].position
        velocity = binary[1].velocity - binary[0].velocity

    specific_energy = (1.0/2.0)*velocity.lengths_squared() - G*total_mass/position.lengths()
    semimajor_axis = -G*total_mass/(2.0*specific_energy)
    return semimajor_axis

def eccentricity_from_binary(binary, G=constants.G):
    """ Calculates the eccentricity of a binary system. """
    if len(binary) != 2:
        raise Exception("Expects binary")

    else:
        total_mass = binary.mass.sum()
        position = binary[1].position - binary[0].position
        velocity = binary[1].velocity - binary[0].velocity

    specific_energy = (1.0/2.0)*velocity.lengths_squared() - G*total_mass/position.lengths()

    specific_angular_momentum = position.cross(velocity)
    eccentricity_argument = 2*specific_angular_momentum.lengths()**2 * specific_energy/(G**2*total_mass**2)
    if (eccentricity_argument <= -1): 
        eccentricity = 0.0
    else: 
        eccentricity = numpy.sqrt(1.0 + eccentricity_argument)
    return eccentricity


def orbital_elements(*args, **kwargs):
    if 'G' not in kwargs:
        kwargs['G'] = constants.G

    m1, m2, a, e, f, i, W, w = orbital_elements_amuse.orbital_elements_from_binary(*args, **kwargs)

    if f < 0:
        f += 360
    if i < 0:
        i += 360
    if w < 0:
        w += 360
    if W < 0:
        W += 360

    if 'angle' in kwargs:
        if kwargs['angle'] == 'radians':
            i_rad = numpy.radians(i)
            W_rad = numpy.radians(W)
            w_rad = numpy.radians(w)
            f_rad = numpy.radians(f)
            return a, e, i_rad, w_rad, W_rad, f_rad
        else:
            return a, e, i, w, W, f
    else:
        return a, e, i, w, W, f 
    

def get_orbital_elements(binary, G=constants.G, reference_direction=None, 
                     angle='degrees'):
    """ 
    Calculates the orbital elements of a binary system. 

    In this function, variables ending with an underscore indicate
    that they are vectors.

    energy := specific orbital energy or vis-viva energy, as it is defined
              at http://en.wikipedia.org/wiki/Specific_orbital_energy

    r_  :=  [rx, ry, rz]                     (relative position vector) 
    v_  :=  [vx, vy, vz]                     (relative velocity vector)
    h_  :=  r_ cross v_                     (specific angular momentum)
    mu  :=  G*binary.mass.sum()      (standard gravitational parameter)
    e_  :=  (v_ cross h_)/mu - r_/r               (eccentricity vector)
    n_  :=  k_ cross h_         (vector pointing towardsascending node)


    Parameters
    ----------
    binary : amuse.datamodel.particles.Particles instance
    G: Gravitational Constant
    reference_direction: shape (3,) array.


    Returns
    -------
    The orbital elements: a, e, i, w, W, f, which stand for:

    a: semi-major axis
    e: eccentricity
    i: inclination
    w: argument of periapsis
    W: longitude of ascending node
    f: true anomomaly

    """
    if len(binary) != 2:
        raise Exception("Expects binary")

    if angle not in ['degrees', 'radians']:
        raise Exception("invalid argument for angle keyword")

    #TODO: calculate from arbitrary reference direction
    # this means redefining n_ = k_ cross h_ ?
    if reference_direction is None:
        reference_direction = numpy.array([1, 0, 0])
    
    r_ = binary[1].position - binary[0].position
    v_ = binary[1].velocity - binary[0].velocity
    r = r_.length()
    v = v_.length()

    mu = G*binary.mass.sum()
    energy = 0.5*v**2 - mu/r

    ###########  specific angular momentum  ##############
    h_ = r_.cross(v_)
    h = h_.length() 

    ########  semimajoraxis a and eccentricity e  ########
    a = -mu/(2*energy)
    e_ = v_.cross(h_)/mu - r_/r 
    e = numpy.linalg.norm(e_) 
    #e = numpy.sqrt(1 + 2*energy*(h/mu)**2)

    ###################  inclination i  ##################
    i = numpy.arccos(h_.z/h)


    #################  ascending node n_  ################
    n_ = -1*h_.cross([0,0,1]) 
    n = n_.length()


    ##########  longtitude of ascending node W  ##########
    if n.number == 0:
        n_ = reference_direction
        W = 0
    else:
        if n_[1].number >= 0:
            W = numpy.arccos(n_[0]/n)
        else:
            W = 2*numpy.pi * numpy.arccos(n_[0]/n)


    ##############  argument of periapsis w  #############
    # Note that n_ is not unitless by definition
    if e > 0:
        if i == 0:# equatorial orbit 
            w = numpy.arctan2(e_[1], e_[0])
            if (r_.cross(v_))[2].number < 0:
                w = 2*numpy.pi - w
        else:
            w = numpy.arccos((n_.dot(e_)/(n*e)))
            if e_[2] < 0: 
                w = 2*numpy.pi - w
    else: #circular orbit => place w at ascending node n
        w = 0 


    ##################  true anomaly f  ###################
    if e != 0:
        arccos_arg = r_.dot(e_)/(e*r)

        #arccos(x) only defined for -1<x<1, but due to rounding
        #errors this can be greater than 1 or less than -1.
        if nearly_equal(1, arccos_arg):
            arccos_arg = 1
        elif nearly_equal(-1, arccos_arg):
            arccos_arg = -1

        f = numpy.arccos(arccos_arg)

        if r_.dot(v_).number < 0:
            f = 2*numpy.pi - f
    elif e == 0 and i != 0:
        f = numpy.arccos(n_.dot(r_)/(n*r))
        if (n_.dot(v_)).number > 0:
            f = 2*numpy.pi - f
    elif e == 0 and i == 0:
        f = numpy.arccos(r_[0]/r)
        if (v_[0]).number > 0:
            f = 2*numpy.pi - f
    else:
        raise Exception('true anomaly could not be determined')


    if angle == 'degrees':
        i_in_degrees = numpy.degrees(i) % 360.0
        w_in_degrees = numpy.degrees(w) % 360.0
        W_in_degrees = numpy.degrees(W) % 360.0
        f_in_degrees = numpy.degrees(f) % 360.0

        return a, e, i_in_degrees, w_in_degrees, W_in_degrees,\
               f_in_degrees 
    elif angle == 'radians':
        return a, e, i, w, W, f


def new_binary_from_elements(*args, **kwargs):
    """ amuse.ext.orbital_elements.new_binary_from_elements places the center 
    of mass at the origin. This function reverts the transformation."""
    if 'G' not in kwargs:
        kwargs['G'] = constants.G
    binary = orbital_elements_amuse.new_binary_from_orbital_elements(*args, **kwargs)
    binary.position += binary.center_of_mass() 
    binary.velocity += binary.center_of_mass_velocity()
    return binary

def nearly_equal(a,b,sig_fig=4):
    return (a==b or int(a*10**sig_fig) == int(b*10**sig_fig))

def quantify_dset(dset): 
    if 'unit' in dset.attrs:
        unit = evalrefstring(dset.attrs['unit'])
        return dset.value | unit

def retrieve_unit(dset):
    if "unit" in dset.attrs:
        reference_string = dset.attrs['unit']
        return eval(reference_string, core.__dict__)
    else:
        raise Exception("Dataset.attrs has no unit keyword")

def evalrefstring(reference_string):
    return eval(reference_string, core.__dict__)

def printshape(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(name, obj.shape)

def printunit(name, obj):
    if isinstance(obj, h5py.Dataset):
        if 'unit' in obj.attrs:
            print(name, obj.attrs['unit'])

def printdset(name, obj):
    if isinstance(obj, h5py.Dataset):
        if "unit" in obj.attrs:
            unit = eval(obj.attrs['unit'], core.__dict__)
            print(name, obj.shape, unit)
        else:
            print(name, obj.shape)


