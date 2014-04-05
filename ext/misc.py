#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import numpy

from amuse.units import constants, core
from amuse.datamodel.particles import Particles

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

def orbital_elements(binary, G=constants.G, reference_direction=None, 
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
        if i == 0:# => place w at ascending node n
            w = 0
        else:
            #FIXME: possible error here. Tests with inclined orbits are not passing
            w = numpy.arccos((n_.dot(e_)/(n*e)))
            if e_[2] < 0: 
                w = 2*numpy.pi - w
    else: #circular orbit => place w at ascending node n
        w = 0 


    ##################  true anomaly f  ###################
    if e > 0:
        f = numpy.arccos(r_.dot(e_)/(e*r))
        if r_.dot(v_).number < 0:
            f = 2*numpy.pi - f
    elif e == 0:
        f = 0


    if angle == 'degrees':
        #print("e_:{}  n_:{}  h_:{} ".format(e_, n_, h_))
        return a, e, numpy.degrees(i), numpy.degrees(w), numpy.degrees(W),\
               numpy.degrees(f)
    elif angle == 'radians':
        return a, e, i, w, W, f



def new_binary_from_elements(
        mass1,
        mass2,
        semimajor_axis,
        eccentricity = 0,
        true_anomaly = 0,
        inclination = 0,
        longitude_of_the_ascending_node = 0,
        argument_of_periapsis = 0,
        G=constants.G
    ):
    """ 

    Function that returns two-particle Particle set, with the second 
    particle position and velocities computed from the input orbital 
    elements. angles in degrees, inclination between 0 and 180

    Copied from ext.orbital_elements.new_binary_from_elements, but doesn't
    move_to_center.

    """

    inclination = numpy.radians(inclination)
    argument_of_periapsis = numpy.radians(argument_of_periapsis)
    longitude_of_the_ascending_node = numpy.radians(longitude_of_the_ascending_node)
    true_anomaly = numpy.radians(true_anomaly)

    cos_true_anomaly = numpy.cos(true_anomaly)
    sin_true_anomaly = numpy.sin(true_anomaly)

    cos_inclination = numpy.cos(inclination)
    sin_inclination = numpy.sin(inclination)

    cos_arg_per = numpy.cos(argument_of_periapsis)
    sin_arg_per = numpy.sin(argument_of_periapsis)

    cos_long_asc_nodes = numpy.cos(longitude_of_the_ascending_node)
    sin_long_asc_nodes = numpy.sin(longitude_of_the_ascending_node)

    ### alpha is a unit vector directed along the line of node ###
    alphax = cos_long_asc_nodes*cos_arg_per - sin_long_asc_nodes*sin_arg_per*cos_inclination
    alphay = sin_long_asc_nodes*cos_arg_per + cos_long_asc_nodes*sin_arg_per*cos_inclination
    alphaz = sin_arg_per*sin_inclination
    alpha = [alphax,alphay,alphaz]

    ### beta is a unit vector perpendicular to alpha and the orbital angular momentum vector ###
    betax = -cos_long_asc_nodes*sin_arg_per - sin_long_asc_nodes*cos_arg_per*cos_inclination
    betay = -sin_long_asc_nodes*sin_arg_per + cos_long_asc_nodes*cos_arg_per*cos_inclination
    betaz = cos_arg_per*sin_inclination
    beta = [betax,betay,betaz]

#    print 'alpha',alphax**2+alphay**2+alphaz**2 # For debugging; should be 1
#    print 'beta',betax**2+betay**2+betaz**2 # For debugging; should be 1

    ### Relative position and velocity ###
    separation = semimajor_axis*(1.0 - eccentricity**2)/(1.0 + eccentricity*cos_true_anomaly) # Compute the relative separation
    position_vector = separation*cos_true_anomaly*alpha + separation*sin_true_anomaly*beta
    velocity_tilde = (G*(mass1 + mass2)/(semimajor_axis*(1.0 - eccentricity**2))).sqrt() # Common factor
    velocity_vector = -1.0*velocity_tilde*sin_true_anomaly*alpha + velocity_tilde*(eccentricity + cos_true_anomaly)*beta

    result = Particles(2)
    result[0].mass = mass1
    result[1].mass = mass2

    result[1].position = position_vector
    result[1].velocity = velocity_vector

    return result

def quantify_dset(dset):
    if 'unit' in dset.attrs:
        unit = retrieve_unit(dset.attrs['unit'])
        return dset.value | unit

def retrieve_unit(reference_string):
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


