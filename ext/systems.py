import numpy

from amuse.units import constants
from amuse.units import units
from amuse.datamodel.particles import Particles
from amuse.datamodel.particles import Particle

#TODO: absolute path
from misc import new_binary_from_elements

def sun_and_earth():
    """
    Sets up a two-body system representing the sun and earth.
 
    Returns
    -------
    Particles instance representing the sun and earth.

    """
    bodies = Particles(2)

    sun = bodies[0]
    sun.mass = 1.0 | units.MSun
    sun.position = (0.0, 0.0, 0.0) | units.m
    sun.velocity = (0.0, 0.0, 0.0) | (units.m / units.s)
    sun.radius = 1.0 | units.RSun

    earth = bodies[1]
    earth.mass = 5.9736e24 | units.kg
    earth.position = (149.5e6, 0.0, 0.0) | units.km
    earth.velocity = (0.0, 29800, 0.0) | (units.m / units.s)
    earth.radius = 6371.0 | units.km

    return bodies

def twobodies_circular(mass1, mass2, sma):
    """
    Sets up a two-body system with zero eccentricity.
   
    mass1 : mass of central body
    mass2 : mass of orbiting body
    sma : semi-major-axis of the orbit, so here it's just the radius
          of the circular orbit.
 
    Returns
    -------
    Particles instance

    """
    bodies = Particles(2)

    totalmass = mass1+mass2 
    period0 = numpy.sqrt(4.0*numpy.pi**2*sma**3/(constants.G*(totalmass)))
    angular_frequency = 2.0*numpy.pi / period0
    velocity = angular_frequency * sma

    position_orbiting_body = [sma.value_in(sma.unit), 0, 0] | sma.unit
    velocity_orbiting_body =  [0, velocity.value_in(velocity.unit), 0] | velocity.unit

    central = bodies[0]
    central.mass = mass1
    central.position = (0.0, 0.0, 0.0) | units.AU
    central.velocity = (0.0, 0.0, 0.0) | (units.AU / units.s)

    orbitingbody = bodies[1]
    orbitingbody.mass = mass2 
    orbitingbody.position = position_orbiting_body 
    orbitingbody.velocity = velocity_orbiting_body 

    bodies.period0 = period0
    bodies.sma0 = sma

    return bodies

def veras_multiplanet():
    """
    Initial conditions for multi-planet system as described in:
    Veras D, MNRAS 431, 1686-1708 (2013), paragraph 2.2
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

    threebody[0].radius = 363.777818568 |units.RSun 
    threebody[1].radius = 6000 |units.km
    threebody[2].radius = 6000 |units.km

    return threebody

def nbodies(centralmass, *orbiting):
    """
    Parameters
    ----------
    centralmass: mass of the central body
    orbiting: dictionary with args/kwargs to create binary from elements

    """
    bodies = Particles()

    centralbody = Particle(mass = centralmass)
    bodies.add_particle(centralbody) 

    for body in orbiting:
        mass = body['mass']
        elements = body['elements']
        twobody = new_binary_from_elements(centralmass, mass, **elements)
        bodies.add_particle(twobody[1])
        
    return bodies


