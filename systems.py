import numpy

from amuse.units import constants
from amuse.units import units
from amuse.datamodel.particles import Particles

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
    velocity_orbiting_body =  [velocity.value_in(velocity.unit), 0, 0] | velocity.unit

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

def twobodies(mass1, mass2):
    #TODO: generalize twobodies_circular for eccentric orbits
    pass


