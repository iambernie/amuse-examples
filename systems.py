from amuse.units import units
from amuse.datamodel.particles import Particles


def sun_and_earth():
    """
    Sets up a two-body system representing the sun and earth.
 
    Returns
    -------
    Particles instance representing the sun and earth.

    """
    stars = Particles(2)

    sun = stars[0]
    sun.mass = 1.0 | units.MSun
    sun.position = (0.0, 0.0, 0.0) | units.m
    sun.velocity = (0.0, 0.0, 0.0) | (units.m / units.s)
    sun.radius = 1.0 | units.RSun

    earth = stars[1]
    earth.mass = 5.9736e24 | units.kg
    earth.radius = 6371.0 | units.km
    earth.position = (149.5e6, 0.0, 0.0) | units.km
    earth.velocity = (0.0, 29800, 0.0) | (units.m / units.s)

    return stars



