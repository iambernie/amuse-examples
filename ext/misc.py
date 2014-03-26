#!/usr/bin/env python

from amuse.units import units as u
from amuse.units import constants

def semimajoraxis_from_binary(binary, G=constants.G):
    """ Calculates the semimajoraxis for a binary system. """

    if len(binary) != 2:
        raise Exception("Expects binary")

    else:
        total_mass = binary.mass.sum()
        position = binary[1].position - binary[0].position
        velocity = binary[1].velocity - binary[0].velocity

    specific_energy = (1.0/2.0)*velocity.lengths_squared() - G*total_mass/position.lengths()
    semimajor_axis = -G*total_mass/(2.0*specific_energy)
    return semimajor_axis

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


