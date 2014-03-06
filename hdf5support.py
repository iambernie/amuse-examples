#!/usr/bin/env python
# name: hdf5support.py

import unittest
import h5py
import numpy 

from amuse.units.quantities import AdaptingVectorQuantity
from amuse.units.quantities import VectorQuantity
from amuse.units import units


class Results(object):
    """ Container for the results of the hydrodynamics simulation. 
    Instantiate a Results instance by passing it a dictionary of
    VectorQuantities.
 
    Example usage:
    >>> data = dict(masses=[1,2,3]|units.kg, positions=[1,2,3]|units.m)
    >>> results = Results(data)
    >>> results.masses
    quantity<[1, 2, 3] kg>
    >>> results.positions
    quantity<[1, 2, 3] m>
    """
    def __init__(self, keywords):
        """ Generates the attributes of Results using the keywords 
        of the dictionary as attribute names."""
        self.__dict__.update(keywords)

class HDF5Ready(object):
    """ Wraps VectorQuantities/lists/nparrays such that they can be 
    written by write_to_hdf5(). Basically needed to add information 
    as attributes to the hdf5 files. It contains information such as
    the unit of the VectorQuantity."""
    def __init__(self, vq, keyword):
        """
        vq: VectorQuantity
        keyword: 
        """
        self.keyword = keyword
        if isinstance(vq, VectorQuantity):
            self.nparray = vq.value_in(vq.unit)
        elif isinstance(vq, list):
            self.nparray = vq
        elif isinstance(vq, numpy.ndarray):
            self.nparray = vq
        else:
            raise TypeError

        self.attributes = {}
        try:
            self.attributes['unit'] = str(vq.unit.to_reduced_form())
        except AttributeError:
            pass

def write_to_hdf5(filename, data):
    """ Writes the attibutes of Results instance to an hdf5 file."""
    f = h5py.File(filename,'w')

    for keyword in data:
        hdf5ready = HDF5Ready(data[keyword], keyword)
        f[keyword] = hdf5ready.nparray

        for name in hdf5ready.attributes:
            f[keyword].attrs[name] = hdf5ready.attributes[name]
    f.close()
    del f
    print "Written hdf5 file to %s."%filename

def read_from_hdf5(filename):
    """ Reads an hdf5 file and returns a Results instance."""
    f = h5py.File(filename,'r')

    data = {}
    for keyword in f.keys():
        try:
            unitstring = f[keyword].attrs['unit']
            evaluable_unit = parse_unitsstring(unitstring)
            vq = f[keyword].value | eval(evaluable_unit)
            avq = AdaptingVectorQuantity()
            avq.extend(vq)
            data.update({keyword:avq})
        except KeyError:
            data.update({keyword:f[keyword].value})

    f.close()
    del f
    results = Results(data)
    return results

def parse_unitsstring(string):
    """ Parses the unitstring so it can be evaluated with eval to create
    a Quantity."""
    units_splitted = string.split(' * ')
    plus_units = ["units."+elem for elem in units_splitted]
    full_unitstring =  " * ".join(plus_units)
    return full_unitstring


class test_hdf5_units_retrieval(unittest.TestCase):
    """ 
    Tests. 
    """
    
    def setUp(self):
        self.avq = AdaptingVectorQuantity()
        
    def test_m(self):
        vq = range(3) | units.m
        unitstring = vq.unit.to_reduced_form()
        values = vq.value_in(vq.unit)
        evaluable_unit = parse_unitsstring(unitstring)
        self.assertEqual( vq, values | eval(evaluable_unit) )
        
        

    def test_2(self):
        pass



if __name__ == "__main__":
    unittest.main(verbosity=2)



