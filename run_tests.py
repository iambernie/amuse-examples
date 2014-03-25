#!/usr/bin/env python
# run_tests.py

from amuse.units import units
from amuse.units import core
from amuse.units.quantities import Quantity
from amuse.units.quantities import VectorQuantity

import os
import h5py
import numpy 
import unittest
from hdf5utils import HDF5Handler
from hdf5utils import HDF5HandlerAmuse
            

class test_HDF5Handler_ndarrays(unittest.TestCase):
    def setUp(self):
        self.Handler = HDF5Handler
        self.filename = 'test.hdf5'

        self.ints1d = numpy.ones(12345*4)
        self.floats1d = numpy.linspace(0, 4123, 10000*3)
        self.ints = numpy.ones(12345*4).reshape(12345, 4)
        self.floats = numpy.linspace(0, 4123, 10000*3).reshape(10000, 3)

        self.sumints1d = numpy.sum(self.ints1d)
        self.sumfloats1d = numpy.sum(self.floats1d)
        self.sumints = numpy.sum(self.ints)
        self.sumfloats = numpy.sum(self.floats)

        #TODO: write a benchmark module to test differen shapes
        self.kwargs = dict(chunksize=1000, blockfactor=100) #choose wisely!

    def test_group_creation(self):
        with self.Handler(self.filename) as h:
            for row in self.ints:
                h.append(row, 'testgroup/testset')
            self.assertTrue( isinstance(h.file['testgroup'], h5py.Group) )

    def test_hdf5file_dataset_creation(self):
        with self.Handler(self.filename) as h:
            for row in self.ints:
                h.append(row, 'test') 
            self.assertTrue(isinstance(h.file['test'], h5py.Dataset))
            
    def test_group_and_dataset_creation(self):
        with self.Handler(self.filename) as h:
            for row in self.ints:
                h.append(row,'testgroup/testset')
            self.assertTrue( isinstance(h.file['testgroup/testset'], h5py.Dataset) )
            self.assertTrue( isinstance(h.file['testgroup']['testset'], h5py.Dataset) )

    def test_group_creation_after_closing(self):
        with self.Handler(self.filename) as h:
            for row in self.ints:
                h.append(row, 'testgroup/testset')

        f = h5py.File(self.filename)
        self.assertTrue( isinstance(f['testgroup'], h5py.Group) )
        
    def test_hdf5file_dataset_creation_after_closing(self):
        with self.Handler(self.filename) as h:
            for row in self.ints:
                h.append(row, 'test') 
            self.assertTrue(isinstance(h.file['test'], h5py.Dataset))

        f = h5py.File(self.filename)
        self.assertTrue( isinstance(f['test'], h5py.Dataset) )
            
    def test_group_and_dataset_creation_after_closing(self):
        with self.Handler(self.filename) as h:
            for row in self.ints:
                h.append(row,'testgroup/testset')

        f = h5py.File(self.filename)
        self.assertTrue( isinstance(f['testgroup/testset'], h5py.Dataset) )
        self.assertTrue( isinstance(f['testgroup']['testset'], h5py.Dataset) )

    def test_multiple_datasets(self):
        ndarrA =  numpy.ones(10000*3).reshape(10000, 3)

        with self.Handler(self.filename) as h:
            for i in range(10000):
                h.append(ndarrA[i], 'testA') 
                h.append(ndarrA[i], 'testB') 
                h.append(ndarrA[i], 'testC') 

        f = h5py.File(self.filename)
        self.assertEqual(numpy.sum(ndarrA), f['testA'].value.sum())
        self.assertEqual(numpy.sum(ndarrA), f['testB'].value.sum())
        self.assertEqual(numpy.sum(ndarrA), f['testC'].value.sum())
        self.assertEqual(3, len(f.keys()) )
   
    def test_flushbuffers(self):
        ndarr = numpy.ones(12345*3).reshape(12345, 3)

        with self.Handler(self.filename) as h:
            for row in ndarr:
                h.append(row, 'test') 

        f = h5py.File(self.filename)
        self.assertEqual(numpy.sum(ndarr), f['test'].value.sum())
 
    def test_trimming(self):
        ndarr = numpy.ones(12345*3).reshape(12345, 3)

        with self.Handler(self.filename) as h:
            for row in ndarr:
                h.append(row, 'test') 

        f = h5py.File(self.filename)
        self.assertEqual(ndarr.shape, f['test'].shape)

    def test_flushbuffers_and_trim(self):
        ndarr = numpy.ones(12345*3).reshape(12345, 3)

        with self.Handler(self.filename) as h:
            for row in ndarr:
                h.append(row, 'test') 

        f = h5py.File(self.filename)
        self.assertEqual(numpy.sum(ndarr), f['test'].value.sum())
        self.assertEqual(ndarr.shape, f['test'].shape)

    #####################   Value tests  ####################

    def test_sum_ints_scalar(self):
        with self.Handler(self.filename) as h:
            for element in self.ints1d:
                h.append(element, 'test') 

        f = h5py.File(self.filename)
        self.assertEqual(self.sumints1d, f['test'].value.sum())

    def test_sum_flts_scalar_almostequal7(self):
        with self.Handler(self.filename) as h:
            for element in self.floats1d:
                h.append(element, 'test') 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats1d, f['test'].value.sum(), places=7)

    def test_sum_flts_scalar_almostequal6(self):
        with self.Handler(self.filename) as h:
            for element in self.floats1d:
                h.append(element, 'test') 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats1d, f['test'].value.sum(), places=6)

    def test_sum_flts_scalar_almostequal5(self):
        with self.Handler(self.filename) as h:
            for element in self.floats1d:
                h.append(element, 'test') 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats1d, f['test'].value.sum(), places=5)


    def test_sum_flts_scalar_almostequal4(self):
        with self.Handler(self.filename) as h:
            for element in self.floats1d:
                h.append(element, 'test') 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats1d, f['test'].value.sum(), places=4)


    def test_sum_flts_scalar_almostequal3(self):
        with self.Handler(self.filename) as h:
            for element in self.floats1d:
                h.append(element, 'test') 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats1d, f['test'].value.sum(), places=3)


    def test_sum_flts_scalar_almostequal2(self):
        with self.Handler(self.filename) as h:
            for element in self.floats1d:
                h.append(element, 'test') 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats1d, f['test'].value.sum(), places=2)


    def test_sum_flts_scalar_almostequal1(self):
        with self.Handler(self.filename) as h:
            for element in self.floats1d:
                h.append(element, 'test') 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats1d, f['test'].value.sum(), places=1)

         
    def test_sum_ints_array(self):
        with self.Handler(self.filename) as h:
            for row in self.ints:
                h.append(row, 'test') 

        f = h5py.File(self.filename)
        self.assertEqual(self.sumints, f['test'].value.sum())

    def test_sum_flts_array_almostequal7(self):
        with self.Handler(self.filename) as h:
            for row in self.floats:
                h.append(row, 'test') 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats, f['test'].value.sum(), places=7)

    def test_sum_flts_array_almostequal6(self):
        with self.Handler(self.filename) as h:
            for row in self.floats:
                h.append(row, 'test') 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats, f['test'].value.sum(), places=6)

    def test_sum_flts_array_almostequal5(self):
        with self.Handler(self.filename) as h:
            for row in self.floats:
                h.append(row, 'test') 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats, f['test'].value.sum(), places=5)

    def test_sum_flts_array_almostequal4(self):
        with self.Handler(self.filename) as h:
            for row in self.floats:
                h.append(row, 'test') 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats, f['test'].value.sum(), places=4)

    def test_sum_flts_array_almostequal3(self):
        with self.Handler(self.filename) as h:
            for row in self.floats:
                h.append(row, 'test') 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats, f['test'].value.sum(), places=3)

    def test_sum_flts_array_almostequal2(self):
        with self.Handler(self.filename) as h:
            for row in self.floats:
                h.append(row, 'test') 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats, f['test'].value.sum(), places=2)

    def test_sum_flts_array_almostequal1(self):
        with self.Handler(self.filename) as h:
            for row in self.floats:
                h.append(row, 'test') 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats, f['test'].value.sum(), places=1)
        
    def tearDown(self):
        try:
            os.remove(self.filename)
        except OSError:
            pass


class test_HDF5HandlerAmuse(test_HDF5Handler_ndarrays):
    def setUp(self):
        self.Handler = HDF5HandlerAmuse
        self.filename = 'amusequantitiestest.hdf5'
        self.unit = units.AU
        self.ints1d = numpy.ones(12345*4) |self.unit
        self.floats1d = numpy.linspace(0, 4123, 10000*3) |self.unit
        self.ints = numpy.ones(12345*4).reshape(12345, 4) |self.unit
        self.floats = numpy.linspace(0, 4123, 10000*3).reshape(10000, 3) |self.unit
        self.sumints1d = numpy.sum(self.ints1d.value_in(self.unit))
        self.sumfloats1d = numpy.sum(self.floats1d.value_in(self.unit))
        self.sumints = numpy.sum(self.ints.value_in(self.unit))
        self.sumfloats = numpy.sum(self.floats.value_in(self.unit))


class test_HDF5HandlerAmuseUnits(unittest.TestCase):
    def setUp(self):
        self.Handler = HDF5HandlerAmuse
        self.filename = 'amuseunits.hdf5'

    def test_attrs_has_unit(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.kg, 'test') 

        f = h5py.File(self.filename)
        self.assertTrue('unit' in f['test'].attrs)

    def test_retrieve_unit_A(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.A, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.A)

    def test_retrieve_unit_AU(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.AU, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.AU)

    def test_retrieve_unit_AUd(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.AUd, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.AUd)

    def test_retrieve_unit_C(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.C, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.C)

    def test_retrieve_unit_E_h(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.E_h, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.E_h)

    def test_retrieve_unit_F(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.F, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.F)

    def test_retrieve_unit_GeV(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.GeV, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.GeV)

    def test_retrieve_unit_Gpc(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.Gpc, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.Gpc)

    def test_retrieve_unit_Gyr(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.Gyr, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.Gyr)

    def test_retrieve_unit_Hz(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.Hz, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.Hz)

    def test_retrieve_unit_J(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.J, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.J)

    def test_retrieve_unit_K(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.K, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.K)

    def test_retrieve_unit_LSun(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.LSun, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.LSun)

    def test_retrieve_unit_MHz(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.MHz, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.MHz)


    def test_retrieve_unit_MJupiter(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.MJupiter, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.MJupiter)

    def test_retrieve_unit_MSun(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.MSun, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.MSun)

    def test_retrieve_unit_MeV(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.MeV, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.MeV)

    def test_retrieve_unit_Mpc(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.Mpc, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.Mpc)

    def test_retrieve_unit_Myr(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.Myr, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.Myr)

    def test_retrieve_unit_N(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.N, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.N)

    def test_retrieve_unit_Pa(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.Pa, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.Pa)

    def test_retrieve_unit_RJupiter(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.RJupiter, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.RJupiter)

    def test_retrieve_unit_Ry(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.Ry, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.Ry)

    def test_retrieve_unit_S(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.S, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.S)

    def test_retrieve_unit_T(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.T, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.T)

    def test_retrieve_unit_V(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.V, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.V)

    def test_retrieve_unit_W(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.W, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.W)

    def test_retrieve_unit_Wb(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.Wb, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.Wb)

    def test_retrieve_unit_amu(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.amu, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.amu)

    def test_retrieve_unit_angstrom(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.angstrom, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.angstrom)


    def test_retrieve_unit_barye(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.barye, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.barye)


    def test_retrieve_unit_cd(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.cd, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.cd)


    def test_retrieve_unit_cm(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.cm, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.cm)


    def test_retrieve_unit_day(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.day, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.day)

    def test_retrieve_unit_e(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.e, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.e)

    def test_retrieve_unit_eV(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.eV, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.eV)

    def test_retrieve_unit_erg(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.erg, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.erg)

    def test_retrieve_unit_g(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.g, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.g)

    def test_retrieve_unit_gyr(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.gyr, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.gyr)

    def test_retrieve_unit_hour(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.hour, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.hour)

    def test_retrieve_unit_julianyr(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.julianyr, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.julianyr)

    def test_retrieve_unit_kg(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.kg, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.kg)

    def test_retrieve_unit_km(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.km, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.km)

    def test_retrieve_unit_kms(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.kms, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.kms)

    def test_retrieve_unit_kpc(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.kpc, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.kpc)

    def test_retrieve_unit_lightyear(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.lightyear, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.lightyear)

    def test_retrieve_unit_m(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.m, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.m)

    def test_retrieve_unit_minute(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.minute, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.minute)

    def test_retrieve_unit_mol(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.mol, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.mol)

    def test_retrieve_unit_ms(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.ms, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.ms)

    def test_retrieve_unit_myr(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.myr, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.myr)

    def test_retrieve_unit_ohm(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.ohm, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.ohm)

    def test_retrieve_unit_parsec(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.parsec, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.parsec)

    def test_retrieve_unit_rad(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.rad, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.rad)

    def test_retrieve_unit_s(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.s, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.s)

    def test_retrieve_unit_sr(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.sr, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.sr)

    def test_retrieve_unit_tesla(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.tesla, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.tesla)

    def test_retrieve_unit_weber(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.weber, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.weber)

    def test_retrieve_unit_yr(self):
        with self.Handler(self.filename) as h:
             h.append(1 | units.yr, 'test') 

        f = h5py.File(self.filename)
        unit = eval(f['test'].attrs['unit'], core.__dict__)
        self.assertTrue(unit == units.yr)

    def test_retrieve_quantity(self):
        with self.Handler(self.filename) as h:
             h.append(1|units.kg, 'test') 

        f = h5py.File(self.filename)
        dset = f['test']
        ndarr = dset.value
        unit = eval(dset.attrs['unit'], core.__dict__)
        vectorquantity = ndarr | unit
        self.assertTrue(vectorquantity, VectorQuantity)

    def tearDown(self):
        try:
            os.remove(self.filename)
        except OSError:
            pass


if __name__ == "__main__":
    from colored import ColoredTextTestRunner
    unittest.main(verbosity=2, testRunner=ColoredTextTestRunner)



