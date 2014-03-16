#!/usr/bin/env python
# run_tests.py

from amuse.units import units

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
        self.ints = numpy.ones(12345*4).reshape(12345, 4)
        self.floats = numpy.linspace(0, 4123, 10000*3).reshape(10000, 3)
        self.sumints = numpy.sum(self.ints)
        self.sumfloats = numpy.sum(self.floats)

        self.kwargs = dict(chunksize=1000, blockfactor=100) #choose wisely!

    def test_group_creation(self):
        with self.Handler(self.filename) as h:
            for row in self.ints:
                h.append(row, 'testgroup/testset', **self.kwargs)
            self.assertTrue( isinstance(h.file['testgroup'], h5py._hl.group.Group) )

    def test_hdf5file_dataset_creation(self):
        with self.Handler(self.filename) as h:
            for row in self.ints:
                h.append(row, 'test', **self.kwargs) 
            self.assertTrue(isinstance(h.file['test'], h5py._hl.dataset.Dataset))
            
    def test_group_and_dataset_creation(self):
        with self.Handler(self.filename) as h:
            for row in self.ints:
                h.append(row,'testgroup/testset', **self.kwargs)
            self.assertTrue( isinstance(h.file['testgroup/testset'], h5py._hl.dataset.Dataset) )
            self.assertTrue( isinstance(h.file['testgroup']['testset'], h5py._hl.dataset.Dataset) )

    def test_group_creation_after_closing(self):
        with self.Handler(self.filename) as h:
            for row in self.ints:
                h.append(row, 'testgroup/testset', **self.kwargs)

        f = h5py.File(self.filename)
        self.assertTrue( isinstance(f['testgroup'], h5py._hl.group.Group) )
        
    def test_hdf5file_dataset_creation_after_closing(self):
        with self.Handler(self.filename) as h:
            for row in self.ints:
                h.append(row, 'test', **self.kwargs) 
            self.assertTrue(isinstance(h.file['test'], h5py._hl.dataset.Dataset))

        f = h5py.File(self.filename)
        self.assertTrue( isinstance(f['test'], h5py._hl.dataset.Dataset) )
            
    def test_group_and_dataset_creation_after_closing(self):
        with self.Handler(self.filename) as h:
            for row in self.ints:
                h.append(row,'testgroup/testset', **self.kwargs)

        f = h5py.File(self.filename)
        self.assertTrue( isinstance(f['testgroup/testset'], h5py._hl.dataset.Dataset) )
        self.assertTrue( isinstance(f['testgroup']['testset'], h5py._hl.dataset.Dataset) )

    def test_multiple_datasets(self):
        ndarrA =  numpy.ones(10000*3).reshape(10000, 3)

        with self.Handler(self.filename) as h:
            for i in range(10000):
                h.append(ndarrA[i], 'testA', **self.kwargs) 
                h.append(ndarrA[i], 'testB', **self.kwargs) 
                h.append(ndarrA[i], 'testC', **self.kwargs) 

        f = h5py.File(self.filename)
        self.assertEqual(numpy.sum(ndarrA), f['testA'].value.sum())
        self.assertEqual(numpy.sum(ndarrA), f['testB'].value.sum())
        self.assertEqual(numpy.sum(ndarrA), f['testC'].value.sum())
        self.assertEqual(3, len(f.keys()) )
   
    def test_flushbuffers(self):
        ndarr = numpy.ones(12345*3).reshape(12345, 3)

        with self.Handler(self.filename) as h:
            for row in ndarr:
                h.append(row, 'test', **self.kwargs) 

        f = h5py.File(self.filename)
        self.assertEqual(numpy.sum(ndarr), f['test'].value.sum())
 
    def test_trimming(self):
        ndarr = numpy.ones(12345*3).reshape(12345, 3)

        with self.Handler(self.filename) as h:
            for row in ndarr:
                h.append(row, 'test', **self.kwargs) 

        f = h5py.File(self.filename)
        self.assertEqual(ndarr.shape, f['test'].shape)

    def test_flushbuffers_and_trim(self):
        ndarr = numpy.ones(12345*3).reshape(12345, 3)

        with self.Handler(self.filename) as h:
            for row in ndarr:
                h.append(row, 'test', **self.kwargs) 

        f = h5py.File(self.filename)
        self.assertEqual(numpy.sum(ndarr), f['test'].value.sum())
        self.assertEqual(ndarr.shape, f['test'].shape)

    #####################   Value tests  ####################

    def test_sum_ints(self):
        with self.Handler(self.filename) as h:
            for row in self.ints:
                h.append(row, 'test', **self.kwargs) 

        f = h5py.File(self.filename)
        self.assertEqual(self.sumints, f['test'].value.sum())

    def test_sum_flts_almostequal7(self):
        with self.Handler(self.filename) as h:
            for row in self.floats:
                h.append(row, 'test', **self.kwargs) 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats, f['test'].value.sum(), places=7)

    def test_sum_flts_almostequal6(self):
        with self.Handler(self.filename) as h:
            for row in self.floats:
                h.append(row, 'test', **self.kwargs) 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats, f['test'].value.sum(), places=6)

    def test_sum_flts_almostequal5(self):
        with self.Handler(self.filename) as h:
            for row in self.floats:
                h.append(row, 'test', **self.kwargs) 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats, f['test'].value.sum(), places=5)

    def test_sum_flts_almostequal4(self):
        with self.Handler(self.filename) as h:
            for row in self.floats:
                h.append(row, 'test', **self.kwargs) 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats, f['test'].value.sum(), places=4)

    def test_sum_flts_almostequal3(self):
        with self.Handler(self.filename) as h:
            for row in self.floats:
                h.append(row, 'test', **self.kwargs) 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats, f['test'].value.sum(), places=3)

    def test_sum_flts_almostequal2(self):
        with self.Handler(self.filename) as h:
            for row in self.floats:
                h.append(row, 'test', **self.kwargs) 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats, f['test'].value.sum(), places=2)

    def test_sum_flts_almostequal1(self):
        with self.Handler(self.filename) as h:
            for row in self.floats:
                h.append(row, 'test', **self.kwargs) 

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
        self.ints = numpy.ones(12345*4).reshape(12345, 4) |self.unit
        self.floats = numpy.linspace(0, 4123, 10000*3).reshape(10000, 3) |self.unit
        self.sumints = numpy.sum(self.ints.value_in(self.unit))
        self.sumfloats = numpy.sum(self.floats.value_in(self.unit))

        self.kwargs = dict(chunksize=1000, blockfactor=100) #choose wisely!



if __name__ == "__main__":
    from colored import ColoredTextTestRunner
    unittest.main(verbosity=2, testRunner=ColoredTextTestRunner)


