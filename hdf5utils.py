#!/usr/bin/env python
# name: hdf5support.py

import os
import unittest
import h5py
import numpy 

from amuse.units.quantities import AdaptingVectorQuantity
from amuse.units.quantities import VectorQuantity
from amuse.units import units

class HDF5Handler(object):

    def __init__(self, filename, chunksize=10000, max_length=None, mode='a'):
        """
        filename   : filename of the hdf5 file.
        max_length : typically the number of steps.
        
        """
        self.filename = str(filename)
        self.chunksize = int(chunksize)
        self.max_length = max_length
        self.mode = str(mode)
        self.counters = dict()

    def __enter__(self):
        self.hdf5file = h5py.File(self.filename, self.mode) 
        return self

    def __exit__(self, extype, exvalue, traceback):
        """
        close hdf5 file
        check if units can be retrieved
        if unit retrieval fails, raise an exception

        """
        self.hdf5file.close()
        return False

    def store(self, dsetname, data, group='/', dtype=None):
        """
        Write (vectorquantity | ndarray | list) to open hdf5 file.
        
        Does the dataset exist? 
             No --> create dataset in group
                 if max_length:
                     create fixed_length dataset
                 else:
                     create dataset with chunked storage 
                     recommended chunksizes: 10 KiB < size < 1 MiB
             Yes --> extend dataset
                  
        """
        f = self.hdf5file

        try: #no exceptions if existing right?
            dset = f[group][dsetname]
            assert isinstance(dset, h5py._hl.dataset.Dataset)

            #this dictionary usage is ugly. rewrite counter implementation later
            counter = self.counters[group+'/'+dsetname]
            dset[counter] = data #counter should be >= 1 
            self.counters[group+'/'+dsetname] = counter + 1
         
        except KeyError: #if group OR dataset doesn't exist.

            try:
                h5py_group = f.create_group(group)
            except ValueError: #group exist 
                h5py_group = f[group]
            finally:
                assert isinstance(h5py_group, h5py._hl.group.Group )

            if self.max_length:
                dset = h5py_group.create_dataset(dsetname, shape=(self.max_length, data.shape[0]) )
                dset[0] = data
                self.counters[group+'/'+dsetname] = 1
                

class test_HDF5Handler_base(unittest.TestCase):
    """
    Base testing class.

    """
    def setUp(self):
        self.filename = 'test.hdf5'
        self.dsetclass = h5py._hl.dataset.Dataset
        self.groupclass = h5py._hl.group.Group


    def test_open_close(self):
        with HDF5Handler(self.filename) as h:
            pass
        #file should be closed here.
        self.assertTrue( os.path.exists(self.filename) )

    def test_opening_opened_file(self):
        f = h5py.File(self.filename)
        # see note on thread-safety below

    def tearDown(self):
        try:
            os.remove(self.filename)
        except OSError:
            pass


class test_HDF5Handler_ndarrays_store_ints(test_HDF5Handler_base):
    def test_hdf5file_write_ints(self):
        dsetname = 'int_dataset'
        ndarr_ints =  numpy.arange(10*3).reshape(10, 3)

        with HDF5Handler(self.filename, max_length=10 ) as h:
            for i in range(10):
                h.store(dsetname, ndarr_ints[i]) 

            self.assertTrue(isinstance(h.hdf5file[dsetname], self.dsetclass))
        #also check number of items in '/'

class test_HDF5Handler_ndarrays_store_flts(test_HDF5Handler_base):
    def test_hdf5file_write_flts(self):
        dsetname = 'flt_dataset'
        ndarr_flts =  numpy.linspace(0, 2, 30).reshape(10, 3)

        with HDF5Handler(self.filename, max_length=10 ) as h:
            for i in range(10):
                h.store(dsetname, ndarr_flts[i]) 

            self.assertTrue(isinstance(h.hdf5file[dsetname], self.dsetclass))
        #also check number of items in '/'

        

class test_HDF5Handler_ndarrays_read(test_HDF5Handler_base):
    """ 
    Tests if created hdf5file can be read and tests for some correctness.

    """
        
    def test_sum_ints(self):
        dsetname = 'int_dataset'
        ndarr_ints =  numpy.arange(10*3).reshape(10, 3)
        arraysum = numpy.sum(ndarr_ints)

        with HDF5Handler(self.filename, max_length=10 ) as h:
            for i in range(10):
                h.store(dsetname, ndarr_ints[i]) 

        f = h5py.File(self.filename)
        self.assertEqual(arraysum, f[dsetname].value.sum())
        

    def test_sum_flts_almostequal(self):
        dsetname = 'flt_dataset'
        ndarr_flts =  numpy.linspace(0, 2, 30).reshape(10, 3)
        arraysum = numpy.sum(ndarr_flts)

        with HDF5Handler(self.filename, max_length=10 ) as h:
            for i in range(10):
                h.store(dsetname, ndarr_flts[i]) 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(arraysum, f[dsetname].value.sum(), places=4)
        #do i need to close here for the next test?
        # This has something to do with concurrency and whether h5py is thread-safe.


    def test_sum_flts_equal(self):
        dsetname = 'flt_dataset'
        ndarr_flts =  numpy.linspace(0, 2, 30).reshape(10, 3)
        arraysum = numpy.sum(ndarr_flts)

        with HDF5Handler(self.filename, max_length=10 ) as h:
            for i in range(10):
                h.store(dsetname, ndarr_flts[i]) 

        f = h5py.File(self.filename)

        self.assertEqual(arraysum, f[dsetname].value.sum())


if __name__ == "__main__":
    from colored import ColoredTextTestRunner
    unittest.main(verbosity=2, testRunner=ColoredTextTestRunner)



