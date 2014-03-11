#!/usr/bin/env python
# name: hdf5support.py

import os
import unittest
import h5py
import numpy 

class HDF5Handler(object):

    def __init__(self, filename, max_length=None, mode='a'):
        """
        filename   : filename of the hdf5 file.
        max_length : typically the number of steps.
        
        """
        self.filename = str(filename)
        self.max_length = max_length
        self.mode = str(mode)
        self.counters = dict()

    def __enter__(self):
        self.file = h5py.File(self.filename, self.mode) 
        return self

    def __exit__(self, extype, exvalue, traceback):
        self.file.close()
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

        TODO: if max_length == None  --> create buffer for dataset
                                         in order to write data to
                                         dataset in chunks.
                  
        """
        f = self.file

        try: #no exceptions if existing right?
            dset = f[group][dsetname]
            assert isinstance(dset, h5py._hl.dataset.Dataset)

            #FIXME: this dictionary usage is ugly. rewrite counter implementation later
            counter = self.counters[group+'/'+dsetname]
            dset[counter] = data  
            self.counters[group+'/'+dsetname] = counter + 1
         
        except KeyError: # group OR dataset doesn't exist.
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

            else: #implement chunked storage
                # FIXME 
                #dset = h5py_group.create_dataset(dsetname, shape=(2, data.shape[0]) )
                #dset[0] = data
                #self.counters[group+'/'+dsetname] = 1
                pass
                
                
class test_HDF5Handler_ndarrays_fixed_shape(unittest.TestCase):
    """
    max_length known.
    """
    def setUp(self):
        self.filename = 'test.hdf5'
        self.ints = numpy.arange(10*3).reshape(10, 3)
        self.floats = numpy.linspace(0, 2, 30).reshape(10, 3)
        #i.e. you will be appending arrays with shape (3,) 10 times.

    def test_open_close(self):
        with HDF5Handler(self.filename) as h:
            pass
        self.assertTrue( os.path.exists(self.filename))

    def test_hdf5file_write_ints(self):
        with HDF5Handler(self.filename, max_length=10) as h:
            for row in self.ints:
                h.store('test', row) 
            self.assertTrue(isinstance(h.file['test'], h5py._hl.dataset.Dataset))

    def test_hdf5file_write_flts(self):
        with HDF5Handler(self.filename, max_length=10) as h:
            for row in self.floats:
                h.store('test', row) 
            self.assertTrue(isinstance(h.file['test'], h5py._hl.dataset.Dataset))

    def test_sum_ints(self):
        with HDF5Handler(self.filename, max_length=10) as h:
            for row in self.ints:
                h.store('test', row) 
            self.assertEqual(numpy.sum(self.ints), h.file['test'].value.sum())
        
    def test_sum_flts_almostequal(self):
        with HDF5Handler(self.filename, max_length=10) as h:
            for row in self.floats:
                h.store('test', row) 
            self.assertAlmostEqual(numpy.sum(self.floats), h.file['test'].value.sum(), places=5)

    def test_sum_flts_equal(self):
        with HDF5Handler(self.filename, max_length=10) as h:
            for row in self.floats:
                h.store('test', row) 
            self.assertEqual(numpy.sum(self.floats), h.file['test'].value.sum())

    def test_multiple_datasets(self):
        ndarrA =  numpy.arange(30).reshape(10, 3)
        ndarrB =  numpy.arange(30, 60).reshape(10, 3)

        with HDF5Handler(self.filename, max_length=10) as h:
            for i in range(10):
                h.store('testA', ndarrA[i]) 
                h.store('testB', ndarrB[i]) 
            self.assertEqual(numpy.sum(ndarrA), h.file['testA'].value.sum())
            self.assertEqual(numpy.sum(ndarrB), h.file['testB'].value.sum())
            self.assertEqual(2, len(h.file.keys()) )

    def test_group_creation(self):
        with HDF5Handler(self.filename, max_length=10) as h:
            for row in self.ints:
                h.store('test', row, group='testgroup')
            self.assertTrue( isinstance(h.file['testgroup'], h5py._hl.group.Group) )
            
    def test_group_and_dataset_creation(self):
        with HDF5Handler(self.filename, max_length=10) as h:
            for row in self.ints:
                h.store('test', row, group='testgroup')
            self.assertTrue( isinstance(h.file['testgroup/test'], h5py._hl.dataset.Dataset) )
            self.assertTrue( isinstance(h.file['testgroup']['test'], h5py._hl.dataset.Dataset) )
        
    def tearDown(self):
        try:
            os.remove(self.filename)
        except OSError:
            pass

class test_HDF5Handler_ndarrays_unlimited(unittest.TestCase):
    
    def setUp(self):
        self.filename = 'test.hdf5'
        self.ints = numpy.arange(10*3).reshape(10, 3)
        self.floats = numpy.linspace(0, 2, 30).reshape(10, 3)

    def test_hdf5file_write_ints(self):
        with HDF5Handler(self.filename) as h:
            for row in self.ints:
                h.store('test', row) 
            self.assertTrue(isinstance(h.file['test'], h5py._hl.dataset.Dataset))

    def test_hdf5file_write_flts(self):
        with HDF5Handler(self.filename) as h:
            for row in self.floats:
                h.store('test', row) 
            self.assertTrue(isinstance(h.file['test'], h5py._hl.dataset.Dataset))

    def test_sum_ints(self):
        with HDF5Handler(self.filename) as h:
            for row in self.ints:
                h.store('test', row) 
            self.assertEqual(numpy.sum(self.ints), h.file['test'].value.sum())
        
    def test_sum_flts_almostequal(self):
        with HDF5Handler(self.filename) as h:
            for row in self.floats:
                h.store('test', row) 
            self.assertAlmostEqual(numpy.sum(self.floats), h.file['test'].value.sum(), places=5)

    def test_sum_flts_equal(self):
        with HDF5Handler(self.filename) as h:
            for row in self.floats:
                h.store('test', row) 
            self.assertEqual(numpy.sum(self.floats), h.file['test'].value.sum())

    def test_multiple_datasets(self):
        ndarrA =  numpy.arange(30).reshape(10, 3)
        ndarrB =  numpy.arange(30, 60).reshape(10, 3)

        with HDF5Handler(self.filename, max_length=10) as h:
            for i in range(10):
                h.store('testA', ndarrA[i]) 
                h.store('testB', ndarrB[i]) 
            self.assertEqual(numpy.sum(ndarrA), h.file['testA'].value.sum())
            self.assertEqual(numpy.sum(ndarrB), h.file['testB'].value.sum())
            self.assertEqual(2, len(h.file.keys()) )

    def test_group_creation(self):
        with HDF5Handler(self.filename) as h:
            for row in self.ints:
                h.store('test', row, group='testgroup')
            self.assertTrue( isinstance(h.file['testgroup'], h5py._hl.group.Group) )
            
    def test_group_and_dataset_creation(self):
        with HDF5Handler(self.filename) as h:
            for row in self.ints:
                h.store('test', row, group='testgroup')
            self.assertTrue( isinstance(h.file['testgroup/test'], h5py._hl.dataset.Dataset) )
            self.assertTrue( isinstance(h.file['testgroup']['test'], h5py._hl.dataset.Dataset) )
        
    def tearDown(self):
        try:
            os.remove(self.filename)
        except OSError:
            pass


if __name__ == "__main__":
    from colored import ColoredTextTestRunner
    unittest.main(verbosity=2, testRunner=ColoredTextTestRunner)



