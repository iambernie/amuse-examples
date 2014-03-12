#!/usr/bin/env python
# name: hdf5utils.py

import os
import unittest
import h5py
import numpy 

class Dataset(object):
    def __init__(self, dset):
        """
        dset: h5py Dataset

        """
        self.dset = dset
        self.chunkcounter = 0  
        self.blockcounter = 0
        self.chunksize = dset.chunks[0]
        self.blocksize = dset.shape[0] 
        self.axis1 = dset.shape[1]

        self.dbuffer = list()


    def append(self, array):
        """
        array: 1D ndarray or list
        """
        # Validation of array. Just wing it for now.
        #
        #if isinstance(array, numpy.ndarray):
        #    assert len(array.shape) == 1
        #elif isinstance(array, list):
        #    assert len(numpy.array(array).shape ) == 1
        #else:
        #    raise ValueError("{} is not 1D".format(type(array)))

        self.dbuffer.append(array)

        if len(self.dbuffer) == self.chunksize: # WRITE BUFFER
            begin = self.blockcounter*self.blocksize + self.chunkcounter*self.chunksize
            end = begin + self.chunksize
            #    print("WRITING TO ADDRESS: [{}:{}, :]".format(begin,end))
            dbuffer_ndarray = numpy.array(self.dbuffer)
            self.dset[begin:end, :] = dbuffer_ndarray
            self.dbuffer = list() # 'clear' buffer

            if end == self.dset.shape[0]: #BLOCK IS FULL --> CREATE NEW BLOCK
                self.dset.resize( (end+self.blocksize, self.axis1) )
                self.blockcounter += 1
                self.chunkcounter = 0 
            else:
                self.chunkcounter += 1

    def flush(self): #TODO
        pass


class HDF5Handler(object):

    def __init__(self, filename, max_length=None, mode='a'):
        """

        Parameters
        ----------
        filename   : filename of the hdf5 file.
        max_length : typically the number of steps.


        Usage should roughly be like:
        -----------------------------

            with HDF5Handler('test.hdf5') as h:
                while condition: #
                    h.append(ndarray, '/grp0/position')
                    h.append(ndarray, '/grp0/velocity')
                    h.append(ndarray, '/grp1/position')
                    h.append(ndarray, '/grp1/velocity')
        
        """
        self.filename = filename
        self.max_length = max_length
        self.mode = mode
        self.index = dict()

    def __enter__(self):
        self.file = h5py.File(self.filename, self.mode) 
        return self

    def __exit__(self, extype, exvalue, traceback):
        self.flushbuffers() 
        self.file.close()
        return False

    def append(self, array, dset_path, **kwargs):
        """ 
        Parameters
        ----------
        array : ndarray or list 
        dset_path  : unix-style path ( 'group/datasetname' )

        """
        if dset_path in self.index:
            self.index[dset_path].append(array)
        else:
            self.create_dset(dset_path, array, **kwargs) 
            self.index[dset_path].append(array)

    def create_dset(self, dset_path, array, **kwargs):
        """
        Define h5py dataset parameters here. 

        IMPORTANT:
            - choose chunkshape wisely (has serious performance implications)
              See h5py docs on chunked storage. 

        Parameters
        ----------
        dset_path: unix-style path

        """
        if isinstance(array, numpy.ndarray):
            axis1 = array.shape[0] 
        elif isinstance(array, list):
            axis1 = len(array) 
        else:
            raise TypeError("{} not supported".format(type(array)))

        chunksize = kwargs['chunks'][0]
        blocksize = 100 * chunksize #FIXME: don't hardcode this, but what is a good blocksize?
                                   # (smaller blocksize means more resizes.)

        dset = self.file.create_dataset(dset_path, shape=(blocksize, axis1), **kwargs)
        self.index.update({dset_path: Dataset(dset) })

    def flushbuffers(self):
        """
        When the number of h.append calls is not a multiple of buffersize, then there 
        will be unwritten arrays in dbuffer, since dbuffer is only written when it is full.
        """
        #TODO: flush 'em 
        pass



class test_HDF5Handler_ndarrays_resizable(unittest.TestCase):
    
    def setUp(self):
        self.filename = 'test.hdf5'
        self.ints = numpy.ones(10000*3).reshape(10000, 3)
        self.floats = numpy.linspace(0, 4123, 10000*3).reshape(10000, 3)

        self.kwargs = dict(chunks=(1000, 3), maxshape=(None, 3)) #choose wisely!

    def test_group_creation(self):
        with HDF5Handler(self.filename) as h:
            for row in self.ints:
                h.append(row, 'testgroup/testset', **self.kwargs)
            self.assertTrue( isinstance(h.file['testgroup'], h5py._hl.group.Group) )
            
    def test_group_and_dataset_creation(self):
        with HDF5Handler(self.filename) as h:
            for row in self.ints:
                h.append(row,'testgroup/testset', **self.kwargs)
            self.assertTrue( isinstance(h.file['testgroup/testset'], h5py._hl.dataset.Dataset) )
            self.assertTrue( isinstance(h.file['testgroup']['testset'], h5py._hl.dataset.Dataset) )

    def test_hdf5file_write_ints(self):
        with HDF5Handler(self.filename) as h:
            for row in self.ints:
                h.append(row, 'test', **self.kwargs) 
            self.assertTrue(isinstance(h.file['test'], h5py._hl.dataset.Dataset))

    def test_hdf5file_write_flts(self):
        with HDF5Handler(self.filename) as h:
            for row in self.floats:
                h.append(row, 'test', **self.kwargs) 
            self.assertTrue(isinstance(h.file['test'], h5py._hl.dataset.Dataset))

    def test_sum_ints(self):
        with HDF5Handler(self.filename) as h:
            for row in self.ints:
                h.append(row, 'test', **self.kwargs) 
            self.assertEqual(numpy.sum(self.ints), h.file['test'].value.sum())
        
    def test_sum_flts_almostequal(self):
        with HDF5Handler(self.filename) as h:
            for row in self.floats:
                h.append(row, 'test', **self.kwargs) 
            self.assertAlmostEqual(numpy.sum(self.floats), h.file['test'].value.sum(), places=5)

    def test_sum_flts_equal(self):
        with HDF5Handler(self.filename) as h:
            for row in self.floats:
                h.append(row, 'test', **self.kwargs) 
            self.assertEqual(numpy.sum(self.floats), h.file['test'].value.sum())

    def test_multiple_datasets(self):
        ndarrA =  numpy.ones(10000*3).reshape(10000, 3)

        with HDF5Handler(self.filename, max_length=10) as h:
            for i in range(10000):
                h.append(ndarrA[i], 'testA', **self.kwargs) 
                h.append(ndarrA[i], 'testB', **self.kwargs) 
                h.append(ndarrA[i], 'testC', **self.kwargs) 
            self.assertEqual(numpy.sum(ndarrA), h.file['testA'].value.sum())
            self.assertEqual(numpy.sum(ndarrA), h.file['testB'].value.sum())
            self.assertEqual(numpy.sum(ndarrA), h.file['testC'].value.sum())
            self.assertEqual(3, len(h.file.keys()) )

        
    def tearDown(self):
        try:
            os.remove(self.filename)
        except OSError:
            pass


if __name__ == "__main__":
    from colored import ColoredTextTestRunner
    unittest.main(verbosity=2, testRunner=ColoredTextTestRunner)



