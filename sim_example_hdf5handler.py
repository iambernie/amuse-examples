import os
import numpy
import h5py

from amuse.units import units

from ext.hdf5utils import HDF5Handler
from ext.hdf5utils import HDF5HandlerAmuse

def main():
    append_scalars()
    append_ndarrays()
    append_scalarquantities()
    append_vectorquantities()

def append_scalars():
    filename = 'test.hdf5'

    ones = numpy.ones(10000)

    with HDF5Handler(filename) as h:
        for row in ones:
            h.append(row, 'ones' )

    f = h5py.File(filename)
    print(f['ones'])
    os.remove(filename)


def append_ndarrays():
    filename = 'test.hdf5'

    ones = numpy.ones(10000*2).reshape(10000, 2)

    with HDF5Handler(filename) as h:
        for row in ones:
            h.append(row, 'ones')

    f = h5py.File(filename)
    print(f['ones'])
    os.remove(filename)

def append_scalarquantities():
    filename = 'test.hdf5'

    ones = numpy.ones(10000) |units.kg

    with HDF5HandlerAmuse(filename) as h:
        for sq in ones:
            h.append(sq, 'ones')

    f = h5py.File(filename)
    print(f['ones'], f['ones'].attrs['unit'])
    os.remove(filename)

def append_vectorquantities():
    filename = 'test.hdf5'

    ones = numpy.ones(10000*2).reshape(10000,2) |units.kg

    with HDF5HandlerAmuse(filename) as h:
        for vq in ones:
            h.append(vq, 'ones')

    f = h5py.File(filename)
    print(f['ones'], f['ones'].attrs['unit'])
    os.remove(filename)

if __name__ == "__main__":
    main()

