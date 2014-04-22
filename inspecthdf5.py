#!/usr/bin/env python

import argparse
import h5py
from ext.misc import printdset

def main():
    f = h5py.File(args.filename, 'r')
    f.visititems(printdset)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', 
                        help="hdf5 file created by sim_veras_multiplanet.py")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_arguments()
    main()

