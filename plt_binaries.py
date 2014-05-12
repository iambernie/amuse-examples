#!/usr/bin/env python

import argparse
import h5py
import numpy

import matplotlib.pyplot as plt
from matplotlib import cm

from amuse.units import units

from ext.misc import quantify_dset

from itertools import cycle

def main():

    f = h5py.File(args.filename, 'r')

    linecycler_1 = cycle(["-"])
    linecycler_2 = cycle(["-"])
    cmap = cm.jet

    for intr in f.values():
        fig = plt.figure()
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.set_color_cycle(cmap(numpy.linspace(0, 0.95, 10)))
        ax2.set_color_cycle(cmap(numpy.linspace(0, 0.95, 10)))
        ax3.set_color_cycle(cmap(numpy.linspace(0, 0.95, 10)))

        for sim in intr.values():
            time = (quantify_dset(sim['time'])).value_in(units.yr)
            eccentricity = sim['p0/eccentricity'].value
            sma = (quantify_dset(sim['p0/sma'])).value_in(units.AU)
            ml_index = sim['p0/massloss_index'].value

            sma_clipped = sma[numpy.where(sma>0)]
            time_clipped = time[numpy.where(sma>0)]

            ax1.plot(time, eccentricity, lw=1, ls=next(linecycler_1),
                     label="e0= "+str(round(eccentricity[0], 1)))
            ax2.plot(time_clipped, sma_clipped, lw=1, ls=next(linecycler_2),
                     label="e0= "+str(round(eccentricity[0], 1)))
            ax3.plot(time, ml_index, lw=1, ls=next(linecycler_2),
                     label="e0= "+str(round(eccentricity[0], 1)))

        adiabat = sma_analytical(sma[0], 5e-5, time_clipped, 1)
        ax2.plot(time_clipped, adiabat, lw=2, c='k', label='adiabat')

        ax1.set_xlabel('time [yr]')
        ax1.set_ylabel('eccentricity ')
        ax1.set_xlim(0, time[-1])
        ax1.set_ylim(0, 1.3)
        ax1.grid(True)
        ax1.legend(loc='upper left', ncol=2)

        ax2.set_xlabel('time [yr]')
        ax2.set_ylabel('semi-major-axis [AU]')
        ax2.set_xlim(0, time[-1])
        ax2.grid(True)

        ax3.set_xlabel('time [yr]')
        ax3.set_ylabel('massloss_index')
        ax3.set_xlim(0, time[-1])
        ax3.set_yscale('log')
        ax3.grid(True)
       
        if args.ylim2:
            ax2.set_ylim(args.ylim[0], args.ylim[1])
        else:
            ax2.set_ylim(sma_clipped.min(), sma_clipped.max())

        ax2.legend(loc='upper left', ncol=1)

        plt.show()

def sma_analytical(a0, mdot, t, mu0):
    return a0*(1 - mdot*t/mu0)**(-1)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help="hdf5 file created by sim_binaries.py")
    parser.add_argument('--ylim2', type=float, nargs=2, 
                        help="y axis limits")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_arguments()
    print(args)
    main()

 
