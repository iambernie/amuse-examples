#!/usr/bin/env python
import numpy
import argparse
import h5py
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
from matplotlib import animation
from amuse.units import units
from ext import progressbar as pb
from ext.misc import quantify_dset
from ext.colors import rundark

def main():
    f = h5py.File(args.filename)
    
    if args.sim:
        sim = f[args.sim]
    else:
        sim = f.values()[0]

    create_mpeg(sim)


def create_mpeg(sim):
    orange = '#FF6500'
    green = '#07D100'

    time_vq = quantify_dset(sim['time'])
    time = time_vq.value_in(units.yr) 

    if args.frames is None:
        frames = range(1, len(time), 5)
    else:
        frames = range(*tuple(args.frames))

    pbar = pb.ProgressBar(widgets=drawwidget(str(sim)), 
                          maxval=frames[-1]).start()

    #seperation_vq  = (position_vq[:, 1, :] - position_vq[:, 2, :]).lengths()
    #seperation = seperation_vq.value_in(units.AU)
    #kinetic_energy = sim['kinetic_energy'].value
    #potential_energy = sim['potential_energy'].value
    #total_energy = sim['total_energy'].value
    #mu0 = quantify_dset(sim['mass'])[0].sum()
    #sma_vq = quantify_dset(sim['p0/sma'])
    #sma = sma_inner_vq.value_in(units.AU)
    #eccentricity = sim['p0/eccentricity'].value
    #true_anomaly = sim['p0/true_anomaly'].value

    position_vq = quantify_dset(sim['position'])
    position = position_vq.value_in(units.AU)

    CM_position_vq = quantify_dset(sim['CM_position'])
    CM_position = CM_position_vq.value_in(units.AU)

    period_vq = quantify_dset(sim['p0/period'])
    period = period_vq.value_in(units.yr)

    mass_vq = quantify_dset(sim['mass'])
    mass = mass_vq.value_in(units.MSun)

    x, y = 0, 1

    if args.restframe == 'cm':
        restframe = CM_position
    elif args.restframe == 'M':
        restframe = position[:, 0, :]
    elif args.restframe == 'm':
        restframe = position[:, 1, :]
    else: #lab frame
        restframe = numpy.zeros(CM_position.size).reshape(CM_position.shape)
 
    central_x = position[:, 0, x] - restframe[:, x]
    central_y = position[:, 0, y] - restframe[:, y]
    orbiting_x = position[:, 1, x] - restframe[:, x]
    orbiting_y = position[:, 1, y] - restframe[:, y] 
    CM_x = CM_position[:, x] - restframe[:, x]
    CM_y = CM_position[:, y] - restframe[:, y]

    #nr_datapoints = len(time)
    #yrs_per_datapoint =  time[-1]/nr_datapoints

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)

    CM_loc, = ax1.plot([], [], c='r', ls="o", mfc="r", mec="r", marker='o', ms=6)
    central_loc, = ax1.plot([], [], c='y', ls="o", mfc="y", mec="y", marker='o', ms=6)
    orbiting_loc, = ax1.plot([], [], c='w', ls="o", mfc="w", mec="w", marker='o', ms=6)

    CM_artist, = ax1.plot([], [], c='r', ls="-", lw=2, alpha=0.8)
    central_artist, = ax1.plot([], [], c=orange, ls="-", lw=2, alpha=0.8)
    orbiting_artist, = ax1.plot([], [], c=green, ls="-", lw=2, alpha=0.8)

    all_x = numpy.hstack([CM_x, central_x, orbiting_x])
    all_y = numpy.hstack([CM_y, central_y, orbiting_y])
    ax1.set_xlim(all_x.min(), all_x.max() )
    ax1.set_ylim(all_y.min(), all_y.max() )
    ax1.set_xlabel('x [AU]')
    ax1.set_ylabel('y [AU]')

    def update_fig(i):
        #t = time[i]
        #lag = int(20*period[i]/(yrs_per_datapoint))
        #orbitlag = int(0.9*period[i]/(yrs_per_datapoint))
        subs = (mass[i,0], mass[i,1], time[i], period[i])
        ax1.set_title('M:{:.2f}  m:{:.2f}  time:{:.2f}  period:{:.2f}'.format(*subs))

        #time_range =  time[0:i]
        CM_artist.set_data(CM_x[0:i], CM_y[0:i])
        central_artist.set_data(central_x[0:i], central_y[0:i])
        orbiting_artist.set_data(orbiting_x[0:i], orbiting_y[0:i])

        try:
            CM_loc.set_data(CM_x[i-1], CM_y[i-1])
            central_loc.set_data(central_x[i-1], central_y[i-1])
            orbiting_loc.set_data(orbiting_x[i-1], orbiting_y[i-1])
        except IndexError:
            pass

        pbar.update(i)

    ffmpeg = animation.FFMpegWriter(fps=args.fps, codec='mpeg4', extra_args=['-vcodec','libx264'])
    anim = animation.FuncAnimation(fig, update_fig, frames=frames, interval=50)

    if args.animate:
        plt.show()
    elif args.output:
        anim.save(args.output, writer=ffmpeg, dpi=args.dpi)


def drawwidget(proces_discription):
    """ Formats the progressbar. """
    widgets = [proces_discription.ljust(20), pb.Percentage(), ' ',
               pb.Bar(marker='#',left='[',right=']'),
               ' ', pb.ETA()]
    return widgets

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', 
                        help="hdf5 file created by sim_veras_multiplanet.py")

    parser.add_argument('-s','--sim', default=None,
                        help="hdf5 path of simulation. e.g. /Hermite/sim_00")

    parser.add_argument('--dpi', type=int, default=80,
                        help="dpi")

    parser.add_argument('-o','--output', default=None,
                        help="output path/filename for mpeg")

    parser.add_argument('-a','--animate', action='store_true', 
                        help="Don't save")

    parser.add_argument('--frames', type=int, default=None, nargs=3,
                        help="range() arguments: range(START, STOP, STEP) to \
                        set frames keyword in FuncAnimation \
                        e.g.: --frames 0 250000 10")

    parser.add_argument('--fps', type=int, default=30,
                        help="FFMpegWriter fps argument")

    parser.add_argument("--restframe", default='cm', choices=['cm', 'lab', 'M', 'm'])

    args = parser.parse_args()
    if args.output is None and args.animate is None:
        raise Exception("--output must be specified or \
                         --animate switch must be on.")
    return args

if __name__ == "__main__":
    args = get_arguments()
    print(args)
    rundark()
    main()

 
