#!/usr/bin/env python

import argparse
import h5py

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import animation

from amuse.units import units

from ext import progressbar as pb

from ext.misc import quantify_dset
from ext.colors import rundark

def main():
    f = h5py.File(args.filename)
    sim = f[args.sim]
    create_mpeg(sim)


def create_mpeg(sim):

    time_vq = quantify_dset(sim['time'])
    time = time_vq.value_in(units.yr) 

    if args.frames is None:
        frames = range(1, len(time), 5)
    else:
        frames = range(args.frames[0], args.frames[1], args.frames[2])
    pbar = pb.ProgressBar(widgets=drawwidget(args.sim), maxval=frames[-1]).start()


    position_vq = quantify_dset(sim['position'])
    position = position_vq.value_in(units.AU)

    seperation_vq  = (position_vq[:, 1, :] - position_vq[:, 2, :]).lengths()
    seperation = seperation_vq.value_in(units.AU)

    CM_position_vq = quantify_dset(sim['CM_position'])
    CM_position = CM_position_vq.value_in(units.AU)

    period_inner_vq = quantify_dset(sim['p0/period'])
    period_inner = period_inner_vq.value_in(units.yr)

    mass_vq = quantify_dset(sim['mass'])
    mass = mass_vq[:,0].value_in(units.MSun)

    kinetic_energy = sim['kinetic_energy'].value
    potential_energy = sim['potential_energy'].value
    total_energy = sim['total_energy'].value

    mu0 = quantify_dset(sim['mass'])[0].sum()
    #massloss_index_inner = sim['p0/massloss_index'].value
    #massloss_index_outer = sim['p1/massloss_index'].value

    sma_inner_vq = quantify_dset(sim['p0/sma'])
    sma_inner = sma_inner_vq.value_in(units.AU)
    sma_outer_vq = quantify_dset(sim['p1/sma'])
    sma_outer = sma_outer_vq.value_in(units.AU)

    eccentricity_inner = sim['p0/eccentricity'].value
    eccentricity_outer = sim['p1/eccentricity'].value

    true_anomaly_inner = sim['p0/true_anomaly'].value
    true_anomaly_outer = sim['p1/true_anomaly'].value

    sma_inner_analytical_vq = sma_analytical(sma_inner_vq[0], 1.244e-5|(units.MSun/units.yr), time_vq, mu0)
    sma_inner_analytical = sma_inner_analytical_vq.value_in(units.AU)
    sma_outer_analytical_vq = sma_analytical(sma_outer_vq[0], 1.244e-5|(units.MSun/units.yr), time_vq, mu0)
    sma_outer_analytical = sma_outer_analytical_vq.value_in(units.AU)

    #eccentricity_inner_analytical = eccentricity_analytical(eccentricity_inner[0],
    #                                    massloss_index_inner[0], true_anomaly_inner)
    #eccentricity_outer_analytical = eccentricity_analytical(eccentricity_outer[0],
    #                                    massloss_index_outer[0], true_anomaly_outer)

    x, y = 0, 1
    central_x, central_y =  position[:, 0, x] - CM_position[:,x], position[:, 0, y] - CM_position[:,y]
    inner_x, inner_y =  position[:, 1, x] - CM_position[:,x], position[:, 1, y] - CM_position[:,y]
    outer_x, outer_y =  position[:, 2, x] - CM_position[:,x], position[:, 2, y] - CM_position[:,y]

    nr_datapoints = len(time)
    yrs_per_datapoint =  time[-1]/nr_datapoints

    fig = plt.figure(figsize=(22, 12))
    gs = gridspec.GridSpec(4, 4)

    ax1 = fig.add_subplot(gs[:2, :2])
    ax2 = fig.add_subplot(gs[2,0])
    ax3 = fig.add_subplot(gs[3,0])
    ax4 = fig.add_subplot(gs[2,1])
    ax5 = fig.add_subplot(gs[3,1])
    ax6 = fig.add_subplot(gs[0,2:])
    ax7 = fig.add_subplot(gs[1,2:])
    ax8 = fig.add_subplot(gs[2,2:])
    ax9 = fig.add_subplot(gs[3,2:])

    orange = '#FF6500'
    green = '#07D100'

    central_loc, = ax1.plot([], [], c='y', ls="o", mfc="y", mec="y", marker='o', ms=6 )
    inner_loc, = ax1.plot([], [], c='w', ls="o", mfc="w", mec="w", marker='o', ms=4 )
    outer_loc, = ax1.plot([], [], c='w', ls="o", mfc="w", mec="w", marker='o', ms=4 )
    inner_artist, = ax1.plot([], [], c=green, ls="-", lw=2, alpha=0.8)
    outer_artist, = ax1.plot([], [], c='r', ls="-", lw=2, alpha=0.8)
    sma0_ana_artist, = ax2.plot([], [], c='w', ls="-", lw=2)
    sma1_ana_artist, = ax3.plot([], [], c='w', ls="-", lw=2)
    sma0_artist, = ax2.plot([], [], c=green, ls="-", lw=2)
    sma1_artist, = ax3.plot([], [], c='r', ls="-", lw=2)
    ecc0_artist, = ax4.plot([], [], c=green, ls="-", lw=2)
    ecc1_artist, = ax5.plot([], [], c='r', ls="-", lw=2)
    ecc0_ana_artist, = ax4.plot([], [], c='w', ls="-", lw=2)
    ecc1_ana_artist, = ax5.plot([], [], c='w', ls="-", lw=2)
    true_anomaly0_artist, = ax6.plot([], [], c=green, ls="-", lw=2)
    true_anomaly1_artist, = ax6.plot([], [], c='r', ls="-", lw=2)
    seperation_artist, = ax7.plot([], [], c=orange, ls="-", lw=2)
    K_artist, = ax8.plot([], [], c='r', ls="-", lw=2)
    U_artist, = ax8.plot([], [], c=green, ls="-", lw=2)
    E_artist, = ax8.plot([], [], c=orange, ls="-", lw=2)
    mass_artist, = ax9.plot([], [], c=orange, ls="-", lw=2)

    ax2.set_xlabel('time [yr]')
    ax2.set_ylabel('sma [AU]')

    ax3.set_xlabel('time [yr]')
    ax3.set_ylabel('sma [AU]')

    ax4.set_xlabel('time [yr]')
    ax4.set_ylabel('eccentricity ')

    ax5.set_xlabel('time [yr]')
    ax5.set_ylabel('eccentricity ')

    ax6.set_xlabel('time [yr]')
    ax6.set_ylabel('true anomaly [degrees] ')

    ax7.set_xlabel('time [yr]')
    ax7.set_ylabel('seperation [AU]')

    ax8.set_xlabel('time [yr]')
    ax8.set_ylabel('energies')

    ax9.set_ylabel('central mass [MSun]')
    ax9.set_xlabel('time [yr]')

    ax1.set_xlim(-200, 150)
    ax1.set_ylim(-250, 100)

    def update_fig(i):
        t = time[i]
        lag = int(20*period_inner[i]/(yrs_per_datapoint))
        orbitlag = int(0.9*period_inner[i]/(yrs_per_datapoint))

        if i <= lag :
            tbegin = time[0]
            time_range =  time[0:i]
            inner_artist.set_data(inner_x[0:i], inner_y[0:i])
            outer_artist.set_data(outer_x[0:i], outer_y[0:i])
            sma0_artist.set_data(time[0:i], sma_inner[0:i])
            sma1_artist.set_data(time[0:i], sma_outer[0:i])
            sma0_ana_artist.set_data(time[0:i], sma_inner_analytical[0:i])
            sma1_ana_artist.set_data(time[0:i], sma_outer_analytical[0:i])
            ecc0_artist.set_data(time[0:i], eccentricity_inner[0:i])
            ecc1_artist.set_data(time[0:i], eccentricity_outer[0:i])
            #ecc0_ana_artist.set_data(time[0:i], eccentricity_inner_analytical[0:i])
            #ecc1_ana_artist.set_data(time[0:i], eccentricity_outer_analytical[0:i])
            true_anomaly0_artist.set_data(time[0:i], true_anomaly_inner[0:i])
            true_anomaly1_artist.set_data(time[0:i], true_anomaly_outer[0:i])
            K_artist.set_data(time[0:i], kinetic_energy[0:i]) 
            U_artist.set_data(time[0:i], potential_energy[0:i]) 
            E_artist.set_data(time[0:i], total_energy[0:i]) 
            mass_artist.set_data(time[0:i], mass[0:i])
            seperation_artist.set_data(time[0:i], seperation[0:i])
        else:
            tbegin = time[i-lag]
            time_range = time[i-lag:i]

            inner_artist.set_data(inner_x[i-orbitlag:i], inner_y[i-orbitlag:i])
            outer_artist.set_data(outer_x[i-orbitlag:i], outer_y[i-orbitlag:i])
            sma0_artist.set_data(time_range, sma_inner[i-lag:i])
            sma1_artist.set_data(time_range, sma_outer[i-lag:i])
            sma0_ana_artist.set_data(time_range, sma_inner_analytical[i-lag:i])
            sma1_ana_artist.set_data(time_range, sma_outer_analytical[i-lag:i])
            ecc0_artist.set_data(time_range, eccentricity_inner[i-lag:i])
            ecc1_artist.set_data(time_range, eccentricity_outer[i-lag:i])
            #ecc0_ana_artist.set_data(time_range, eccentricity_inner_analytical[i-lag:i])
            #ecc1_ana_artist.set_data(time_range, eccentricity_outer_analytical[i-lag:i])
            true_anomaly0_artist.set_data(time_range, true_anomaly_inner[i-lag:i])
            true_anomaly1_artist.set_data(time_range, true_anomaly_outer[i-lag:i])
            K_artist.set_data(time_range, kinetic_energy[i-lag:i]) 
            U_artist.set_data(time_range, potential_energy[i-lag:i]) 
            E_artist.set_data(time_range, total_energy[i-lag:i]) 
            mass_artist.set_data(time_range, mass[i-lag:i])
            seperation_artist.set_data(time_range, seperation[i-lag:i])

        try:
            central_loc.set_data(central_x[i-1], central_y[i-1])
            inner_loc.set_data(inner_x[i-1], inner_y[i-1])
            outer_loc.set_data(outer_x[i-1], outer_y[i-1])
        except IndexError:
            pass

        ax2.set_ylim(sma_inner_analytical[i]*0.99, sma_inner_analytical[i]*1.01)
        ax3.set_ylim(sma_outer_analytical[i]*0.99, sma_outer_analytical[i]*1.01)
        ax4.set_ylim(eccentricity_inner[i]- 0.01, eccentricity_inner[i] + 0.01)
        ax5.set_ylim(eccentricity_outer[i]- 0.01, eccentricity_outer[i] + 0.01)
        ax6.set_ylim(0, 360)
        ax7.set_ylim(5, 250)
        ax8.set_ylim(-2.5e36, 2e36)
        ax9.set_ylim(mass[i]*0.99, mass[i]*1.01)

        ax2.set_xlim(tbegin, t)
        ax3.set_xlim(tbegin, t)
        ax4.set_xlim(tbegin, t)
        ax5.set_xlim(tbegin, t)
        ax6.set_xlim(tbegin, t)
        ax7.set_xlim(tbegin, t)
        ax8.set_xlim(tbegin, t)
        ax9.set_xlim(tbegin, t)
        pbar.update(i)

    ffmpeg = animation.FFMpegWriter(fps=30, codec='mpeg4', extra_args=['-vcodec','libx264'])
    anim = animation.FuncAnimation(fig, update_fig, frames=frames, interval=50)

    if args.animate:
        plt.show()
    elif args.output:
        anim.save(args.output, writer=ffmpeg, dpi=args.dpi)


def sma_analytical(a0, mdot, t, mu0):
    return a0/(1 - mdot*t/mu0)

#def eccentricity_analytical(e0, phi0, f_in_deg):
#    f_in_rad = numpy.radians(f_in_deg)
#    return e0 + phi0*(1-e0**2)**(3.0/2)*numpy.sin(f_in_rad)/(1-e0*numpy.cos(f_in_rad)) 


def drawwidget(proces_discription):
    """ Formats the progressbar. """
    widgets = [proces_discription.ljust(20), pb.Percentage(), ' ',
               pb.Bar(marker='#',left='[',right=']'),
               ' ', pb.ETA()]
    return widgets


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--filename', required=True,
                        help="hdf5 file created by sim_veras_multiplanet.py")
    parser.add_argument('-s','--sim', required=True,
                        help="hdf5 path of simulation. e.g. /Hermite/sim_00")
    parser.add_argument('--dpi', type=int, default=80,
                        help="dpi")
    parser.add_argument('-o','--output', default=None,
                        help="output path/filename for mpeg")
    parser.add_argument('-a','--animate', action='store_true', 
                        help="Don't save")
    parser.add_argument('--frames', type=int, default=None, nargs=3,
                        help="range() arguments: range(START, STOP, STEP) to \
                        set frames keyword in FuncAnimation e.g.: --frames 0 250000 10")
    args = parser.parse_args()
    if args.output is None and args.animate is None:
        raise Exception("--output must be specified or --animate switch must be on.")
    return args

if __name__ == "__main__":
    args = get_arguments()
    print(args)
    rundark()
    main()

 
