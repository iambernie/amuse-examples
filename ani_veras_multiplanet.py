#!/usr/bin/env python

import argparse
import h5py

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import animation

from amuse.units import units

#from ext import progressbar

from ext.misc import quantify_dset
from ext.colors import rundark

def main():
    f = h5py.File(args.filename)
    sim = f[args.sim]
    create_mpeg(sim)


def create_mpeg(sim):

    time_vq = quantify_dset(sim['time'])
    time = time_vq.value_in(units.yr) 

    position_vq = quantify_dset(sim['position'])
    position = position_vq.value_in(units.AU)

    seperation_vq  = (position_vq[:, 1, :] - position_vq[:, 2, :]).lengths()
    seperation = seperation_vq.value_in(units.AU)

    CM_position_vq = quantify_dset(sim['CM_position'])
    CM_position = CM_position_vq.value_in(units.AU)

    #mu0 = quantify_dset(sim['mass'])[0].sum()
    #sma_an_vq = sma_analytical(sma_vq[0], 1.244e-5|(units.MSun/units.yr), time_vq, mu0)
    #sma_an = sma_an_vq.value_in(units.AU)

    mass_vq = quantify_dset(sim['mass'])
    mass = mass_vq[:,0].value_in(units.MSun)

    kinetic_energy = sim['kinetic_energy'].value
    potential_energy = sim['potential_energy'].value
    total_energy = sim['total_energy'].value

    sma_inner_vq = quantify_dset(sim['p0/sma'])
    sma_inner = sma_inner_vq.value_in(units.AU)
    sma_outer_vq = quantify_dset(sim['p1/sma'])
    sma_outer = sma_outer_vq.value_in(units.AU)
    eccentricity_inner = sim['p0/eccentricity'].value
    eccentricity_outer = sim['p1/eccentricity'].value
    true_anomaly_inner = sim['p0/true_anomaly'].value
    true_anomaly_outer = sim['p1/true_anomaly'].value

    x, y = 0, 1
    central_x, central_y =  position[:, 0, x] - CM_position[:,x], position[:, 0, y] - CM_position[:,y]
    inner_x, inner_y =  position[:, 1, x] - CM_position[:,x], position[:, 1, y] - CM_position[:,y]
    outer_x, outer_y =  position[:, 2, x] - CM_position[:,x], position[:, 2, y] - CM_position[:,y]

    #total_frames = len(time)
    lag = 100

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

    central_loc, = ax1.plot([], [], c='y', ls="o", mfc="y", mec="y", marker='o', ms=6 )
    inner_loc, = ax1.plot([], [], c='w', ls="o", mfc="w", mec="w", marker='o', ms=4 )
    outer_loc, = ax1.plot([], [], c='w', ls="o", mfc="w", mec="w", marker='o', ms=4 )
    inner_artist, = ax1.plot([], [], c='#FF6500', ls="-", lw=2, alpha=0.8)
    outer_artist, = ax1.plot([], [], c='r', ls="-", lw=2, alpha=0.8)
    sma0_artist,  = ax2.plot([], [], **line.orange)
    sma1_artist,  = ax3.plot([], [], **line.red)
    ecc0_artist,  = ax4.plot([], [], **line.orange)
    ecc1_artist,  = ax5.plot([], [], **line.red)
    true_anomaly0_artist,  = ax6.plot([], [], **line.orange)
    true_anomaly1_artist,  = ax6.plot([], [], **line.red)
    seperation_artist, = ax7.plot([], [], **line.yellow)
    K_artist, = ax8.plot([], [], **line.cyan)
    U_artist, = ax8.plot([], [], **line.magenta)
    E_artist, = ax8.plot([], [], **line.yellow)
    mass_artist, = ax9.plot([], [], **line.yellow)

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

    #ax1.set_xlim(-60, 30)
    #ax1.set_ylim(-50, 50)
    ax1.set_xlim(-200, 150)
    ax1.set_ylim(-250, 100)


    def update_fig(i):
        t = time[i]

        if i <= lag :
            tbegin = time[0]
            time_range =  time[0:i]
            inner_artist.set_data(inner_x[0:i], inner_y[0:i])
            outer_artist.set_data(outer_x[0:i], outer_y[0:i])
            sma0_artist.set_data(time[0:i], sma_inner[0:i])
            sma1_artist.set_data(time[0:i], sma_outer[0:i])
            ecc0_artist.set_data(time[0:i], eccentricity_inner[0:i])
            ecc1_artist.set_data(time[0:i], eccentricity_outer[0:i])
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

            inner_artist.set_data(inner_x[i-lag:i], inner_y[i-lag:i])
            outer_artist.set_data(outer_x[i-lag:i], outer_y[i-lag:i])
            sma0_artist.set_data(time_range, sma_inner[i-lag:i])
            sma1_artist.set_data(time_range, sma_outer[i-lag:i])
            ecc0_artist.set_data(time_range, eccentricity_inner[i-lag:i])
            ecc1_artist.set_data(time_range, eccentricity_outer[i-lag:i])
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

        ax2.set_ylim(sma_inner[i]*0.99, sma_inner[i]*1.01)
        ax3.set_ylim(sma_outer[i]*0.99, sma_outer[i]*1.01)
        ax4.set_ylim(eccentricity_inner[i]*0.99, eccentricity_inner[i]*1.01)
        ax5.set_ylim(eccentricity_outer[i]*0.99, eccentricity_outer[i]*1.01)
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
        print("i",i)

    ffmpeg = animation.FFMpegWriter(fps=30, codec='mpeg4', extra_args=['-vcodec','libx264'])
    anim = animation.FuncAnimation(fig, update_fig, frames=range(200000, 250000, 10), interval=100)

    if args.animate:
        plt.show()
    elif args.output:
        anim.save(args.output, writer=ffmpeg)

def sma_analytical(a0, mdot, t, mu0):
    return a0*(1 - mdot*t/mu0)**(-1)

class Line(object):
    def __init__(self):
        orange = '#FF6500'
        green = '#07D100'
        lightblue = '#00C8FF'
        blue = '#0049FF'
        purple = '#BD00FF'
        self.red = dict(c='r', ls="-", lw=1, alpha=1.0)
        self.orange = dict(c=orange, ls="-", lw=1, alpha=1.0)
        self.yellow  = dict(c='y', ls="-", lw=1, alpha=1.0)
        self.green = dict(c=green, ls="-", lw=1, alpha=1.0)
        self.purple = dict(c=purple, ls="-", lw=1, alpha=1.0)
        self.lightblue = dict(c=lightblue, ls="-", lw=1, alpha=1.0)
        self.cyan = dict(c='c', ls="-", lw=1, alpha=1.0)
        self.blue = dict(c=blue, ls="-", lw=1, alpha=1.0)
        self.magenta = dict(c='m', ls="-", lw=1, alpha=1.0)
        self.white = dict(c='w', ls="-", lw=1, alpha=1.0)
        self.black = dict(c='k', ls="-", lw=1, alpha=1.0)

class Dots(object):
    def __init__(self):
        orange = '#FF6500'
        green = '#07D100'
        lightblue = '#00C8FF'
        blue = '#0049FF'
        purple = '#BD00FF'
        self.red = dict(c='r', ls="o", mfc="r", mec="r", marker='o', alpha=1.0, ms=1)
        self.orange  = dict(c=orange, ls="o", mfc=orange, mec=orange,  marker='o', alpha=1.0, ms=1)
        self.yellow  = dict(c='y', ls="o", mfc="y", mec="y", marker='o', alpha=1.0, ms=1)
        self.green = dict(c=green, ls="o", mfc=green, mec=green, marker='o', alpha=1.0, ms=1)
        self.purple = dict(c=purple, ls="o", mfc=purple, mec=purple, marker='o', alpha=1.0, ms=1)
        self.lightblue = dict(c=lightblue, ls="o", mfc=lightblue, mec=lightblue, marker='o', alpha=1.0, ms=1)
        self.cyan = dict(c='c', ls="o", mfc="c", mec="c", marker='o', alpha=1.0, ms=1)
        self.blue = dict(c=blue, ls="o", mfc=blue, mec=blue, marker='o', alpha=1.0, ms=1)
        self.magenta = dict(c='m', ls="o", mfc="m", mec="m", marker='o', alpha=1.0, ms=1)
        self.white = dict(c='w', ls="o", mfc="w", mec="w", marker='o', alpha=1.0, ms=1)
        self.black = dict(c='k', ls="o", mfc="k", mec="k", marker='o', alpha=1.0, ms=1)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--filename', required=True,
                        help="hdf5 file created by sim_veras_multiplanet.py")
    parser.add_argument('-s','--sim', required=True,
                        help="hdf5 path of simulation. e.g. /Hermite/sim_00")
    parser.add_argument('-o','--output', default=None,
                        help="output path/filename for mpeg")
    parser.add_argument('-a','--animate', action='store_true', 
                        help="Don't save")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_arguments()
    print(args)
    rundark()
    dots = Dots()
    line = Line()
    main()

 
