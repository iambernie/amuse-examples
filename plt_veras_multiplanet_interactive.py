#!/usr/bin/env python

import argparse
import h5py
import numpy

import matplotlib.pyplot as plt

from amuse.units import units
from amuse.units.quantities import AdaptingVectorQuantity

from ext.misc import printdset
from ext.misc import retrieve_unit
from ext.misc import quantify_dset
from ext.colors import rundark, runbright

def main():

    dots = Dots()
    line = Line()

    f = h5py.File(args.filename, 'r')
    f.visititems(printdset)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    for intr in f.values():

        timesteps_vq = AdaptingVectorQuantity()
        final_smas_p0_vq = AdaptingVectorQuantity()
        final_smas_p1_vq = AdaptingVectorQuantity()

        for sim in intr.values():
            timesteps_vq.append(sim['timestep'][0] | retrieve_unit(sim['timestep']))
            final_smas_p0_vq.append(sim['p0/sma'][-1]| retrieve_unit(sim['p0/sma']))
            final_smas_p1_vq.append(sim['p1/sma'][-1]| retrieve_unit(sim['p1/sma']))

        timesteps = timesteps_vq.value_in(units.yr)
        final_smas_p0 = final_smas_p0_vq.value_in(units.AU)
        final_smas_p1 = final_smas_p1_vq.value_in(units.AU)

        ax1.plot(timesteps, final_smas_p0, marker='o', label=intr.name, picker=5)
        ax2.plot(timesteps, final_smas_p1, marker='o', label=intr.name, picker=5)

        ax1.set_xlabel('Mass update interval [yr]')
        ax2.set_xlabel('Mass update interval [yr]')
        ax1.set_ylabel('final sma inner [AU]')
        ax2.set_ylabel('final sma outer [AU]')
        
    ax1.axhline(53.19, xmin=0, xmax=1, c='m', label='analytical')
    ax1.legend(loc='best')
    ax2.axhline(159.58, xmin=0, xmax=1, c='m', label='analytical')
    ax2.legend(loc='best')

    def sma_analytical(a0, mdot, t, mu0):
        return a0*(1 - mdot*t/mu0)**(-1)


    def onpick(event):
        print("artist:{} ind:{}".format(event.artist, event.ind))
        print("button: {}".format(event.mouseevent.button))

        intr = f[event.artist.get_label()]
        sim = intr.values()[event.ind]

        time_vq = quantify_dset(sim['time'])
        time = time_vq.value_in(units.yr) 

        position_vq = quantify_dset(sim['position'])
        position = position_vq.value_in(units.AU)

        CM_position_vq = quantify_dset(sim['CM_position'])
        CM_position = CM_position_vq.value_in(units.AU)

        x, y = 0, 1
        central_x, central_y =  position[:, 0, x] - CM_position[:,x], position[:, 0, y] - CM_position[:,y]
        inner_x, inner_y =  position[:, 1, x] - CM_position[:,x], position[:, 1, y] - CM_position[:,y]
        outer_x, outer_y =  position[:, 2, x] - CM_position[:,x], position[:, 2, y] - CM_position[:,y]

        mass_vq = quantify_dset(sim['mass'])
        mass = mass_vq[:,0].value_in(units.MSun)

        if event.mouseevent.button == 1:

            mu0 = quantify_dset(sim['mass'])[0].sum()

            period_vq = quantify_dset(sim['p0/period'])
            period = period_vq.value_in(units.yr)

            true_anomaly = sim['p0/true_anomaly'].value

            sma_vq = quantify_dset(sim['p0/sma'])
            sma = sma_vq.value_in(units.AU)
            sma_an_vq = sma_analytical(sma_vq[0], 1.244e-5|(units.MSun/units.yr), time_vq, mu0)
            sma_an = sma_an_vq.value_in(units.AU)

            eccentricity = sim['p0/eccentricity'].value

            newfig = plt.figure()
            newax1 = newfig.add_subplot(511)
            newax2 = newfig.add_subplot(512)
            newax3 = newfig.add_subplot(513)
            newax4 = newfig.add_subplot(514)
            newax5 = newfig.add_subplot(515)

            newax1.plot(time, sma, label='numerical')
            newax1.plot(time, sma_an, label='analytical_adiabatic')
            
            newax1.set_xlabel('time [yr]')
            newax1.set_ylabel('sma [AU]')
            newax1.legend(loc='best')

            newax2.plot(time, eccentricity)
            newax2.set_xlabel('time [yr]')
            newax2.set_ylabel('eccentricity ')

            newax3.plot(time, true_anomaly)
            newax3.set_xlabel('time [yr]')
            newax3.set_ylabel('true anomaly [degrees] ')

            newax5.plot(time, period)
            newax5.set_xlabel('time [yr]')
            newax5.set_ylabel('period')

        else:
            newfig = plt.figure()
            newax1 = newfig.add_subplot(321)
            newax2 = newfig.add_subplot(322)
            newax3 = newfig.add_subplot(323)
            newax4 = newfig.add_subplot(324)
            newax5 = newfig.add_subplot(325)
            newax6 = newfig.add_subplot(326)

            mass_vq = quantify_dset(sim['mass'])
            mass = mass_vq.value_in(units.MSun)

            kinetic_energy = sim['kinetic_energy'].value
            potential_energy = sim['potential_energy'].value
            total_energy = sim['total_energy'].value

            CM_velocity_vq = quantify_dset(sim['CM_velocity'])
            CM_velocity_mod = CM_velocity_vq.lengths().value_in(units.km/units.hour)

            walltime = sim['walltime'].value - sim['walltime'][0]

            newax1.plot(time, mass)
            newax1.set_ylabel('central mass [MSun]')
            newax1.set_xlabel('time [yr]')

            newax2.plot(time, kinetic_energy, label='kinetic', **line.red)
            newax2.plot(time, potential_energy,label='potential',  **line.orange)
            newax2.plot(time, total_energy, label='total', **line.white)
            newax2.set_xlabel('time [yr]')
            newax2.legend()

            newax3.plot(central_x, central_y, **dots.white)
            newax3.plot(inner_x, inner_y, **dots.red)
            newax3.plot(outer_x, outer_y, **dots.yellow)
            newax3.set_xlabel('x [AU]')
            newax3.set_ylabel('y [AU]')

            newax4.plot(CM_position[:, x], CM_position[:, y] )
            newax4.set_xlim(-5, 5)
            newax4.set_xlabel('CM position x [AU]')
            newax4.set_ylabel('CM position y [AU]')

            newax5.plot(time, CM_velocity_mod)
            newax5.set_ylim(20, 30)
            newax5.set_xlabel('time [yr]')
            newax5.set_ylabel('CM velocity [km/hour]')

            newax6.plot(time, walltime)
            newax6.set_xlabel('time [yr]')
            newax6.set_ylabel('walltime [s]')

        newfig.show()

    fig.canvas.mpl_connect('pick_event', onpick) 
    #fig.canvas.mpl_connect('button_press_event', onclick) 
    plt.show()

         
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
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_arguments()
    print(args)
    rundark()
    main()

 
