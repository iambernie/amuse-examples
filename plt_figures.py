#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy
import h5py

from amuse.units import units
from ext.misc import quantify_dset
from ext.misc import Parameters

from itertools import cycle

def main():
    directory = 'figures/' 
    binaries_highly_adiabatic_filename = 'data/binaries_seperation10_points1000_interval1yr_SmallN.hdf5'
    binaries_highly_adiabatic_filename = 'data/binaries_seperation200_points1000_interval1yr_SmallN.hdf5'
    binaries_highly_adiabatic_filename = 'data/binaries_seperation100_points1000_interval1yr_SmallN.hdf5'
    binaries_highly_adiabatic_filename = 'testargper2.hdf5'
    binaries_highly_adiabatic = Parameters(binaries_highly_adiabatic_filename)
    
    #TODO: plot suites    
    #TODO: labels    

    def plot_figure_01(integrator=None, imgname = '01.png'):
        f = binaries_highly_adiabatic.hdf5file
        if integrator is None:
            integrator = binaries_highly_adiabatic.available_integrators()[0]

        cmap = cm.jet
    
        fig = plt.figure(figsize=(32,8))
        ax = fig.add_subplot(111)
        ax.set_color_cycle(cmap(numpy.linspace(0, 0.95, 10)))

        for sim in f[integrator].values():
            time = quantify_dset(sim['time']).value_in(units.yr)
            sma = quantify_dset(sim['p0/sma']).value_in(units.AU)
            eccentricity = sim['p0/eccentricity'].value
            ax.plot(time, sma, lw=1, ls='-', label="e0= "+str(round(eccentricity[0], 1)))
            ax.set_xlim(time[0], time[-1])
    
        plt.savefig(directory+imgname, bbox_inches='tight', dpi=150)
        plt.close()

    def plot_figure_02(integrator=None, imgname='02.png'):
        f = binaries_highly_adiabatic.hdf5file
        if integrator is None:
            integrator = binaries_highly_adiabatic.available_integrators()[0]

        cmap = cm.jet

        fig = plt.figure(figsize=(32,8))
        ax = fig.add_subplot(111)
        ax.set_color_cycle(cmap(numpy.linspace(0, 0.95, 10)))


        for sim in f[integrator].values():
            time = quantify_dset(sim['time']).value_in(units.yr)
            sma = quantify_dset(sim['p0/sma']).value_in(units.AU)
            #TODO, store mdot and mu0 from in the hdf5 file so it doesn't
            #need to be hardcoded here in the sma_adiabatic call
            sma_adiabatic = sma_analytical(sma[0], 5e-5, time, 1)
            eccentricity = sim['p0/eccentricity'].value
            ax.set_xlim(time[0], time[-1])

            ax.plot(time, (sma-sma_adiabatic)/sma_adiabatic, lw=1, ls='-', alpha=0.5, label="e0= "+str(round(eccentricity[0], 1)))
    
        plt.savefig(directory+imgname, bbox_inches='tight', dpi=150)
        plt.close()

    def plot_sma_vs_adiabaticity(integrator=None, imgname='02_ad.png'):
        f = binaries_highly_adiabatic.hdf5file
        if integrator is None:
            integrator = binaries_highly_adiabatic.available_integrators()[0]

        cmap = cm.jet

        fig = plt.figure(figsize=(32,8))
        ax = fig.add_subplot(111)
        ax.set_color_cycle(cmap(numpy.linspace(0, 0.95, 10)))


        for sim in f[integrator].values():
            time = quantify_dset(sim['time']).value_in(units.yr)
            ml_index = sim['p0/massloss_index'].value
            sma = quantify_dset(sim['p0/sma']).value_in(units.AU)
            sma_adiabatic = sma_analytical(sma[0], 5e-5, time, 1)
            eccentricity = sim['p0/eccentricity'].value
            ax.set_xlim(ml_index[0], ml_index[-1])

            ax.plot(ml_index, (sma-sma_adiabatic)/sma_adiabatic, lw=1, ls='-', alpha=0.5, label="e0= "+str(round(eccentricity[0], 1)))
    
        plt.savefig(directory+imgname, bbox_inches='tight', dpi=150)
        plt.close()

    def plot_figure_03(integrator=None, imgname='03.png'):
        f = binaries_highly_adiabatic.hdf5file
        if integrator is None:
            integrator = binaries_highly_adiabatic.available_integrators()[0]

        cmap = cm.jet

        fig = plt.figure(figsize=(32,16))
        ax1 = fig.add_subplot(111)
        ax1.set_color_cycle(cmap(numpy.linspace(0, 0.95, 10)))

        for sim in f[integrator].values():
            time = quantify_dset(sim['time']).value_in(units.yr)
            eccentricity = sim['p0/eccentricity'].value
            label = "e0= "+str(round(eccentricity[0], 1))
            ax1.plot(time, eccentricity, lw=2, ls='-', label=label)
            ax1.set_xlim(time[0], time[-1])
    
        plt.savefig(directory+imgname, bbox_inches='tight', dpi=150)
        plt.close()

    def plot_eccentricity_vs_adiabaticity(integrator=None, imgname='03_ad.png'):
        f = binaries_highly_adiabatic.hdf5file
        if integrator is None:
            integrator = binaries_highly_adiabatic.available_integrators()[0]

        cmap = cm.jet

        fig = plt.figure(figsize=(32,16))
        ax = fig.add_subplot(111)
        ax.set_color_cycle(cmap(numpy.linspace(0, 0.95, 10)))

        for sim in f[integrator].values():
            ml_index = sim['p0/massloss_index'].value
            eccentricity = sim['p0/eccentricity'].value
            label = "e0= "+str(round(eccentricity[0], 1))
            ax.plot(ml_index, eccentricity, lw=2, ls='-', label=label)
            ax.set_xlim(ml_index[0], ml_index[-1])
    
        plt.savefig(directory+imgname, bbox_inches='tight', dpi=150)
        plt.close()

    def plot_figure_04(integrator=None, imgname='04.png'):
        f = binaries_highly_adiabatic.hdf5file
        if integrator is None:
            integrator = binaries_highly_adiabatic.available_integrators()[0]

        cmap = cm.jet

        fig = plt.figure(figsize=(30, 10))
        colorcycle = cycle(cmap(numpy.linspace(0, 0.95, 10)))
        position_generator = axis_position(10, 1) 

        totalsims = len(f[integrator].values())

        for i, sim in enumerate(f[integrator].values()):
            ax = fig.add_subplot(*next(position_generator))
            time = quantify_dset(sim['time']).value_in(units.yr)
            eccentricity = sim['p0/eccentricity'].value
            true_anomaly = sim['p0/true_anomaly'].value
            label = "e0= "+str(round(eccentricity[0], 1))
            ax.plot(time, true_anomaly, lw=2, ls='-', c=next(colorcycle), label=label)
            ax.set_xlim(time[0], time[-1])
            ax.set_ylim(0, 360)
            ax.set_yticks([90, 180, 270])

            if i+1 != totalsims:
                ax.set_xticklabels([])
            else:
                ax.set_xticks(numpy.arange(1000, 15000, 1000))
    
        plt.subplots_adjust(hspace=0.001)
        plt.savefig(directory+imgname, bbox_inches='tight', dpi=150)
        plt.close()

    def plot_figure_05(integrator=None, imgname='05.png'):
        f = binaries_highly_adiabatic.hdf5file
        if integrator is None:
            integrator = binaries_highly_adiabatic.available_integrators()[0]
        cmap = cm.jet
        colorcycle = cycle(cmap(numpy.linspace(0, 0.95, 10)))
        position_generator = axis_position(3, 4) 

        fig = plt.figure(figsize=(20, 20))

        for sim in f[integrator].values():
            ax = fig.add_subplot(*next(position_generator), polar=True)
            time = quantify_dset(sim['time']).value_in(units.yr)
            eccentricity = sim['p0/eccentricity'].value
            argument_of_periapsis = sim['p0/argument_of_periapsis'].value
            label = "e0= "+str(round(eccentricity[0], 1))
            ax.plot(argument_of_periapsis, time, lw=1, ls='-',  c=next(colorcycle), label=label)
            ax.set_rgrids(numpy.array([5000,10000,15000]), angle=270)
    
        plt.savefig(directory+imgname, bbox_inches='tight', dpi=150)
        plt.close()

    def plot_figure_06(integrator=None, imgname='06.png'):
        f = binaries_highly_adiabatic.hdf5file
        if integrator is None:
            integrator = binaries_highly_adiabatic.available_integrators()[0]
        cmap = cm.jet
        colorcycle1 = cycle(cmap(numpy.linspace(0, 0.95, 10)))
        colorcycle2 = cycle(cmap(numpy.linspace(0, 0.95, 10)))
        colorcycle3 = cycle(cmap(numpy.linspace(0, 0.95, 10)))

        fig = plt.figure(figsize=(8, 16))
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)

        for sim in f[integrator].values():
            time = quantify_dset(sim['time']).value_in(units.yr)
            eccentricity = sim['p0/eccentricity'].value
            period = quantify_dset(sim['p0/period']).value_in(units.yr)
            ml_index = sim['p0/massloss_index'].value
            label = "e0= "+str(round(eccentricity[0], 1))
            ax1.plot(time, period, lw=1, ls='-',  c=next(colorcycle1), label=label)
            ax2.plot(time, period, lw=1, ls='-',  c=next(colorcycle2), label=label)
            ax3.plot(time, ml_index, lw=1, ls='-',  c=next(colorcycle3), label=label)

        ax1.set_yscale('log')
        ax2.set_yscale('log')
        ax3.set_yscale('log')

        plt.subplots_adjust(hspace=0.001)
        plt.savefig(directory+imgname, bbox_inches='tight', dpi=150)
        plt.close()

    def plot_figure_07(integrator=None, imgname='07.jpg'):
        f = binaries_highly_adiabatic.hdf5file
        if integrator is None:
            integrator = binaries_highly_adiabatic.available_integrators()[0]
        cmap = cm.jet

        colorcycle = cycle(cmap(numpy.linspace(0, 0.95, 10)))

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        for sim in f[integrator].values():
            time = quantify_dset(sim['time']).value_in(units.yr)
            eccentricity = sim['p0/eccentricity'].value
            angular_momentum = quantify_dset(sim['angular_momentum']).number
            label = "e0= "+str(round(eccentricity[0], 1))
            ang0 = angular_momentum[0]
            angular_momentum_error = (ang0 - angular_momentum)/ang0
            ax.plot(time, angular_momentum_error, lw=1, ls='-',  c=next(colorcycle), label=label)
            ax.set_xlim(time[0], time[-1])

        plt.savefig(directory+imgname, bbox_inches='tight', dpi=150)
        plt.close()

    def plot_angular_momentum_vs_adabaticity(integrator=None, imgname='07_ad.png'):
        f = binaries_highly_adiabatic.hdf5file
        if integrator is None:
            integrator = binaries_highly_adiabatic.available_integrators()[0]
        cmap = cm.jet

        colorcycle = cycle(cmap(numpy.linspace(0, 0.95, 10)))

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        for sim in f[integrator].values():
            eccentricity = sim['p0/eccentricity'].value
            ml_index = sim['p0/massloss_index'].value
            angular_momentum = quantify_dset(sim['angular_momentum']).number
            label = "e0= "+str(round(eccentricity[0], 1))
            ang0 = angular_momentum[0]
            angular_momentum_error = (ang0 - angular_momentum)/ang0
            ax.plot(ml_index, angular_momentum_error, lw=1, ls='-',  c=next(colorcycle), label=label)
            ax.set_xlim(ml_index[0], ml_index[-1])

        plt.savefig(directory+imgname, bbox_inches='tight', dpi=150)
        plt.close()

    plot_figure_01()
    plot_figure_02()
    plot_sma_vs_adiabaticity()
    plot_figure_03()
    plot_eccentricity_vs_adiabaticity()
    plot_figure_04()
    plot_figure_05()
    plot_figure_06()
    plot_figure_07()
    plot_angular_momentum_vs_adabaticity()

def axis_position(rows, columns):
    total = rows * columns
    i=1
    while i <= total:
        yield (rows, columns, i)
        i += 1

def sma_analytical(a0, mdot, t, mu0):
    return a0*(1 - mdot*t/mu0)**(-1)
    

if __name__ == '__main__':
    main()
