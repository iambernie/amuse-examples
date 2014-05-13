#!/usr/bin/env python

import os
import numpy

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

from amuse.units import units
from ext.misc import quantify_dset
from ext.misc import SimData

from itertools import cycle

def main():
    directory = 'figures/' 
    if not os.path.exists(directory):
        os.makedirs(directory)

    simple_binary_1 = SimData('data/single_binary_M1.0_m1.0_mdot5e-06_elems10.0_0.1_0.0_0.0_0.0_0.0_i1.0_t10000.0_p2000_Sm_ph_He_Hu.hdf5', figdir='simple_binary_1/')
    simple_binary_2 = SimData('data/single_binary_M1.0_m1.0_mdot5e-06_elems10.0_0.1_0.0_0.0_0.0_0.0_i1.0_t10000.0_p20000_Sm.hdf5', figdir='simple_binary_2/')
    binaries_1 = SimData('data/binaries_M1_m0.001_mdot5e-05_s10.0_i1_t15000.0_p15000_Sm_ph_He_Hu.hdf5', figdir='binaries_10/')
    binaries_2 = SimData('data/binaries_M1_m0.001_mdot5e-05_s50.0_i1_t15000.0_p15000_Sm_ph_He_Hu.hdf5', figdir='binaries_50/')
    binaries_3 = SimData('data/binaries_M1_m0.001_mdot5e-05_s100.0_i1_t15000.0_p5000_Sm_ph_He_Hu.hdf5', figdir='binaries_100/')
    binaries_4 = SimData('data/binaries_M1_m0.001_mdot5e-05_s150.0_i1_t15000.0_p5000_Sm_ph_He_Hu.hdf5', figdir='binaries_150/')
    binaries_5 = SimData('data/binaries_M1_m0.001_mdot5e-05_s200.0_i1_t15000.0_p5000_Sm_ph_He_Hu.hdf5', figdir='binaries_200/')
    binaries_6 = SimData('data/binaries_M1_m0.001_mdot5e-05_s1000.0_i1_t15000.0_p5000_Sm_ph_He_Hu.hdf5', figdir='binaries_1000/')
    

    def binaries(simdata, **kwargs):
        f = simdata.hdf5file
        figdir = simdata.figdir
        targetdir = directory+figdir

        if not os.path.exists(targetdir):
            os.makedirs(targetdir)

        if 'integrator' in kwargs:
           integrator = kwargs['integrator'] 
        else: 
            integrator = simdata.available_integrators()[0]

        centralmass = float(simdata.parameters['M'].split()[0])
        interval = float(simdata.parameters['interval'].split()[0])
        datapoints = int(simdata.parameters['datapoints'])
        mdot = float(simdata.parameters['mdot'].split()[0]) 
        sma0 = float(simdata.parameters['sma'].split()[0]) 

        print("sma0 {} AU".format(sma0))

        firstavailablesim = f[integrator].values()[0]

        time = quantify_dset(firstavailablesim['time']).value_in(units.yr)
        totalmass = quantify_dset(firstavailablesim['mass']).lengths().value_in(units.MSun)

        smas = []
        eccentricities = []
        ecc_labels = []
        smas_ad = []
        ml_indices = []
        arguments_of_periapsis = []
        angular_momenta = []
        true_anomalies = []
        periods = []
        
        for sim in f[integrator].values():
            sma = quantify_dset(sim['p0/sma']).value_in(units.AU)
            smas.append(sma)
            smas_ad.append(sma_analytical(sma0, mdot, time, centralmass))
            ml_indices.append(sim['p0/massloss_index'].value)
            eccentricities.append(sim['p0/eccentricity'].value)
            ecc_labels.append("e0= "+str(round(sim['p0/eccentricity'][0], 1)) )
            arguments_of_periapsis.append(sim['p0/argument_of_periapsis'].value)
            periods.append(quantify_dset(sim['p0/period']).value_in(units.yr))
            angular_momenta.append(quantify_dset(sim['angular_momentum']).number)
            true_anomalies.append(sim['p0/true_anomaly'].value)

        cmap = cm.jet
        savefigargs = dict(bbox_inches='tight', dpi=150)

        def sma_vs_time():
            imgname = 'sma_vs_time.png'
            fig = plt.figure(figsize=(32,8))
            ax = fig.add_subplot(111)
            ax.set_color_cycle(cmap(numpy.linspace(0, 0.95, 10)))

            for sma, lbl in zip(smas, ecc_labels):
                ax.plot(time, sma, lw=1, ls='-', label=lbl)

            ax.set_xlim(time[0], time[-1])
            ax.set_xlabel('time [yr]')
            ax.set_ylabel('sma [AU]')
            plt.savefig(targetdir+imgname, **savefigargs )
            plt.close()

        def sma_vs_adiabaticity():
            imgname = 'sma_vs_adiabaticity.png'
            fig = plt.figure(figsize=(32,8))
            ax = fig.add_subplot(111)
            ax.set_color_cycle(cmap(numpy.linspace(0, 0.95, 10)))

            for sma, ml_index, lbl in zip(smas, ml_indices, ecc_labels):
                ax.plot(ml_index, sma, lw=1, ls='-', label=lbl)

            ax.set_xlim(ml_index[0], ml_index[-1])
            ax.set_xlabel('massloss-index')
            ax.set_ylabel('sma [AU]')
            plt.savefig(targetdir+imgname, **savefigargs)
            plt.close()

        def sma_error_vs_time():
            imgname = 'sma_error_vs_time.png'
            fig = plt.figure(figsize=(32,8))
            ax = fig.add_subplot(111)
            ax.set_color_cycle(cmap(numpy.linspace(0, 0.95, 10)))

            for sma, sma_ad, lbl in zip(smas, smas_ad, ecc_labels):
                ax.plot(time, (sma-sma_ad)/sma_ad, lw=1, ls='-', label=lbl)
        
            ax.set_xlim(time[0], time[-1])
            ax.set_xlabel('time [yr]')
            ax.set_ylabel('(sma-sma_ad)/sma_ad ')
            plt.savefig(targetdir+imgname, **savefigargs)
            plt.close()

        def sma_error_vs_adiabaticity():
            imgname = 'sma_error_vs_adiabaticity.png'
            fig = plt.figure(figsize=(32,8))
            ax = fig.add_subplot(111)
            ax.set_color_cycle(cmap(numpy.linspace(0, 0.95, 10)))

            for ml_index, sma_ad, lbl in zip(ml_indices, smas_ad, ecc_labels):
                ax.plot(ml_index, (sma-sma_ad)/sma_ad, lw=1, ls='-', label=lbl)
        
            ax.set_xlim(ml_index[0], ml_index[-1])
            ax.set_xlabel('massloss-index')
            ax.set_ylabel('(sma-sma_ad)/sma_ad ')
            plt.savefig(targetdir+imgname, **savefigargs)
            plt.close()

        def eccentricity_vs_time():
            imgname = 'eccentricity_vs_time.png'
            fig = plt.figure(figsize=(32,16))
            ax = fig.add_subplot(111)
            ax.set_color_cycle(cmap(numpy.linspace(0, 0.95, 10)))

            for ecc, lbl in zip(eccentricities, ecc_labels):
                ax.plot(time, ecc, lw=2, ls='-', label=lbl)

            ax.set_xlim(time[0], time[-1])
            ax.set_xlabel('time [yr]')
            ax.set_ylabel('eccentricity')
            plt.savefig(targetdir+imgname, **savefigargs)
            plt.close()

        def eccentricity_vs_adiabaticity():
            imgname = 'eccentricity_vs_adiabaticity.png'
            fig = plt.figure(figsize=(32,16))
            ax = fig.add_subplot(111)
            ax.set_color_cycle(cmap(numpy.linspace(0, 0.95, 10)))

            for ecc, ml_index, lbl in zip(eccentricities, ml_indices, ecc_labels):
                ax.plot(ml_index, ecc, lw=2, ls='-', label=lbl)

            ax.set_xlim(ml_index[0], ml_index[-1])
            ax.set_xlabel('massloss-index')
            ax.set_ylabel('eccentricity')
            plt.savefig(targetdir+imgname, **savefigargs)
            plt.close()

        def sma_error_vs_eccentricity_error():
            imgname = 'sma_error_vs_eccentricity_error.png'
            fig = plt.figure(figsize=(32,8))
            ax = fig.add_subplot(111)
            ax.set_color_cycle(cmap(numpy.linspace(0, 0.95, 10)))

            for sma, sma_ad, ecc, lbl in zip(smas, smas_ad, eccentricities, ecc_labels):
                ax.plot((ecc-ecc[0])/ecc[0], (sma-sma_ad)/sma_ad, lw=1, ls='-', label=lbl)
        
            ax.set_xlabel('(ecc-ecc0)/ecc0')
            ax.set_ylabel('(sma-sma_ad)/sma_ad ')
            plt.savefig(targetdir+imgname, **savefigargs)
            plt.close()

        def true_anomaly_vs_time():
            imgname = 'true_anomaly_vs_time.png'
            fig = plt.figure(figsize=(30, 10))
            colorcycle = cycle(cmap(numpy.linspace(0, 0.95, 10)))
            position_generator = axis_position(10, 1) 

            totalsims = len(f[integrator].values())

            for i, true_anomaly, lbl in zip(range(totalsims), true_anomalies, ecc_labels):
                ax = fig.add_subplot(*next(position_generator))
                ax.plot(time, true_anomaly, lw=2, ls='-', c=next(colorcycle), label=lbl)

                if i+1 != totalsims:
                    ax.set_xticklabels([])
                else:
                    ax.set_xticks(numpy.arange(1000, 15000, 1000))

            ax.set_xlim(time[0], time[-1])
            ax.set_ylim(0, 360)
            ax.set_yticks([90, 180, 270])
            ax.set_xlabel('time [yr]')
            ax.set_ylabel('f [degrees]')
        
            plt.subplots_adjust(hspace=0.001)
            plt.savefig(targetdir+imgname, **savefigargs)
            plt.close()

        def argument_of_periapsis_vs_time():
            imgname = 'argument_of_periapsis_vs_time.png'
            fig = plt.figure(figsize=(20, 20))
            colorcycle = cycle(cmap(numpy.linspace(0, 0.95, 10)))
            position_generator = axis_position(3, 4) 

            for w, lbl in zip(arguments_of_periapsis, ecc_labels):
                ax = fig.add_subplot(*next(position_generator), polar=True)
                ax.plot(w, time, lw=1, ls='-',  c=next(colorcycle), label=lbl)
                ax.set_rgrids(numpy.array([5000,10000,15000]), angle=270)
        
            plt.savefig(targetdir+imgname, **savefigargs)
            plt.close()

        def adiabaticity_vs_time():
            imgname = 'adiabaticity_vs_time.png'
            fig = plt.figure(figsize=(8, 16))
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)
            colorcycle1 = cycle(cmap(numpy.linspace(0, 0.95, 10)))
            colorcycle2 = cycle(cmap(numpy.linspace(0, 0.95, 10)))
            colorcycle3 = cycle(cmap(numpy.linspace(0, 0.95, 10)))

            for ml_index, period, lbl in zip(ml_indices, periods, ecc_labels):
                ax1.plot(time, mdot/totalmass, lw=1, ls='-', c=next(colorcycle1), label=lbl)
                ax2.plot(time, period, lw=1, ls='-', c=next(colorcycle2), label=lbl)
                ax3.plot(time, ml_index, lw=1, ls='-', c=next(colorcycle3), label=lbl)

            ax3.set_yscale('log')
            ax1.set_xlabel('time [yr]')
            ax2.set_xlabel('time [yr]')
            ax3.set_xlabel('time [yr]')

            plt.subplots_adjust(hspace=0.001)
            plt.savefig(targetdir+imgname, **savefigargs)
            plt.close()

        def angular_momentum_error_vs_time():
            imgname = 'angular_momentum_error_vs_time.png'
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            colorcycle = cycle(cmap(numpy.linspace(0, 0.95, 10)))

            for h, lbl in zip(angular_momenta, ecc_labels):
                h_error = (h[0] - h)/h[0]
                ax.plot(time, h_error, lw=1, ls='-', c=next(colorcycle), label=lbl)

            ax.set_xlim(time[0], time[-1])
            plt.savefig(targetdir+imgname, **savefigargs)
            plt.close()

        def angular_momentum_error_vs_adiabaticity():
            imgname = 'angular_momentum_error_vs_adiabaticity.png'
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            colorcycle = cycle(cmap(numpy.linspace(0, 0.95, 10)))

            for h, ml_index, lbl in zip(angular_momenta, ml_indices, ecc_labels):
                h_error = (h[0] - h)/h[0]
                ax.plot(ml_index, h_error, lw=1, ls='-',  c=next(colorcycle), label=lbl)
                ax.set_xlim(ml_index[0], ml_index[-1])

            plt.savefig(targetdir+imgname, **savefigargs)
            plt.close()

        sma_vs_time()
        sma_vs_adiabaticity()
        sma_error_vs_time()
        sma_error_vs_adiabaticity()
        eccentricity_vs_time()
        eccentricity_vs_adiabaticity()
        sma_error_vs_eccentricity_error()
        true_anomaly_vs_time()
        argument_of_periapsis_vs_time()
        adiabaticity_vs_time()
        angular_momentum_error_vs_time()
        angular_momentum_error_vs_adiabaticity()

    def simple_binary(simdata, *args, **kwargs):
        f = simdata.hdf5file
        figdir = simdata.figdir
        targetdir = directory+figdir

        if not os.path.exists(targetdir):
            os.makedirs(targetdir)

        firstsim = f.values()[0]
        time = quantify_dset(firstsim['time']).value_in(units.yr)
        totalmass = quantify_dset(firstsim['mass']).lengths().value_in(units.MSun)

        smas = []
        eccentricities = []
        intr_labels = []
        smas_ad = []
        ml_indices = []
        arguments_of_periapsis = []
        angular_momenta = []
        O_angular_momenta = []
        true_anomalies = []
        periods = []

        for intr in f.values():
            sma = quantify_dset(intr['p0/sma']).value_in(units.AU)
            smas.append(sma)
            #smas_ad.append(sma_analytical(sma0, mdot, time, centralmass))
            ml_indices.append(intr['p0/massloss_index'].value)
            eccentricities.append(intr['p0/eccentricity'].value)
            intr_labels.append(str(intr))
            arguments_of_periapsis.append(intr['p0/argument_of_periapsis'].value)
            periods.append(quantify_dset(intr['p0/period']).value_in(units.yr))
            angular_momenta.append(quantify_dset(intr['angular_momentum']).lengths().number)
            O_angular_momenta.append(quantify_dset(intr['O_angular_momentum']).lengths().number)
            true_anomalies.append(intr['p0/true_anomaly'].value)

        cmap = cm.jet
        savefigargs = dict(bbox_inches='tight', dpi=150)
        
        def angular_momentum_vs_time():
            imgname = 'angular_momentum_vs_time.png'
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            colorcycle = cycle(['k','r','g','y', 'c', 'm', 'b'])

            for h, lbl in zip(angular_momenta, intr_labels):
                ax.plot(time, h, lw=1, ls='-', c=next(colorcycle), label=lbl)
            
            ax.legend(loc='best')
            ax.set_xlim(time[0], time[-1])

            plt.savefig(targetdir+imgname, **savefigargs)
            plt.close()

        def O_angular_momentum_vs_time():
            imgname = 'O_angular_momentum_vs_time.png'
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            colorcycle = cycle(['k','r','g','y', 'c', 'm', 'b'])

            for h, lbl in zip(O_angular_momenta, intr_labels):
                ax.plot(time, h, lw=1, ls='-',  c=next(colorcycle), label=lbl)
            
            ax.legend(loc='best')
            ax.set_xlim(time[0], time[-1])

            plt.savefig(targetdir+imgname, **savefigargs)
            plt.close()

        def eccentricity_vs_time():
            imgname = 'eccentricity_vs_time.png'
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            colorcycle = cycle(['k','r','g','y', 'c', 'm', 'b'])

            for e, lbl in zip(eccentricities, intr_labels):
                ax.plot(time, e, lw=1, ls='-', c=next(colorcycle), label=lbl)
            
            ax.legend(loc='best')
            ax.set_xlim(time[0], time[-1])

            plt.savefig(targetdir+imgname, **savefigargs)
            plt.close()

        angular_momentum_vs_time()
        O_angular_momentum_vs_time()
        eccentricity_vs_time()

        

    #binaries(binaries_1)
    #binaries(binaries_2)
    #binaries(binaries_3)
    #binaries(binaries_4)
    #binaries(binaries_5)
    #binaries(binaries_6)
    simple_binary(simple_binary_1)
    simple_binary(simple_binary_2)

def axis_position(rows, columns):
    total = rows * columns
    i=1
    while i <= total:
        yield (rows, columns, i)
        i += 1

def sma_analytical(a0, mdot, t, mu0):
    return a0*(1 - mdot*t/mu0)**(-1)
    

if __name__ == '__main__':
    mpl.rc('font',  size=9, family='sans-serif')
    main()
