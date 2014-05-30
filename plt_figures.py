#!/usr/bin/env python

import os
import numpy

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

from matplotlib.collections import LineCollection
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm

from amuse.units import units
from ext.misc import quantify_dset
from ext.misc import SimData

from itertools import cycle

def main():
    directory = 'figures/' 
    if not os.path.exists(directory):
        os.makedirs(directory)


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
        ecc_ads = []
        ml_indices = []
        arguments_of_periapsis = []
        angular_momenta = []
        true_anomalies = []
        periods = []
        
        for sim in f[integrator].values():
            e0 = round(sim['p0/eccentricity'][0], 1)
            ml = sim['p0/massloss_index'].value
            sma = quantify_dset(sim['p0/sma']).value_in(units.AU)
            smas.append(sma)
            smas_ad.append(sma_adiabatic(sma0, mdot, time, centralmass))
            ml_indices.append(sim['p0/massloss_index'].value)
            eccentricities.append(sim['p0/eccentricity'].value)
            ecc_labels.append("e0= "+str(e0) )
            arguments_of_periapsis.append(sim['p0/argument_of_periapsis'].value)
            periods.append(quantify_dset(sim['p0/period']).value_in(units.yr))
            angular_momenta.append(quantify_dset(sim['angular_momentum']).number)
	    true_anomaly = sim['p0/true_anomaly'].value
            true_anomalies.append(true_anomaly)
            ecc_ads.append(ecc_adiabatic(e0, ml[0], true_anomaly))

        cmap = cm.jet
        savefigargs = dict(bbox_inches='tight', dpi=150)


        def sma_vs_time():
            imgname = 'sma_vs_time.png'
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            ax.set_color_cycle(cmap(numpy.linspace(0, 0.95, 10)))

            for sma, lbl in zip(smas, ecc_labels):
                ax.plot(time, sma, lw=1, ls='-', label=lbl)

            ax.set_xlim(time[0], time[-1])
            ax.set_xlabel('time [yr]')
            ax.set_ylabel('sma [AU]')
            plt.savefig(targetdir+imgname, **savefigargs )
            plt.close()

        def sma_error_vs_time():
            imgname = 'sma_error_vs_time.png'
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            ax.set_color_cycle(cmap(numpy.linspace(0, 0.95, 10)))

            for sma, sma_ad, lbl in zip(smas, smas_ad, ecc_labels):
                ax.plot(time, (sma-sma_ad)/sma_ad, lw=1, ls='-', label=lbl)
        
            ax.set_xlim(time[0], time[-1])
            ax.set_xlabel('time [yr]')
            ax.set_ylabel('(sma-sma_ad)/sma_ad ')
            plt.savefig(targetdir+imgname, **savefigargs)
            plt.close()

        def eccentricity_vs_time():
            imgname = 'eccentricity_vs_time.png'
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            ax.set_color_cycle(cmap(numpy.linspace(0, 0.95, 10)))

            for ecc, lbl in zip(eccentricities, ecc_labels):
                ax.plot(time, ecc, lw=2, ls='-', label=lbl)

            ax.set_xlim(time[0], time[-1])
            ax.set_xlabel('time [yr]')
            ax.set_ylabel('eccentricity')
            plt.savefig(targetdir+imgname, **savefigargs)
            plt.close()


        def true_anomaly_vs_time():
            imgname = 'true_anomaly_vs_time.png'
            fig = plt.figure(figsize=(16, 8))
            colorcycle = cycle(cmap(numpy.linspace(0, 0.95, 10)))
            position_generator = axis_position(10, 1) 

            totalsims = len(f[integrator].values())

            for i, true_anomaly, lbl in zip(range(totalsims), true_anomalies, ecc_labels):
                ax = fig.add_subplot(*next(position_generator))
                ax.plot(time, true_anomaly, lw=2, ls='-', c=next(colorcycle), label=lbl)

                ax.set_ylabel('f [degrees]')
                ax.set_xlim(time[0], time[-1])
                ax.set_ylim(0, 360)
                ax.set_yticks([90, 180, 270])

                if i+1 != totalsims:
                    ax.set_xticklabels([])
                else:
                    pass
                    #ax.set_xticks(numpy.arange(1000, 15000, 1000))

            ax.set_xlabel('time [yr]')
        
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
                #ax.set_rgrids(numpy.array([5000,10000,15000]), angle=270)
        
            plt.savefig(targetdir+imgname, **savefigargs)
            plt.close()

        def adiabaticity_vs_time():
            imgname = 'adiabaticity_vs_time.png'
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            colorcycle1 = cycle(cmap(numpy.linspace(0, 0.95, 10)))

            for ml_index, period, lbl in zip(ml_indices, periods, ecc_labels):
                ax.plot(time, ml_index, lw=1, ls='-', c=next(colorcycle1), label=lbl)

            ax.set_xlim(time[0], time[-1])
            ax.set_xlabel('time [yr]')
            ax.set_ylabel('mass-loss index')

            plt.savefig(targetdir+imgname, **savefigargs)
            plt.close()

        def angular_momentum_vs_time():
            imgname = 'angular_momentum_vs_time.png'
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            colorcycle = cycle(cmap(numpy.linspace(0, 0.95, 10)))

            for h, lbl in zip(angular_momenta, ecc_labels):
                ax.plot(time, h, lw=1, ls='-', c=next(colorcycle), label=lbl)

            ax.set_xlim(time[0], time[-1])
            ax.set_xlabel('time [yr]')
            ax.set_ylabel('angular momentum [m**2 * kg * s**-1]')
            plt.savefig(targetdir+imgname, **savefigargs)
            plt.close()

        sma_vs_time()
        sma_error_vs_time()
        eccentricity_vs_time()
        true_anomaly_vs_time()
        argument_of_periapsis_vs_time()
        adiabaticity_vs_time()
        angular_momentum_vs_time()


    def simple_binary(simdata, *args, **kwargs):
        f = simdata.hdf5file
        figdir = simdata.figdir
        targetdir = directory+figdir
        print(figdir)

        if not os.path.exists(targetdir):
            os.makedirs(targetdir)

        if 'integrator' in kwargs:
           integrator = kwargs['integrator'] 
        else: 
            integrator = simdata.available_integrators()[0]

        sim = f[integrator]
        time = quantify_dset(sim['time']).value_in(units.yr)
        totalmass = quantify_dset(sim['mass']).lengths().value_in(units.MSun)

        kinetic_energy = sim['kinetic_energy'].value
        potential_energy = sim['potential_energy'].value
        total_energy = sim['total_energy'].value
        true_anomaly = sim['p0/true_anomaly'].value
        argument_of_periapsis = sim['p0/argument_of_periapsis'].value
        longitude_of_ascending_node = sim['p0/longitude_of_ascending_node'].value
        sma = quantify_dset(sim['p0/sma']).value_in(units.AU)
        position = quantify_dset(sim['position']).value_in(units.AU)
        CM_position = quantify_dset(sim['CM_position']).value_in(units.AU)
        period = quantify_dset(sim['p0/period']).value_in(units.yr)
        angular_momentum = quantify_dset(sim['angular_momentum']).lengths().number
        O_angular_momentum = quantify_dset(sim['O_angular_momentum']).lengths().number
        eccentricity = sim['p0/eccentricity'].value

        massupdatetime = quantify_dset(sim['massupdate/massupdatetime']).value_in(units.yr)
        true_anomaly_at_update = sim['massupdate/true_anomaly'].value
        position_at_update = quantify_dset(sim['massupdate/position']).value_in(units.AU)

        try:
            CM_position_at_update = quantify_dset(sim['massupdate/CM_position']).value_in(units.AU)
        except:
            pass

        savefigargs = dict(bbox_inches='tight', dpi=100)
        
        def angular_momentum_vs_time():
            imgname = 'angular_momentum_vs_time.png'
            fig = plt.figure(figsize=(6, 2))
            ax = fig.add_subplot(111)
            ax.plot(time, angular_momentum, lw=2, ls='-', c='k')
            ax.set_xlim(time[0], time[-1])
            ax.set_xlabel('time [yr]')
            ax.set_ylabel('angular momentum [m**2 * kg * s**-1]')
            plt.savefig(targetdir+imgname, **savefigargs)
            plt.close()

        def O_angular_momentum_vs_time():
            imgname = 'O_angular_momentum_vs_time.png'
            fig = plt.figure(figsize=(6, 2))
            ax = fig.add_subplot(111)
            ax.plot(time, O_angular_momentum, lw=2, ls='-',  c='k')
            ax.set_xlim(time[0], time[-1])
            ax.set_xlabel('time [yr]')
            ax.set_ylabel('angular momentum [m**2 * kg * s**-1]')
            plt.savefig(targetdir+imgname, **savefigargs)
            plt.close()

        def eccentricity_vs_time():
            imgname = 'eccentricity_vs_time.png'
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111)
            ax.plot(time, eccentricity, lw=1, ls='-', c='k')
            ax.set_xlabel('time [yr]')
            ax.set_ylabel('eccentricity')
            ax.set_xlim(time[0], time[-1])

            plt.savefig(targetdir+imgname, **savefigargs)
            plt.close()

        def sma_vs_time():
            imgname = 'sma_vs_time.png'
            fig = plt.figure(figsize=(6, 2))
            ax = fig.add_subplot(111)
            ax.plot(time, sma, lw=1, ls='-', c='k')
            ax.set_xlabel('time [yr]')
            ax.set_ylabel('sma [AU]')
            ax.set_xlim(time[0], time[-1])

            plt.savefig(targetdir+imgname, **savefigargs)
            plt.close()

        def period_vs_time():
            imgname = 'period_vs_time.png'
            fig = plt.figure(figsize=(6, 2))
            ax = fig.add_subplot(111)
            ax.plot(time, period, lw=2, ls='-', c='k')
            ax.set_xlim(time[0], time[-1])
            ax.set_xlabel('time [yr]')
            ax.set_ylabel('period [yr]')

            plt.savefig(targetdir+imgname, **savefigargs)
            plt.close()

        def true_anomaly_vs_time():
            imgname = 'true_anomaly_massupdates_vs_time.png'
            fig = plt.figure(figsize=(20, 20))
            ax = fig.add_subplot(111, polar=True)
            ax.scatter(true_anomaly, time, s=1)
            #ax.scatter(massupdatetime, time, s=3, color='r')
            ax.scatter(true_anomaly_at_update, massupdatetime, s=20, color='r')
            ax.set_rmax(time[-1])
            #ax.set_xlim(time[0], time[-1])
            #ax.set_xlabel('time [yr]')
            #ax.set_ylabel('period [yr]')

            plt.savefig(targetdir+imgname, **savefigargs)
            plt.close()

        def argument_of_periapsis_vs_time():
            imgname = 'argument_of_periapsis_vs_time.png'
            fig = plt.figure(figsize=(20, 20))
            ax = fig.add_subplot(111, polar=True)
            ax.scatter(argument_of_periapsis, time, s=1)
            ax.scatter(true_anomaly_at_update, massupdatetime, s=6, color='r')
            ax.set_rmax(time[-1])

            plt.savefig(targetdir+imgname, **savefigargs)
            plt.close()

        def longitude_of_ascending_node_vs_time():
            imgname = 'longitude_of_ascending_node_vs_time.png'
            fig = plt.figure(figsize=(20, 20))
            ax = fig.add_subplot(111, polar=True)
            ax.scatter(longitude_of_ascending_node, time, s=1)
            ax.set_rmax(time[-1])

            plt.savefig(targetdir+imgname, **savefigargs)
            plt.close()

        def positions_in_time(frame='lab'):
            imgname = 'positions_in_time'+'_'+frame+'.png'
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)

            if frame == 'cm':
                restframe = CM_position
            elif frame == 'M':
                restframe = position[:, 0, :]
            elif frame == 'm':
                restframe = position[:, 1, :]
            else: #lab frame
                restframe = numpy.zeros(CM_position.size).reshape(CM_position.shape)

            x,y = 0,1
            central_x = position[:, 0, x] - restframe[:, x]
            central_y = position[:, 0, y] - restframe[:, y]
            orbiting_x = position[:, 1, x] - restframe[:, x]
            orbiting_y = position[:, 1, y] - restframe[:, y]
            CM_x = CM_position[:, x] - restframe[:, x]
            CM_y = CM_position[:, y] - restframe[:, y]

            try: 
                if frame == 'cm':
                    restframe_at_update = CM_position_at_update
                elif frame == 'M':
                    restframe_at_update = position_at_update[:, 0, :]
                elif frame == 'm':
                    restframe_at_update = position_at_update[:, 1, :]
                else: #lab frame
                    restframe_at_update = numpy.zeros(CM_position.size).reshape(CM_position.shape)

                orbiting_x_at_update = position_at_update[:, 1, x] - restframe_at_update[:, x]
                orbiting_y_at_update = position_at_update[:, 1, y] - restframe_at_update[:, y]

            except:
                pass

            cmap = mpl.cm.hsv
            points = numpy.array([orbiting_x, orbiting_y]).T.reshape(-1, 1, 2)
            segments = numpy.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap, norm=mpl.colors.Normalize(vmin=time.min(),vmax=time.max()))
            lc.set_array(time)
            lc.set_linewidth(1)

            cb = fig.colorbar(lc, shrink=0.8)
            cb.set_label('time [yr]')
            ax.add_collection(lc)
            #ax.scatter(orbiting_x_at_update, orbiting_y_at_update, s=10, color='k')

            all_x = numpy.hstack([CM_x, central_x, orbiting_x])
            all_y = numpy.hstack([CM_y, central_y, orbiting_y])
            ax.set_xlim(all_x.min() - 0.1*abs(all_x.min()), all_x.max() + 0.1*abs(all_x.max()) )
            ax.set_ylim(all_y.min() - 0.1*abs(all_y.min()), all_y.max() + 0.1*abs(all_y.max()) )
            ax.set_xlabel('x [AU]')
            ax.set_ylabel('y [AU]')
            ax.set_aspect('equal')

            plt.savefig(targetdir+imgname, bbox_inches='tight', dpi=150)
            plt.close()
            

        angular_momentum_vs_time()
        O_angular_momentum_vs_time()
        eccentricity_vs_time()
        sma_vs_time()    
        period_vs_time()
        true_anomaly_vs_time()
        argument_of_periapsis_vs_time()
        positions_in_time(frame='M')


    def fixed_ecc(simdata, *args, **kwargs):
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
        mdot = float(simdata.parameters['mdot'].split()[0]) 


        firstsim = f[integrator].values()[0]
        time = quantify_dset(firstsim['time']).value_in(units.yr)
        totalmass = quantify_dset(firstsim['mass']).lengths().value_in(units.MSun)

        smas = []
        eccentricities = []
        sma_labels = []
        smas_ad = []
        ml_indices = []
        periods = []
        
        for sim in f[integrator].values():
            ml = sim['p0/massloss_index'].value
            sma = quantify_dset(sim['p0/sma']).value_in(units.AU)
            smas.append(sma)
            smas_ad.append(sma_adiabatic(sma[0], mdot, time, centralmass))
            ml_indices.append(sim['p0/massloss_index'].value)
            eccentricities.append(sim['p0/eccentricity'].value)
            sma_labels.append("a0= "+str(sma[0]))
            periods.append(quantify_dset(sim['p0/period']).value_in(units.yr))

        cmap = cm.jet
        savefigargs = dict(bbox_inches='tight', dpi=150)

        def sma_vs_time():
            imgname = 'sma_vs_time.png'
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            colorcycle = cycle(['k','r','g','y', 'c', 'm', 'b'])

            for a, lbl in zip(smas, sma_labels):
                ax.plot(time, a/a[0], lw=1, ls='-', c=next(colorcycle), label=lbl)
                ax.set_xlim(time[0], time[-1])

            ax.legend(loc='best')
            ax.set_xlabel('time [yr]')
            ax.set_ylabel('sma/sma0 [AU]')
            ax.set_xticks(numpy.arange(0, 5000, 500))
            ax.grid(True)

            plt.savefig(targetdir+imgname, **savefigargs)
            plt.close()

        def sma_error_vs_time():
            imgname = 'sma_error_vs_time.png'
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            colorcycle = cycle(['k','r','g','y', 'c', 'm', 'b'])

            for a, ad, lbl in zip(smas, smas_ad, sma_labels):
                ax.plot(time, (a-ad)/ad, lw=1, ls='-', c=next(colorcycle), label=lbl)
                ax.set_xlim(time[0], time[-1])

            ax.set_xticks(numpy.arange(0, 5000, 500))
            ax.set_xlabel('time [yr]')
            ax.set_ylabel('(sma-sma_ad)/sma_ad [AU]')
            ax.grid(True)
            ax.legend(loc='best')

            plt.savefig(targetdir+imgname, **savefigargs)
            plt.close()

        sma_vs_time()
        sma_error_vs_time()

 
       
    #binaries(SimData('data/binaries_M1_m0.001_mdot5e-05_s10.0_i1.0_t5000.0_p1000_Sm_ph_He_Hu.hdf5', figdir='binaries_10/'))
    #binaries(SimData('data/binaries_M1_m0.001_mdot5e-05_s50.0_i1.0_t5000.0_p1000_Sm_ph_He_Hu.hdf5', figdir='binaries_50/'))
    #binaries(SimData('data/binaries_M1_m0.001_mdot5e-05_s100.0_i1.0_t5000.0_p1000_Sm_ph_He_Hu.hdf5', figdir='binaries_100/'))
    #binaries(SimData('data/binaries_M1_m0.001_mdot5e-05_s150.0_i1.0_t5000.0_p1000_Sm_ph_He_Hu.hdf5', figdir='binaries_150/'))
    #binaries(SimData('data/binaries_M1_m0.001_mdot5e-05_s200.0_i1.0_t5000.0_p1000_Sm_ph_He_Hu.hdf5', figdir='binaries_200/'))

    #binaries(SimData('data/binaries_M1_m0.001_mdot5e-05_s10.0_i5.0_t5000.0_p1000_Sm_ph_He_Hu.hdf5', figdir='binaries_10_i5/'))
    #binaries(SimData('data/binaries_M1_m0.001_mdot5e-05_s50.0_i5.0_t5000.0_p1000_Sm_ph_He_Hu.hdf5', figdir='binaries_50_i5/'))
    #binaries(SimData('data/binaries_M1_m0.001_mdot5e-05_s100.0_i5.0_t5000.0_p1000_Sm_ph_He_Hu.hdf5', figdir='binaries_100_i5/'))
    #binaries(SimData('data/binaries_M1_m0.001_mdot5e-05_s150.0_i5.0_t5000.0_p1000_Sm_ph_He_Hu.hdf5', figdir='binaries_150_i5/'))
    #binaries(SimData('data/binaries_M1_m0.001_mdot5e-05_s200.0_i5.0_t5000.0_p1000_Sm_ph_He_Hu.hdf5', figdir='binaries_200_i5/'))

    #binaries(SimData('data/binaries_M1_m0.001_mdot5e-05_s10.0_i10.0_t5000.0_p1000_Sm_ph_He_Hu.hdf5', figdir='binaries_10_i10/'))
    #binaries(SimData('data/binaries_M1_m0.001_mdot5e-05_s50.0_i10.0_t5000.0_p1000_Sm_ph_He_Hu.hdf5', figdir='binaries_50_i10/'))
    #binaries(SimData('data/binaries_M1_m0.001_mdot5e-05_s100.0_i10.0_t5000.0_p1000_Sm_ph_He_Hu.hdf5', figdir='binaries_100_i10/'))
    #binaries(SimData('data/binaries_M1_m0.001_mdot5e-05_s150.0_i10.0_t5000.0_p1000_Sm_ph_He_Hu.hdf5', figdir='binaries_150_i10/'))
    #binaries(SimData('data/binaries_M1_m0.001_mdot5e-05_s200.0_i10.0_t5000.0_p1000_Sm_ph_He_Hu.hdf5', figdir='binaries_200_i10/'))


    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.0_0.0_0.0_0.0_0.0_i1.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.0_0.0_0.0_0.0_0.0_i1.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.5_0.0_0.0_0.0_0.0_i1.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.5_0.0_0.0_0.0_0.0_i1.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.8_0.0_0.0_0.0_0.0_i1.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.8_0.0_0.0_0.0_0.0_i1.0_t5050.0_p20000_Sm.hdf5'))

    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.0_0.0_0.0_0.0_0.0_i0.5_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.0_0.0_0.0_0.0_0.0_i0.5_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.5_0.0_0.0_0.0_0.0_i0.5_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.5_0.0_0.0_0.0_0.0_i0.5_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.9_0.0_0.0_0.0_0.0_i0.5_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.9_0.0_0.0_0.0_0.0_i0.5_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.0_0.0_0.0_0.0_0.0_i10.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.5_0.0_0.0_0.0_0.0_i10.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.5_0.0_0.0_0.0_0.0_i10.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.0_0.0_0.0_0.0_0.0_i10.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.9_0.0_0.0_0.0_0.0_i10.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.9_0.0_0.0_0.0_0.0_i10.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.0_0.0_0.0_0.0_0.0_i25.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.0_0.0_0.0_0.0_0.0_i25.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.5_0.0_0.0_0.0_0.0_i25.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.5_0.0_0.0_0.0_0.0_i25.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.9_0.0_0.0_0.0_0.0_i25.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.9_0.0_0.0_0.0_0.0_i25.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.0_0.0_0.0_0.0_0.0_i50.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.0_0.0_0.0_0.0_0.0_i50.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.5_0.0_0.0_0.0_0.0_i50.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.5_0.0_0.0_0.0_0.0_i50.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.9_0.0_0.0_0.0_0.0_i50.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.9_0.0_0.0_0.0_0.0_i50.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.0_0.0_0.0_0.0_0.0_i100.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.0_0.0_0.0_0.0_0.0_i100.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.5_0.0_0.0_0.0_0.0_i100.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.5_0.0_0.0_0.0_0.0_i100.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.9_0.0_0.0_0.0_0.0_i100.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.9_0.0_0.0_0.0_0.0_i100.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.0_0.0_0.0_0.0_0.0_i250.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.0_0.0_0.0_0.0_0.0_i250.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.5_0.0_0.0_0.0_0.0_i250.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.5_0.0_0.0_0.0_0.0_i250.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.9_0.0_0.0_0.0_0.0_i250.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.9_0.0_0.0_0.0_0.0_i250.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.0_0.0_0.0_0.0_0.0_i500.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.0_0.0_0.0_0.0_0.0_i500.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.5_0.0_0.0_0.0_0.0_i500.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.5_0.0_0.0_0.0_0.0_i500.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.9_0.0_0.0_0.0_0.0_i500.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.9_0.0_0.0_0.0_0.0_i500.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.0_0.0_0.0_0.0_0.0_i1000.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.0_0.0_0.0_0.0_0.0_i1000.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.5_0.0_0.0_0.0_0.0_i1000.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.5_0.0_0.0_0.0_0.0_i1000.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.9_0.0_0.0_0.0_0.0_i1000.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.9_0.0_0.0_0.0_0.0_i1000.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.0_0.0_0.0_0.0_0.0_i1500.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.0_0.0_0.0_0.0_0.0_i1500.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.5_0.0_0.0_0.0_0.0_i1500.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.5_0.0_0.0_0.0_0.0_i1500.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.9_0.0_0.0_0.0_0.0_i1500.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.9_0.0_0.0_0.0_0.0_i1500.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.0_0.0_0.0_0.0_0.0_i2000.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.0_0.0_0.0_0.0_0.0_i2000.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.5_0.0_0.0_0.0_0.0_i2000.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.5_0.0_0.0_0.0_0.0_i2000.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.9_0.0_0.0_0.0_0.0_i2000.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.9_0.0_0.0_0.0_0.0_i2000.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.0_0.0_0.0_0.0_0.0_i2500.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.0_0.0_0.0_0.0_0.0_i2500.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.5_0.0_0.0_0.0_0.0_i2500.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.5_0.0_0.0_0.0_0.0_i2500.0_t5050.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.9_0.0_0.0_0.0_0.0_i2500.0_t5000.0_p20000_Sm.hdf5'))
    simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.9_0.0_0.0_0.0_0.0_i2500.0_t5050.0_p20000_Sm.hdf5'))
    #simple_binary(SimData('data/single_binary_M1_m0.0001_mdot0.0001_elems5.0_0.9_0.0_0.0_0.0_0.0_i100.0_t5000.0_p10000_Sm.hdf5'))





    #fixed_ecc(SimData('data/testecc.hdf5',figdir='testecc/'))
    #fixed_ecc(SimData('data/fixed_ecc_i10.hdf5',figdir='fixed_ecc_i10/'))
    #fixed_ecc(SimData('data/fixed_ecc_i50.hdf5',figdir='fixed_ecc_i50/'))
    #fixed_ecc(SimData('data/fixed_ecc_i500.hdf5',figdir='fixed_ecc_i500/'))

def axis_position(rows, columns):
    total = rows * columns
    i=1
    while i <= total:
        yield (rows, columns, i)
        i += 1

def sma_adiabatic(a0, mdot, t, mu0):
    return a0*(1 - mdot*t/mu0)**(-1)

def ecc_adiabatic(e0, psi0, true_anomaly):
    f = numpy.radians(true_anomaly)
    return e0+psi0*(1-e0**2)**(3/2.0)*numpy.sin(true_anomaly)/(1-e0*numpy.cos(true_anomaly))
    

if __name__ == '__main__':
    mpl.rc('font',  size=9, family='sans-serif')
    main()
