#!/usr/bin/env python

import argparse
import h5py
import matplotlib.pyplot as plt
#from amuse.units import units as u
#from amuse.units import core

def main():
    f = h5py.File(args.filename, 'r')

    #unit = eval(units_string, core.__dict__)

    plot_a_vs_deltat(f)
    plot_position(f)

def plot_a_vs_deltat(f):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    systems = f.values()

    for s in systems:
        sequences = s.values()

        x = []
        y = []
        for seq in sequences:

            sma0 = seq['sma'][0]
            sma = seq['sma'][-1]
            delta_t = seq['time'][-1]
            period0 = seq['period0'][0]

            x.append(delta_t/period0)
            y.append(sma/sma0)
        

        mass0 = sequences[0]['mass'][0][0]
        mass1 = sequences[0]['mass'][0][1]

        massratio = round(mass0/mass1, 1)

        ax.plot(x, y, marker='o', markersize=4, label=str(s.name)+ "  massratio central/orbiting: {}".format(massratio) )
        ax.set_xlabel('delta_t / period0')
        ax.set_ylabel('a / a0')

    ax.legend()
    plt.show(fig)



def plot_position(f):
    """ Just testing, remove later """

    systems = f.values()

    for s in systems:
        sequences = s.values()

        for seq in sequences:

            position = seq['position']
            sma = seq['sma'][-1]
            delta_t = seq['time'][-1]
            #period0 = seq['period0'][0]

            mass0 = sequences[0]['mass'][0][0]
            mass1 = sequences[0]['mass'][0][1]

            massratio = round(mass0/mass1, 1)

            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111)
            ax.scatter(position[:,0,0], position[:,0,1], color='g')
            ax.scatter(position[:,1,0], position[:,1,1], color='b')
            #ax.plot(x, y, marker='o', markersize=4, label=str(s.name)+ "  massratio central/orbiting: {}".format(massratio) )
            ax.set_xlabel('delta_t / period0')
            ax.set_ylabel('a / a0')

            ax.legend()

            name = str(s.name)+"_"+str(delta_t)
            plt.savefig('temp/'+name+".png")
            fig.clf()
            plt.close()



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--filename', required=True,
                        help="hdf5 file created by sim_binary_system_with_massloss.py")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_arguments()
    print(args)
    main()

    

