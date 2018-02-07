#!/usr/bin/env python

import os.path
import copy
import numpy as np
import matplotlib.pyplot as plt
import h5py


from auxiliary_classes import round_coordinates
from physical_conditions import round_cell_size

def plot_velocity_field(path, archfile, snapshots_path, h5format=False):

    """print vector graphs from a file where all velocity directions is stored"""
    if h5format:
        fhdf = h5py.File(path+archfile, "r")
        data = fhdf['All'][()]
        fhdf.close()
    else:
        data = np.loadtxt(path+archfile)

    data[:, 5:8] = round_coordinates(data[:, 5:8], round_cell_size)
#    data[:,5:8]=np.round(data[:,5:8]/0.000625)*0.000625

    coor_x = np.unique(data[:, 5])
    coor_y = np.unique(data[:, 6])
    coor_z = np.unique(data[:, 7])

    mesgrid_x, mesgrid_y = np.meshgrid(coor_x, coor_y)

    args = np.argwhere(data[:, 7] == coor_z[np.round(coor_z.shape[0] / 2)])

    data = data[args][:, 0, :]

    mesgrid_u = copy.deepcopy(mesgrid_x)
    mesgrid_v = copy.deepcopy(mesgrid_y)

    order = np.zeros((mesgrid_u.shape[0], mesgrid_u.shape[1]))

    if os.path.isfile(snapshots_path+'key_of_order.dat'):

        order = np.loadtxt(snapshots_path+'key_of_order.dat', dtype=np.int32)

        for i in range(mesgrid_u.shape[0]):
            for j in range(mesgrid_u.shape[1]):
                order_l = order[i, j]
                mesgrid_u[i, j] = data[order_l, 1]
                mesgrid_v[i, j] = data[order_l, 2]
    else:

        for i in range(mesgrid_u.shape[0]):
            for j in range(mesgrid_u.shape[1]):
                for k in range(data.shape[0]):
                    if mesgrid_x[i, j] == data[k, 5]:
                        if mesgrid_y[i, j] == data[k, 6]:
                            print k
                            order[i, j] = k
                            mesgrid_u[i, j] = data[k, 1]
                            mesgrid_v[i, j] = data[k, 2]
                            break

        np.savetxt(snapshots_path+'key_of_order.dat', order)


    plt.figure()

    plt.quiver(mesgrid_x[::3, ::3], mesgrid_y[::3, ::3], mesgrid_u[::3, ::3], mesgrid_v[::3, ::3],
               pivot='mid', units='inches')
    if h5format:
        plt.savefig(path+'Vels'+archfile[: -3]+'.png')
    else:
        plt.savefig(path+'Vels'+archfile[: -4]+'.png')
    plt.close()


def plot_velocity_field_files(name, path, archfileu, archfilev, pathorder="./"):

    """print vector graphs from files where velocity directions are stored"""

    datau = np.loadtxt(archfileu)
    datav = np.loadtxt(archfilev)
    coor_rounded = round_coordinates(datau[:, -3:], round_cell_size)
    #datau[:,-3:]=np.round(datau[:,-3:]/0.000625)*0.000625

    coor_x = np.unique(coor_rounded[:, 0])
    coor_y = np.unique(coor_rounded[:, 1])
    coor_z = np.unique(coor_rounded[:, 2])

    mesgrid_x, mesgrid_y = np.meshgrid(coor_x, coor_y)

    args = np.argwhere(coor_rounded[:, 2] == coor_z[np.round(coor_z.shape[0] / 2)])[:, 0]

    datau = datau[args, :]
    datav = datav[args, :]

    mesgrid_u = copy.deepcopy(mesgrid_x)
    mesgrid_v = copy.deepcopy(mesgrid_y)

    coor_rounded_reduced = coor_rounded[args, :]

    order = np.zeros((mesgrid_u.shape[0], mesgrid_u.shape[1]), dtype=np.int32)

    if os.path.isfile(pathorder+'key_of_order.dat'):

        order = np.loadtxt(pathorder+'key_of_order.dat', dtype=np.int32)

        # for i in range(mesgrid_u.shape[0]):
        #     for j in range(mesgrid_u.shape[1]):
        #         l=order[i,j]
        #         mesgrid_u[i,j]=data[l,1]
        #         mesgrid_v[i,j]=data[l,2]
    else:

        for i in range(mesgrid_u.shape[0]):
            for j in range(mesgrid_u.shape[1]):
                for k in range(datau.shape[0]):
                    if mesgrid_x[i, j] == coor_rounded_reduced[k, 0]:
                        if mesgrid_y[i, j] == coor_rounded_reduced[k, 1]:
                            print k
                            order[i, j] = k
                            # mesgrid_u[i,j]=data[k,1]
                            # mesgrid_v[i,j]=data[k,2]
                            break

        np.savetxt(pathorder+'key_of_order.dat', order)

    for k in range(datau.shape[1] - 3):
        for i in range(mesgrid_u.shape[0]):
            for j in range(mesgrid_u.shape[1]):
                order_l = order[i, j]
                mesgrid_u[i, j] = datau[order_l, k]
                mesgrid_v[i, j] = datav[order_l, k]


        plt.figure()

        plt.quiver(mesgrid_x[::3, ::3], mesgrid_y[::3, ::3], mesgrid_u[::3, ::3], mesgrid_v[::3, ::3],
                   pivot='mid', units='inches')
        name_q = "%05i" % (k)
        plt.savefig(path+name+name_q+'.png')
        plt.close()
