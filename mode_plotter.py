#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
from mpi4py import MPI
import h5py
import sys
import copy

def plot_pod(datos, dx, dir, name, pod, path):


    c= datos[:,-3:datos.shape[1]]
    c=np.round(c/dx)*dx

    xs = np.unique(c[:, 0])
    ys = np.unique(c[:, 1])
    zs = np.unique(c[:, 2])


    if dir==2:
        args = np.argwhere(c[:, 2] == zs[np.round(zs.shape[0] / 2)])
        data=datos[args][:,0,:]
        x=data[:,-3]
        y=data[:,-2]
    if dir==1:
        args = np.argwhere(c[:, 1] == zs[np.round(ys.shape[0] / 2)])
        data=datos[args][:,0,:]
        x=data[:,-3]
        y=data[:,-1]
    if dir==0:
        args = np.argwhere(c[:, 0] == zs[np.round(xs.shape[0] / 2)])
        data=datos[args][:,0,:]
        x=data[:,-2]
        y=data[:,-1]

    if pod:
        for i in range(0, datos.shape[1]-3):
            zz=data[:,i]
            z=np.zeros(zz.shape)
            z=zz+ random.uniform(0,1.0e-10)
            # for j in range(z.shape[0]):
            #     z[j]='%.07e' % (zz[j]) + random.uniform(0,1.0e-10)
            q = "%05i" % (i)
            plt.figure()
            plt.gca().set_aspect('equal')
            #plt.tricontourf(x, y, z, np.arange(-0.15,0.15, 0.015))
            plt.tricontourf(x, y, z, 20)
            plt.colorbar()
            plt.savefig(path+name+"_"+str(q)+'.png')
            plt.close()
    else:
        for i in range(0, 5):
#        z=data[:,int(variable)]
            z=data[:,i]

            nam=['P', 'U', 'V', 'W', 'T']

            plt.figure()
            plt.gca().set_aspect('equal')
            plt.tricontourf(x, y, z, 20)
            plt.colorbar()
            plt.savefig(path+name+"_"+nam[i]+'.png')
            plt.close()


def modeploter_function(path, file_pod, name, pod_yes=False, h5_local=True, dir=2, dx=0.000625, header=False):
    print "ploting"
    print path+file_pod

    path_local=copy.deepcopy(path)

    if path[-1] != '/':
        service=str(path_local)
        service+="/"
        path_local=service
    if h5_local:
        fhdf = h5py.File(path_local+file_pod, "r")
        data = fhdf['All'][()]
        fhdf.close()
    else:
        if header:
            data=np.loadtxt(path+file_pod, delimiter=',',skiprows=1)
        else:
            data=np.loadtxt(path+file_pod,skiprows=1)

    plot_pod(data, dx, dir, name, pod_yes, path)


if __name__ == '__main__':
    print "ploting"



    parser = argparse.ArgumentParser()
    parser.add_argument('--X_dir', default=False, action='store_true')
    parser.add_argument('--Y_dir', default=False, action='store_true')
    parser.add_argument('--Z_dir', default=True, action='store_true')
    parser.add_argument('--dx', default=0.000625)#, action='store_true')
#    parser.add_argument('--file_pod', default='us_theta_sccm.dat', nargs=1, help='files')
    parser.add_argument('--file_pod', default='database_00000.csv', nargs=1, help='files')
    parser.add_argument('--header', default=False, action='store_true', help='files')
    parser.add_argument('--pod', default=False, action='store_true', help='files')
    parser.add_argument('--name', default='pod_', nargs=1, help='files')
    parser.add_argument('--path', default=['./kkj'], nargs=1, help='files')
    parser.add_argument('--h5_local', default=False, action='store_true')
    args = parser.parse_args()

    dir=2

    #print args
    print "Enters mode_plotter"
    if args.X_dir:
        dir=0
    if args.Y_dir:
        dir=1

    if args.path[0][-1] != '/':
        service=str(args.path[0])
        service+="/"
        path=service

    if args.h5_local:
        fhdf = h5py.File(path+args.file_pod[0], "r")
#        fhdf = h5py.File(path+args.file_pod[0], "r", driver='mpio', comm=MPI.COMM_WORLD)
        data = fhdf['All'][()]
        fhdf.close()
    else:
        if args.header:
            data=np.loadtxt(path+args.file_pod[0], delimiter=',',skiprows=1)
        else:
            data=np.loadtxt(path+args.file_pod[0],skiprows=1)

    # print "llego aqui"
    # exit()

    print "ploting"
    print path+args.file_pod[0]
    plot_pod(data, args.dx, dir, args.name[0], args.pod, path)
