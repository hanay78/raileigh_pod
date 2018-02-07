#!/usr/bin/env python

"""Calculates the pods in a separate manner"""

import os
import fnmatch
import argparse
import numpy as np
from mpi4py import MPI

from auxiliary_pod_calculator import calculate_pod_simple_accuracy
from order_database import order_files

if __name__ == '__main__':



    DBSES = []
    for my_file in os.listdir('./'):
        if fnmatch.fnmatch(my_file, 'XYZ*.csv'):
            DBSES.append(my_file)

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--filesdb', default=DBSES, nargs='+', help='files')
    PARSER.add_argument('--number_column', default=0, nargs=1, help='column to analize')
    PARSER.add_argument('--snapshots_path', default=['./snapshots/'], nargs=1, help='files')
    PARSER.add_argument('--accuracy_pod', default=0.001, nargs=1, type=np.float64)

    ARGS = PARSER.parse_args()

    COMM = MPI.COMM_WORLD
    SIZE = COMM.Get_size()
    RANK = COMM.Get_rank()

    (TIMES, NEWNAMES_FILES) = order_files(ARGS.snapshots_path[0], ARGS.filesdb, RANK)

    COORS = np.empty([0, 0])

    TEMP = np.zeros(0, dtype=np.float64)

    FIRST_FLAG = True

    for my_file in NEWNAMES_FILES:
        datos = np.loadtxt(my_file, delimiter=',', skiprows=1)
        datos = datos.copy(order='C')

        Temp_div = np.array_split(datos[:, ARGS.number_column], SIZE)
        Temp_local = np.array(COMM.scatter(Temp_div, root=0))

        if FIRST_FLAG:
            coors_div = np.array_split(datos[:, -3:datos.shape[1]], SIZE, axis=0)
            coors_local = np.array(COMM.scatter(coors_div, root=0))

            TEMP = np.append(TEMP, Temp_local)
            COORS = np.append(COORS, coors_local[:, 0])
            COORS = np.vstack([COORS, coors_local[:, 1]])
            COORS = np.vstack([COORS, coors_local[:, 2]])
            COORS = COORS.transpose()

            FIRST_FLAG = False
        else:
            TEMP = np.vstack([TEMP, Temp_local])



    TEMP = TEMP.copy(order='C').transpose()

    (LTHETA, UTHETA, N_T, N_T1) = calculate_pod_simple_accuracy(TEMP,
                                                                TEMP,
                                                                ARGS.accuracy_pod[0],
                                                                1.,
                                                                COMM, SIZE, RANK)
#    (avT, theta, LTHETA, UTHETA) = calculate_pod(Temp, ARGS.accuracy_pod, COMM, SIZE, rank)

    print "Number of POD obtained: %i" % LTHETA.size
    print "Eigenvalues"
    print LTHETA
    #print LTHETA.size()
    SERVICE_AUX = np.vstack([COORS[:, 0], COORS[:, 1], COORS[:, 2]])
    SERVICE_AUX = np.vstack([SERVICE_AUX, UTHETA.transpose()]).transpose()

    for p in range(SIZE):
        if RANK == p:
            if RANK == 0:
                f = open('us_sccm.dat', 'wb')
                np.savetxt(f, SERVICE_AUX)
                f.close()
            else:
                f = open('us_sccm.dat', 'ab')
                np.savetxt(f, SERVICE_AUX)
                f.close()
        COMM.Barrier()

