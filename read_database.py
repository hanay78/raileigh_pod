#!/usr/bin/env python
import copy
import sys
import os.path
import fnmatch
import numpy as np
from mpi4py import MPI
import h5py
from vectors import Vectors
from auxiliary_classes import Intexes, round_coordinates, coordinates
from phis_calculator import calculate_neighbors, Neigbors, Boundary
from physical_conditions import number_ghosh_cells, round_cell_size
from order_database import order_files

def read_database(newnames_files, comm, size, rank):

    """Read the database from separated txt files. Distribute it between processes"""

### reading database in texfiles

    ### initilize vectors ###
    coors = coordinates()

    dim_vars = Vectors()
    dim_vars.zeros()

    neigbours = Neigbors()
    boundaries = Boundary()

    new_indexes = Intexes()
    new_indexes.set_csr(comm, size, rank)

    first_flag = True

    for my_file in newnames_files:

        #### read file ####
        comm.barrier()
        if rank == 0:
            print "-------------------------------------------------------------------"
        sys.stdout.flush()
        comm.barrier()
        print "Processor %d Reading and processing data of file %s" % (rank, my_file)
        sys.stdout.flush()
        comm.barrier()
        if rank == 0:
            print "-------------------------------------------------------------------"
        sys.stdout.flush()
        comm.barrier()
        datos_ori = np.loadtxt(my_file, delimiter=',', skiprows=1, dtype=np.float64)

        ### re-order read data
        datos_ori = datos_ori.copy(order='C')

        if first_flag:

            new_indexes.re_order(datos_ori[:, -3:datos_ori.shape[1]])
            datos = datos_ori[new_indexes.inew[:]]

            coors_round = round_coordinates(copy.deepcopy(datos[:, -3:datos.shape[1]]),
                                            round_cell_size)

            new_indexes.organize_split_vectors(coors_round, number_ghosh_cells)

            coors.coors_dim_rounded = new_indexes.scatter_vector(coors_round.copy())
            [neigbours, boundaries] = calculate_neighbors(coors.coors_dim_rounded,
                                                          comm, size, rank)

            temp_local = new_indexes.scatter_vector(datos[:, 4])
            pres_local = new_indexes.scatter_vector(datos[:, 0])
            vels_local = new_indexes.scatter_vector(datos[:, 1:4])

            coors.coors_global_dim_rounded = round_coordinates(datos[:, -3:datos.shape[1]],
                                                               round_cell_size)

            dim_vars.append(pres_local,
                            vels_local[:, 0],
                            vels_local[:, 1],
                            vels_local[:, 2],
                            temp_local)

            first_flag = False
        else:

            datos = datos_ori[new_indexes.inew[:]]

            temp_local = new_indexes.scatter_vector(datos[:, 4])
            pres_local = new_indexes.scatter_vector(datos[:, 0])
            vels_local = new_indexes.scatter_vector(datos[:, 1:4])

            dim_vars.vstack(pres_local,
                            vels_local[:, 0],
                            vels_local[:, 1],
                            vels_local[:, 2],
                            temp_local)

    dim_vars.transpose()
    new_indexes.local_indices()
    new_indexes.distribute_indexes()

    return [dim_vars, coors, neigbours, boundaries, new_indexes]


def read_database_h5(hdf5_database_file, comm, size, rank):

    """Read the snapshot database from a hdf5 file and distribute it into the corresponding database"""

###read database in h5 files


    if rank == 0:
        print "-----------------------------------------------"
        print "Reading preprocessed HDF5 file with snapshots"
        print "-----------------------------------------------"

    ### read datatable ###
    fhdf = h5py.File(hdf5_database_file, "r", driver='mpio', comm=MPI.COMM_WORLD)

    coors = coordinates()


    ts_dim = fhdf['ts_dim'][()]
    coors.coors_global_dim_rounded = fhdf['coors_global_dim_rounded'][()]

    new_indexes = Intexes()
    new_indexes.set_index(coors.coors_global_dim_rounded)
    new_indexes.set_csr(comm, size, rank)
    new_indexes.organize_split_vectors(copy.deepcopy(coors.coors_global_dim_rounded),
                                       number_ghosh_cells)

    coors.coors_dim_rounded =\
        new_indexes.scatter_vector(copy.deepcopy(coors.coors_global_dim_rounded))

    [neigbours, boundaries] = calculate_neighbors(coors.coors_dim_rounded,
                                                  comm, size, rank)

    dim_vars = Vectors()

    dim_vars.read_in_hdf5_file(fhdf, new_indexes.patch_indexes_ghosh_cells)

    fhdf.close()

    if rank == 0:
        print "-----------------------------------------------"
        print "Reading and postprocesing database finished    "
        print "-----------------------------------------------"

    return [dim_vars, coors, neigbours, boundaries, ts_dim, new_indexes]

def write_database_h5(hdf5_database_file,
                      ts_dim,
                      coors,
                      dim_vars,
                      new_indexes,
                      comm, size, rank):


    """Write the distributed snapshot in a single hdf5 file"""

### write h5 files from textfiles

    fhdf = h5py.File(hdf5_database_file, "w", driver='mpio', comm=MPI.COMM_WORLD)

    tsglobal = fhdf.create_dataset("ts_dim", (ts_dim.shape[0],), dtype=np.float64)
    if rank == 0:
        tsglobal[:] = ts_dim[:]

    cgdr = fhdf.create_dataset("coors_global_dim_rounded",
                               (coors.coors_global_dim_rounded.shape[0],
                                coors.coors_global_dim_rounded.shape[1]),
                               dtype=np.float64)

    if rank == 0:
        cgdr[:, :] = coors.coors_global_dim_rounded[:, :]

    dim_vars.save_in_hdf5_file(fhdf,
                               coors.coors_global_dim_rounded.shape[0],
                               new_indexes.global_indices_inner,
                               new_indexes.local_indices_inner)

    fhdf.close()

def provide_database(args, comm, size, rank):

    """General file organizing the processing of the database"""

    hdf5_database_file = args.snapshots_path[0] + "whole_database.hdf5"

    if os.path.isfile(hdf5_database_file) and args.read_processed_data_hdf5:
        [dim_vars,
         coors,
         neigbours,
         boundaries,
         ts_dim,
         new_indexes] = read_database_h5(hdf5_database_file, comm, size, rank)

    else:

        ### locate candidates for the database ###
        dbses = []
        for my_file in os.listdir(args.snapshots_path[0]):
            if fnmatch.fnmatch(my_file, args.snapshots_pattern[0]):
                dbses.append(my_file)




                ### re-order files ###
                #    (ts_dim, newnames_files)=order_files(args.filesdb)
        (ts_dim, newnames_files) = order_files(args.snapshots_path[0], dbses, rank)

        #################################################
        [dim_vars,
         coors,
         neigbours,
         boundaries,
         new_indexes]\
            = read_database(newnames_files, comm, size, rank)

        #################################################

        write_database_h5(hdf5_database_file,
                          ts_dim,
                          coors,
                          dim_vars,
                          new_indexes,
                          comm, size, rank)

    return [dim_vars,
            coors,
            neigbours,
            boundaries,
            ts_dim,
            new_indexes]
