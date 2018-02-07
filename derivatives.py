import os
import numpy as np
from mpi4py import MPI
from vectors import Vectors
import h5py


def coeficients(M, alpha):

    """"Generation of coefficients
    of finite differences formulas

    This function follows the article

    Generation of finite difference formulas
    on arbitrary Spaced grids

    Bengt Fornberg

    Mathematics of computation, 52, 184, 1988

    """
    x0 = np.float64(0.)
    N = alpha.size
    delta = np.zeros((M+1, N, N), dtype=np.float64)
    delta[0, 0, 0] = np.float64(1.)
    c1 = np.float64(1.)

    for n in range(1, N):
        c2 = np.float64(1.)
        for nu in range(n):
            c3 = alpha[n] - alpha[nu]
            c2 = c2*c3
            # if n <= M:
            #     delta[n,n-1, nu]=0.
            # else:
            for m in range(np.min(np.array([n, M])+1)):
                delta[m, n, nu] = ((alpha[n] - x0) * delta[m, n-1, nu]
                                   - m * delta[m-1, n-1, nu]) \
                                   / c3
        for m in range(np.min(np.array([n, M])+1)):
            delta[m, n, n] = c1 / c2 * \
                             (m * delta[m-1, n-1, n-1]
                              - (alpha[n-1] - x0)
                              * delta[m, n-1, n-1])
        c1 = c2
    # return delta
    delta_corrected = np.zeros((M+1, N, N), dtype=np.float64)

    for m in range(1, delta.shape[0]):
        for n in range(1, delta.shape[1]):
            if n >= m:
                delta_corrected[m, n-m+1] = delta[m, n]

    return delta_corrected


def calculate_stencil_coeficients_dx_patch(coors_all_dirs, neigh_all_dirs, new_indexes):
    """ Stelcil coefficients for the first derievative"""

    order = 2
    deltas_all_dirs = np.zeros((3, coors_all_dirs.shape[0], 3), dtype=np.float64)
    for j in range(3):
        [left, right] = neigh_all_dirs[j]
        coors = coors_all_dirs[:, j]
        for i in new_indexes.local_indices_ghosh_cells_minus_2:
            if left[i] == -1:
                aux = coeficients(1,
                                  np.array([0.,
                                            coors[right[i]] - coors[i],
                                            coors[right[right[i]]] - coors[i]]))
                #deltas_all_dirs[j, i, :] = aux[1][order]
                deltas_all_dirs[j, i, :] = aux[1][order-1]
            elif right[i] == -1:
                aux = coeficients(1,
                                  np.array([0.,
                                            coors[left[i]] - coors[i],
                                            coors[left[left[i]]] - coors[i]]))
                #deltas_all_dirs[j, i, :] = aux[1][order]
                deltas_all_dirs[j, i, :] = aux[1][order-1]
            else:
                aux = coeficients(1,
                                  np.array([0.,
                                            coors[right[i]] - coors[i],
                                            coors[left[i]] - coors[i]]))
                deltas_all_dirs[j, i, :] = aux[1][order]


    return  deltas_all_dirs


def calculate_stencil_coeficients_dxdx_patch(coors_all_dirs, neigh_all_dirs, new_indexes):

    """ Stelcil coefficients for the second derievative"""

    #order = 2
    deltas_all_dirs = np.zeros((3, coors_all_dirs.shape[0], 5), dtype=np.float64)
    for j in range(3):
        [left, right] = neigh_all_dirs[j]
        coors = coors_all_dirs[:, j]
        for i in new_indexes.local_indices_inner:
            if left[i] == -1:
                aux = coeficients(2, np.array([0.,
                                               coors[right[i]]-coors[i],
                                               coors[right[right[i]]]-coors[i],
                                               coors[right[right[right[i]]]]-coors[i],
                                               coors[right[right[right[right[i]]]]]-coors[i]]))
                #deltas_all_dirs[j, i, :] = aux[2][3]
                deltas_all_dirs[j, i, :] = aux[2][1]

            elif right[i] == -1:
                aux = coeficients(2, np.array([0.,
                                               coors[left[i]]-coors[i],
                                               coors[left[left[i]]]-coors[i],
                                               coors[left[left[left[i]]]]-coors[i],
                                               coors[left[left[left[left[i]]]]]-coors[i]]))
                #deltas_all_dirs[j, i, :] = aux[2][3]
                deltas_all_dirs[j, i, :] = aux[2][1]

            else:
                if left[left[i]] == -1:
                    aux = coeficients(2, np.array([0.,
                                                   coors[right[i]]-coors[i],
                                                   coors[left[i]]-coors[i],
                                                   coors[right[right[i]]]-coors[i],
                                                   coors[right[right[right[i]]]]-coors[i]]))
                    #deltas_all_dirs[j,i,:] = aux[2][3]
                    deltas_all_dirs[j, i, :] = aux[2][1]

                elif right[right[i]] == -1:
                    aux = coeficients(2, np.array([0.,
                                                   coors[right[i]]-coors[i],
                                                   coors[left[i]]-coors[i],
                                                   coors[left[left[i]]]-coors[i],
                                                   coors[left[left[left[i]]]]-coors[i]]))
                    # aux = coeficients(2, np.array([0.,
                    #                                coors[right[i]]-coors[i],
                    #                                coors[left[i]]-coors[i],
                    #                                np.abs(coors[left[left[i]]]-coors[i]),
                    #                                np.abs(coors[left[left[left[i]]]]-coors[i])]))
                    #deltas_all_dirs[j,i,:] = aux[2][3]
                    deltas_all_dirs[j, i, :] = aux[2][1]

                else:
                    aux = coeficients(2, np.array([0.,
                                                   coors[right[i]]-coors[i],
                                                   coors[left[i]]-coors[i],
                                                   coors[right[right[i]]]-coors[i],
                                                   coors[left[left[i]]]-coors[i]]))
                    deltas_all_dirs[j, i, :] = aux[2][3]
                    #deltas_all_dirs[j,i,:] = aux[2][order]


    return deltas_all_dirs


def calculate_dx_patch(direction, deltas, neigh, field, new_index):

    """calcualtion of first derivative"""

    [left, right] = neigh

    field_dx = np.zeros(field.shape, dtype=np.float64)
    for i in new_index.local_indices_ghosh_cells_minus_2:
        delta = deltas[i]
        if left[i] == -1:
            field_dx[i] = delta[0]*field[i] \
                          + delta[1]*field[right[i]] \
                          + delta[2]*field[right[right[i]]]

        elif right[i] == -1:
            field_dx[i] = delta[0]*field[i] \
                          + delta[1]*field[left[i]] \
                          + delta[2]*field[left[left[i]]]

        else:
            field_dx[i] = delta[0]*field[i] \
                          + delta[1]*field[right[i]] \
                          + delta[2]*field[left[i]]

    return field_dx



def calculate_dxdx_patch(direction, deltas, neigh, field, new_index):

    """calcualtion of second derivative"""

    [left, right] = neigh
    field_dx = np.zeros(field.shape, dtype=np.float64)
    for i in new_index.local_indices_inner:
        delta = deltas[i]
        if left[i] == -1:
            field_dx[i] = delta[0]*field[i] \
                          + delta[1]*field[right[i]] \
                          + delta[2]*field[right[right[i]]]\
                          + delta[3]*field[right[right[right[i]]]] \
                          + delta[4]*field[right[right[right[right[i]]]]]
        elif right[i] == -1:
            field_dx[i] = delta[0]*field[i] \
                          + delta[1]*field[left[i]] \
                          + delta[2]*field[left[left[i]]]\
                          + delta[3]*field[left[left[left[i]]]] \
                          + delta[4]*field[left[left[left[left[i]]]]]
        else:
            if left[left[i]] == -1:
                field_dx[i] = delta[0]*field[i] \
                              + delta[1]*field[right[i]] \
                              + delta[2]*field[left[i]]\
                              + delta[3]*field[right[right[i]]] \
                              + delta[4]*field[right[right[right[i]]]]
            elif right[right[i]] == -1:
                field_dx[i] = delta[0]*field[i] \
                              + delta[1]*field[right[i]] \
                              + delta[2]*field[left[i]]\
                              + delta[3]*field[left[left[i]]] \
                              + delta[4]*field[left[left[left[i]]]]
            else:
                field_dx[i] = delta[0]*field[i] \
                              + delta[1]*field[right[i]] \
                              + delta[2]*field[left[i]]\
                              + delta[3]*field[right[right[i]]] \
                              + delta[4]*field[left[left[i]]]

    return field_dx


def calculate_1_derivatives(read_stencils,
                            snapshots_path,
                            dimless_vars,
                            coors,
                            new_indexes, neigbours,
                            rank):

    """full procedure for the calcuzlation of the first derivative in all variables"""

    first_derivative = Vectors()
    first_derivative.P = np.zeros((dimless_vars.P.shape[0],
                                   dimless_vars.P.shape[1],
                                   3),
                                  dtype=np.float64)
    first_derivative.T = np.zeros((dimless_vars.T.shape[0],
                                   dimless_vars.T.shape[1],
                                   3),
                                  dtype=np.float64)
    first_derivative.u = np.zeros((dimless_vars.u.shape[0],
                                   dimless_vars.u.shape[1],
                                   3),
                                  dtype=np.float64)
    first_derivative.v = np.zeros((dimless_vars.v.shape[0],
                                   dimless_vars.v.shape[1],
                                   3),
                                  dtype=np.float64)
    first_derivative.w = np.zeros((dimless_vars.w.shape[0],
                                   dimless_vars.w.shape[1],
                                   3),
                                  dtype=np.float64)

    neigh = np.array([[neigbours._I, neigbours.I_],
                      [neigbours._J, neigbours.J_],
                      [neigbours._K, neigbours.K_]])

    namef = snapshots_path+"first_derivatives_stencil.hdf5"
    if read_stencils and os.path.isfile(namef):
        if rank == 0:
            print "--------------------------------------"
            print "Reading stencil of the first derivatives"
            print "--------------------------------------"
        delta_dx = np.zeros((3, coors.coors_dim_rounded.shape[0], 3), dtype=np.float64)
        fhdf = h5py.File(namef, "r", driver='mpio', comm=MPI.COMM_WORLD)
        delta_dx[:, new_indexes.local_indices_ghosh_cells_minus_2, :] = \
            fhdf["first_derivatives_stencil"][:, new_indexes.global_indices_ghosh_cells_minus_2, :]
        fhdf.close()
    else:
        if rank == 0:
            print "--------------------------------------"
            print "Calculating stencil of the derivatives"
            print "      First derivatives stencil"
            print "--------------------------------------"
        delta_dx = calculate_stencil_coeficients_dx_patch(coors.coors_dimless_rounded,
                                                          neigh,
                                                          new_indexes)
        fhdf = h5py.File(namef, "w", driver='mpio', comm=MPI.COMM_WORLD)
        dersten = fhdf.create_dataset("first_derivatives_stencil",
                                      (3,
                                       coors.coors_global_dim_rounded.shape[0],
                                       3))
        dersten[:, new_indexes.global_indices_inner, :] = \
            delta_dx[:, new_indexes.local_indices_inner, :]
        fhdf.close()

    if rank == 0:
        print "--------------------------------------"
        print "Calculating 1st derivatives"
        print "--------------------------------------"
    for i in range(3):

        first_derivative.P[ :, :, i] = calculate_dx_patch(i,
                                                          delta_dx[i],
                                                          neigh[i],
                                                          dimless_vars.P,
                                                          new_indexes)
        first_derivative.u[ :, :, i] = calculate_dx_patch(i,
                                                          delta_dx[i],
                                                          neigh[i],
                                                          dimless_vars.u,
                                                          new_indexes)
        first_derivative.v[ :, :, i] = calculate_dx_patch(i,
                                                          delta_dx[i],
                                                          neigh[i],
                                                          dimless_vars.v,
                                                          new_indexes)
        first_derivative.w[ :, :, i] = calculate_dx_patch(i,
                                                          delta_dx[i],
                                                          neigh[i],
                                                          dimless_vars.w,
                                                          new_indexes)
        first_derivative.T[ :, :, i] = calculate_dx_patch(i,
                                                          delta_dx[i],
                                                          neigh[i],
                                                          dimless_vars.T,
                                                          new_indexes)

    return first_derivative

def calculate_derivatives_database(args,
                                   read_stencils,
                                   snapshots_path,
                                   dimless_vars,
                                   coors,
                                   new_indexes,
                                   neigbours,
                                   rank):

    """calculation of the first derivative of the snapshots"""

    hdf5_derivatives_database_file = args.snapshots_path[0] + "whole_database_derivatives.hdf5"

    if os.path.isfile(hdf5_derivatives_database_file) and args.read_processed_data_hdf5:
        if rank == 0:
            print "--------------------------------------"
            print "Reading derivatives of database   "
            print "--------------------------------------"

        first_derivative = read_derivatives_database(hdf5_derivatives_database_file, new_indexes)

    else:
        if rank == 0:
            print "--------------------------------------"
            print "Calculating derivatives of database   "
            print "--------------------------------------"

        first_derivative = calculate_1_derivatives(read_stencils,
                                                   snapshots_path,
                                                   dimless_vars,
                                                   coors,
                                                   new_indexes,
                                                   neigbours,
                                                   rank)

        write_derivatives_database(hdf5_derivatives_database_file,
                                   coors,
                                   new_indexes,
                                   first_derivative)

    return first_derivative

def write_derivatives_database(hdf5_derivatives_database_file,
                               coors,
                               new_indexes,
                               first_derivative):

    """write derivatives of the database in a hdf5 format"""

    my_file = h5py.File(hdf5_derivatives_database_file, "w", driver='mpio', comm=MPI.COMM_WORLD)

    first_derivative.save_derivatives_in_hdf5_file(my_file,
                                                   coors.coors_global_dim_rounded.shape[0],
                                                   new_indexes.global_indices_inner,
                                                   new_indexes.local_indices_inner)

    #
    # grc = file.create_group("first_derivatives")
    #
    # Tgrc = grc.create_dataset("T", (coors.coors_global_dim_rounded.shape[0],
    # dimless_vars.T.shape[1], 3) , dtype=np.float64)
    # Pgrc = grc.create_dataset("P", (coors.coors_global_dim_rounded.shape[0],
    #  dimless_vars.P.shape[1], 3) , dtype=np.float64)
    # ugrc = grc.create_dataset("u", (coors.coors_global_dim_rounded.shape[0],
    # dimless_vars.u.shape[1], 3) , dtype=np.float64)
    # vgrc = grc.create_dataset("v", (coors.coors_global_dim_rounded.shape[0],
    # dimless_vars.v.shape[1], 3) , dtype = np.float64)
    # wgrc = grc.create_dataset("w", (coors.coors_global_dim_rounded.shape[0],
    # dimless_vars.w.shape[1], 3) , dtype = np.float64)
    #
    # Tgrc[new_indexes.global_indices_inner]= first_derivative.T[new_indexes.local_indices_inner]
    # Pgrc[new_indexes.global_indices_inner]= first_derivative.P[new_indexes.local_indices_inner]
    #
    # ugrc[new_indexes.global_indices_inner]= first_derivative.u[new_indexes.local_indices_inner]
    # vgrc[new_indexes.global_indices_inner]= first_derivative.v[new_indexes.local_indices_inner]
    # wgrc[new_indexes.global_indices_inner]= first_derivative.w[new_indexes.local_indices_inner]
    #

    my_file.close()

def read_derivatives_database(hdf5_derivatives_database_file, new_indexes):

    """read derivatives of the database in hdf5 format"""

    my_file = h5py.File(hdf5_derivatives_database_file, "r", driver='mpio', comm=MPI.COMM_WORLD)

    first_derivative = Vectors()

    first_derivative.read_derivatives_in_hdf5_file(my_file, new_indexes.global_indices_ghosh_cells)

    my_file.close()

    return first_derivative


def calculate_1_2_derivatives(read_stencils,
                              snapshots_path,
                              eigenvectors,
                              coors,
                              new_indexes, neigbours,
                              rank):

    """full procedure to calculate the first and second derivative"""

    neigh = np.array([[neigbours._I, neigbours.I_],
                      [neigbours._J, neigbours.J_],
                      [neigbours._K, neigbours.K_]])

    first_derivative = Vectors()
    first_derivative.P = np.zeros((eigenvectors.P.shape[0],
                                   eigenvectors.P.shape[1],
                                   3),
                                  dtype=np.float64)
    first_derivative.T = np.zeros((eigenvectors.T.shape[0],
                                   eigenvectors.T.shape[1],
                                   3),
                                  dtype=np.float64)
    first_derivative.u = np.zeros((eigenvectors.u.shape[0],
                                   eigenvectors.u.shape[1],
                                   3),
                                  dtype=np.float64)
    first_derivative.v = np.zeros((eigenvectors.v.shape[0],
                                   eigenvectors.v.shape[1],
                                   3),
                                  dtype=np.float64)
    first_derivative.w = np.zeros((eigenvectors.w.shape[0],
                                   eigenvectors.w.shape[1],
                                   3),
                                  dtype=np.float64)

    namef = snapshots_path + "first_derivatives_stencil.hdf5"
    if read_stencils and os.path.isfile(namef):
        if rank == 0:
            print "---------------------------------------------"
            print "     Reading stencil of the first derivatives"
            print "---------------------------------------------"
        delta_dx = np.zeros((3, coors.coors_dim_rounded.shape[0], 3), dtype=np.float64)
        fhdf = h5py.File(namef, "r", driver='mpio', comm=MPI.COMM_WORLD)
        delta_dx[:, new_indexes.local_indices_ghosh_cells_minus_2, :] = \
            fhdf["first_derivatives_stencil"][:, new_indexes.global_indices_ghosh_cells_minus_2, :]
        fhdf.close()
    else:
        if rank == 0:
            print "--------------------------------------"
            print "Calculating stencil of the derivatives"
            print "      First derivatives stencil"
            print "--------------------------------------"
        delta_dx = calculate_stencil_coeficients_dx_patch(coors.coors_dimless_rounded,
                                                          neigh,
                                                          new_indexes)
        fhdf = h5py.File(namef, "w", driver='mpio', comm=MPI.COMM_WORLD)
        dersten = fhdf.create_dataset("first_derivatives_stencil",
                                      (3,
                                       coors.coors_global_dim_rounded.shape[0],
                                       3))
        dersten[:, new_indexes.global_indices_inner, :] = \
            delta_dx[:, new_indexes.local_indices_inner, :]
        fhdf.close()


    second_derivative = Vectors()
    second_derivative.P = np.zeros((eigenvectors.P.shape[0],
                                    eigenvectors.P.shape[1],
                                    3,
                                    3),
                                   dtype=np.float64)
    second_derivative.T = np.zeros((eigenvectors.T.shape[0],
                                    eigenvectors.T.shape[1],
                                    3,
                                    3),
                                   dtype=np.float64)
    second_derivative.u = np.zeros((eigenvectors.u.shape[0],
                                    eigenvectors.u.shape[1],
                                    3,
                                    3),
                                   dtype=np.float64)
    second_derivative.v = np.zeros((eigenvectors.v.shape[0],
                                    eigenvectors.v.shape[1],
                                    3,
                                    3),
                                   dtype=np.float64)
    second_derivative.w = np.zeros((eigenvectors.w.shape[0],
                                    eigenvectors.w.shape[1],
                                    3,
                                    3),
                                   dtype=np.float64)

    nameff = snapshots_path+"second_derivatives_stencil.hdf5"
    if read_stencils and os.path.isfile(nameff):
        if rank == 0:
            print "---------------------------------------------"
            print "    Reading stencil of the second derivatives"
            print "---------------------------------------------"
        delta_dxdx = np.zeros((3, coors.coors_dim_rounded.shape[0], 5), dtype=np.float64)
        fhdf = h5py.File(nameff, "r", driver='mpio', comm=MPI.COMM_WORLD)
        delta_dxdx[:, new_indexes.local_indices_inner, :] = \
            fhdf["second_derivatives_stencil"][:, new_indexes.global_indices_inner, :]
        fhdf.close()
    else:
        if rank == 0:
            print "--------------------------------------"
            print "Calculating stencil of the derivatives"
            print "      Second derivatives stencil"
            print "--------------------------------------"
        delta_dxdx = calculate_stencil_coeficients_dxdx_patch(coors.coors_dimless_rounded,
                                                              neigh,
                                                              new_indexes)
        fhdf = h5py.File(nameff, "w", driver='mpio', comm=MPI.COMM_WORLD)
        dersten = fhdf.create_dataset("second_derivatives_stencil",
                                      (3,
                                       coors.coors_global_dim_rounded.shape[0],
                                       5))
        dersten[:, new_indexes.global_indices_inner, :] = \
            delta_dxdx[:, new_indexes.local_indices_inner, :]
        fhdf.close()

  #   first_derivative.P = np.zeros((eigenvectors.P.shape[0], 7, 3), dtype=np.float64)
  #   second_derivative.P = np.zeros((eigenvectors.P.shape[0], 7, 3, 3), dtype=np.float64)
  #   eigenvectors.P = np.zeros((eigenvectors.P.shape[0], 7), dtype=np.float64)
  #
  #   maxx= np.max(coors.coors_global_dimless_rounded[:,0])
  #   eigenvectors.P[:,0] = coors.coors_dimless_rounded[:,0]
  #   eigenvectors.P[:,1] = coors.coors_dimless_rounded[:,1]
  #   eigenvectors.P[:,2] = coors.coors_dimless_rounded[:,0]*coors.coors_dimless_rounded[:,0]
  #   eigenvectors.P[:,3] = coors.coors_dimless_rounded[:,0]*coors.coors_dimless_rounded[:,1]
  #   eigenvectors.P[:,4] = coors.coors_dimless_rounded[:,1]*coors.coors_dimless_rounded[:,1]
  #   eigenvectors.P[:,5] = np.cos(coors.coors_dimless_rounded[:,0]/maxx*2.0*np.pi)
  #   eigenvectors.P[:,6] = np.sin(coors.coors_dimless_rounded[:,0]/maxx*2.0*np.pi)
  #  # eigenvectors.P[:, 1] = coors.coors_dimless_rounded[:, 1] * coors.coors_dimless_rounded[:, 1]
  # #  eigenvectors.P[:, 2] = coors.coors_dimless_rounded[:, 3] * coors.coors_dimless_rounded[:, 3]

    if rank == 0:
        print "--------------------------------------"
        print "Calculating 1st and 2nd derivatives"
        print "--------------------------------------"

    for i in range(3):

        first_derivative.P[ :, :, i] = calculate_dx_patch(i,
                                                          delta_dx[i],
                                                          neigh[i],
                                                          eigenvectors.P,
                                                          new_indexes)
        first_derivative.u[ :, :, i] = calculate_dx_patch(i,
                                                          delta_dx[i],
                                                          neigh[i],
                                                          eigenvectors.u,
                                                          new_indexes)
        first_derivative.v[ :, :, i] = calculate_dx_patch(i,
                                                          delta_dx[i],
                                                          neigh[i],
                                                          eigenvectors.v,
                                                          new_indexes)
        first_derivative.w[ :, :, i] = calculate_dx_patch(i,
                                                          delta_dx[i],
                                                          neigh[i],
                                                          eigenvectors.w,
                                                          new_indexes)
        first_derivative.T[ :, :, i] = calculate_dx_patch(i,
                                                          delta_dx[i],
                                                          neigh[i],
                                                          eigenvectors.T,
                                                          new_indexes)

        second_derivative.P[:, :, i, i] = calculate_dxdx_patch(i,
                                                               delta_dxdx[i],
                                                               neigh[i],
                                                               eigenvectors.P,
                                                               new_indexes)
        second_derivative.T[:, :, i, i] = calculate_dxdx_patch(i,
                                                               delta_dxdx[i],
                                                               neigh[i],
                                                               eigenvectors.T,
                                                               new_indexes)
        second_derivative.u[:, :, i, i] = calculate_dxdx_patch(i,
                                                               delta_dxdx[i],
                                                               neigh[i],
                                                               eigenvectors.u,
                                                               new_indexes)
        second_derivative.v[:, :, i, i] = calculate_dxdx_patch(i,
                                                               delta_dxdx[i],
                                                               neigh[i],
                                                               eigenvectors.v,
                                                               new_indexes)
        second_derivative.w[:, :, i, i] = calculate_dxdx_patch(i,
                                                               delta_dxdx[i],
                                                               neigh[i],
                                                               eigenvectors.w,
                                                               new_indexes)

        for j in range(3):
            if j != i:
                second_derivative.P[:, :, i, j] = calculate_dx_patch(j,
                                                                     delta_dx[j],
                                                                     neigh[j],
                                                                     first_derivative.P[ :, :, i],
                                                                     new_indexes)
                second_derivative.T[:, :, i, j] = calculate_dx_patch(j,
                                                                     delta_dx[j],
                                                                     neigh[j],
                                                                     first_derivative.T[ :, :, i],
                                                                     new_indexes)
                second_derivative.u[:, :, i, j] = calculate_dx_patch(j,
                                                                     delta_dx[j],
                                                                     neigh[j],
                                                                     first_derivative.u[ :, :, i],
                                                                     new_indexes)
                second_derivative.v[:, :, i, j] = calculate_dx_patch(j,
                                                                     delta_dx[j],
                                                                     neigh[j],
                                                                     first_derivative.v[ :, :, i],
                                                                     new_indexes)
                second_derivative.w[:, :, i, j] = calculate_dx_patch(j,
                                                                     delta_dx[j],
                                                                     neigh[j],
                                                                     first_derivative.w[ :, :, i],
                                                                     new_indexes)

    if rank == 0:
        print "-----------------------------------------------------"
        print "Calculation of 1st and 2nd derivatives is READY"
        print "-----------------------------------------------------"

    return first_derivative, second_derivative


def calculate_derivatives_eigenvectors(read_stencils,
                                       snapshots_path,
                                       eigenvectors,
                                       coors,
                                       new_indexes, neigbours,
                                       rank):

    """calculates the derivatives of the pods"""
    if rank == 0:
        print "--------------------------------------"
        print "Calculating derivatives eigenvectors  "
        print "      "
        print "--------------------------------------"
    first_derivative, second_derivative = \
        calculate_1_2_derivatives(read_stencils,
                                  snapshots_path,
                                  eigenvectors,
                                  coors,
                                  new_indexes, neigbours,
                                  rank)


    return [first_derivative, second_derivative]


def fft_2_derivative(data, direction, coors, size):

    """calculates second derivative of pods by the fft procedure"""

    if size != 1:
        exit()


    uniquex = np.unique(coors.coors_dimless_rounded[:, 0])
    uniquey = np.unique(coors.coors_dimless_rounded[:, 1])
    uniquez = np.unique(coors.coors_dimless_rounded[:, 2])

    mesh_xx, mesh_yy, mesh_zz = np.meshgrid(uniquex, uniquey, uniquez, indexing='ij')

    structured_vals = np.zeros(mesh_xx.shape)

    counter = 0
    for i in range(uniquex.shape[0]):
        for j in range(uniquey.shape[0]):
            for k in range(uniquez.shape[0]):

                structured_vals[i, j, k] = data[counter]
                counter = counter+ 1

    fftdata = np.fft.rfftn(structured_vals)

    auxx = np.fft.fftfreq(fftdata.shape[0], np.max(uniquex) - np.min(uniquex))
    auxy = np.fft.fftfreq(fftdata.shape[1], np.max(uniquey) - np.min(uniquey))
    auxz = np.fft.fftfreq(fftdata.shape[2], np.max(uniquez) - np.min(uniquez))

    der = np.zeros(fftdata.shape, dtype=np.complex_)

    if direction == 0:
        aux = auxx
        for i in range(fftdata.shape[0]):
            der[i, :, :] = -fftdata[i, :, :] * 4.0 * np.pi ** 2 * aux[i] ** 2 \
                           * structured_vals.shape[0] ** 2
    if direction == 1:
        aux = auxy
        for i in range(fftdata.shape[1]):
            der[:, i, :] = -fftdata[:, i, :] * 4.0 * np.pi ** 2 * aux[i] ** 2 \
                           * structured_vals.shape[1] ** 2

    if direction == 2:
        aux = auxz
        for i in range(fftdata.shape[2]):
            der[:, :, i] = -fftdata[:, :, i] * 4.0 * np.pi ** 2 * aux[i] ** 2 \
                           * structured_vals.shape[2] ** 2

    derp = np.fft.irfftn(der)

    data_der = np.zeros(data.shape[0])

    counter = 0
    for i in range(uniquex.shape[0]):
        for j in range(uniquey.shape[0]):
            for k in range(uniquez.shape[0]):

                data_der[counter] = np.real(derp[i, j, k])
                counter = counter+ 1

    return data_der
