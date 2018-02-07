#!/usr/bin/env python


from scipy.optimize import minimize

import argparse
import copy

import scipy as sp

from mpi4py import MPI
import numpy as np
import os.path

from info import variables_info, plot_eigenvalues, \
    colineary_analysis, \
    pod_info_patch, \
    outputting_and_plotting_derivatives_database, \
    outputting_and_plotting_derivatives_eigenvectors
from initial_conditions import initial_value_coefficients_patch, reproduce_initial_conditions_patch
from loop_integration import integration_patch
from phis_calculator import product_phis_universal_star_ccm_patch
from physical_conditions import L0, t0, v0, p0, T0, T1,  RR, P00, ymax_dimless, Tup_dimless, Tdown_dimless
#from pod_calculator_star_ccm import pod_calculator_lumped_separated_sovolev
from pod_funtions import pod_calculator_lumped_separated_sovolev
from read_database import provide_database
from derivatives import calculate_derivatives_database, calculate_derivatives_eigenvectors

def calculate_path(args):

    path = './'

    if args.path[0][-1] != '/':
        service = str(args.path[0])
        service += "/"
        path = service

    return path

def bundle_minimization_procedure(epsilon_sovolev,
                                  args,
                                  dimless_homogeneous_vars,
                                  first_derivative_database,
                                  N_pod,
                                  accuracy_ppod,
                                  coors,
                                  new_indexes,
                                  neigbours,
                                  path,
                                  ts,
                                  comm, size, rank):
    if rank==0:
        print "############################"
        print "############################"
        print "############################"
        print "############################"
        print "NEW CALCULATION"
        print "epsilon_sovolev = %f"%(epsilon_sovolev)
        print "############################"
        print "############################"
        print "############################"
        print "############################"

    epsilon_sovolev=np.fabs(epsilon_sovolev)

    if rank==0:
        fp=open(path+'epsilon_sovolev.dat','ab')
        np.savetxt(fp, [epsilon_sovolev,])
        fp.close()

    ### calculate pod ###
    eigenvalues, \
    eigenvectors, \
    layer = pod_calculator_lumped_separated_sovolev(first_derivative_database,
                                                    dimless_homogeneous_vars,
                                                    N_pod, accuracy_ppod,
                                                    args.fixed_number_of_pods,
                                                    new_indexes,
                                                    epsilon_sovolev,
                                                    path,
                                                    args.high_frequency_fraction[0],
                                                    comm, size, rank)

    if args.plot_eigenvalues:
        if rank == 0:
            plot_eigenvalues(path, eigenvalues)

    if args.plot_eigenvectors:
        pod_info_patch(coors, eigenvectors, path, new_indexes, args.snapshots_path[0], comm, rank, size)
        # pod_info(coors_dim_rounded, eigenvectors, path, comm, rank, size)

        ### derivatives eigenvectors

    [first_derivative_eigenvectors, second_derivative_eigenvectors] = calculate_derivatives_eigenvectors(
        args.read_stencils,
        args.snapshots_path[0],
        eigenvectors,
        coors,
        new_indexes, neigbours,
        rank)

    if args.plot_derivative_eigenvectors:
        outputting_and_plotting_derivatives_eigenvectors(
            first_derivative_eigenvectors, second_derivative_eigenvectors,
            coors,
            new_indexes,
            path, comm, rank, size)

        ### calculate matrices for the pod ###
    # tensor, tensor_aux = product_phis_universal_star_ccm_patch(first_derivative_eigenvectors,
    #                                                            second_derivative_eigenvectors,
    #                                                            eigenvectors,
    #                                                            new_indexes,
    #                                                            epsilon_sovolev,
    #                                                            comm,
    #                                                            size,
    #                                                            rank)
    tensor = product_phis_universal_star_ccm_patch(first_derivative_eigenvectors,
                                                   second_derivative_eigenvectors,
                                                   eigenvectors,
                                                   new_indexes,
                                                   epsilon_sovolev,
                                                   comm,
                                                   size,
                                                   rank)

    [a0, a0s] = initial_value_coefficients_patch(first_derivative_database,
                                                 first_derivative_eigenvectors,
                                                 dimless_homogeneous_vars,
                                                 eigenvectors,
                                                 new_indexes,
                                                 ts,
                                                 path,
                                                 epsilon_sovolev,
                                                 comm, size, rank)

    # MY=-np.append(  - Re_inv * np.dot(tensor_aux.B, a0.u)
    #             + RR * np.dot(tensor_aux.C, a0.T)
    #             - np.tensordot(np.tensordot(tensor_aux.Omega, a0.u, axes=([2],[0])), a0.u, axes=([1],[0]))
    #             ,
    #             - Ty * np.dot(tensor_aux.J,a0.u)
    #             - Re_inv * Pr_inv * np.dot(tensor_aux.L, a0.T)
    #             - np.tensordot(np.tensordot(tensor_aux.Kappa, a0.T, axes=([2],[0])), a0.u, axes=([1],[0]))
    #           )
    # MX = np.append( - Re_inv * np.dot(tensor_aux.S, a0.u)
    #                 + RR * np.dot(tensor_aux.G, a0.T)
    #                 - np.tensordot(np.tensordot(tensor_aux.psi + tensor_aux.chi, a0.u, axes=([2],[0])), a0.u, axes=([1],[0]))
    #                 ,
    #                 - Ty * np.dot(tensor_aux.M, a0.u)
    #                 - Re_inv * Pr_inv * np.dot(tensor_aux.N, a0.T)
    #                 - np.tensordot(np.tensordot(tensor_aux.upsilon + tensor_aux.omicron, a0.T, axes=([2],[0])), a0.u, axes=([1],[0]))
    #                 )
    # eeps=np.dot(np.dot(np.linalg.inv(np.dot(MX.transpose(), MX)), MX.transpose()), MY)
    #
    # print "value of epsilon is"
    # print eeps
    # exit()




    ####free memory

    # dimless_homogeneous_vars = None
    # first_derivative_database = None
    first_derivative_eigenvectors = None
    second_derivative_eigenvectors = None

    dimless_homogeneous_vars.make_non_homogeneous_bc(coors, Tup_dimless, Tdown_dimless, ymax_dimless, P00, RR)
    # ### info vars
    if args.reproduce_initial_conditions:
        reproduce_initial_conditions_patch(eigenvectors,
                                           a0,
                                           coors,
                                           path,
                                           dimless_homogeneous_vars,
                                           new_indexes,
                                           args.snapshots_path[0],
                                           comm, size, rank)
    error_process = \
    integration_patch(tensor,  # dimenionless
                      a0,  # dimensionaless
                      a0s, # dimensionaless
                      ts,  # dimensionless
                      coors,  # dimensionless
                      eigenvectors,  # dimensionless
                      dimless_homogeneous_vars,
                      # dimless_vars, #dimensionless  non-homogeneous
                      layer,
                      path,
                      args.snapshots_path[0],
                      comm, rank, size,
                      args.python_version,
                      args.accuracy_integration[0],
                      new_indexes,
                      args.save_snapshots
                      )

    dimless_homogeneous_vars.make_homogeneous_bc(coors, Tup_dimless, Tdown_dimless, ymax_dimless, P00, RR)

    return error_process

if __name__ == '__main__':

    ### initiate the mpi ###
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    N_pod = 10
    accuracy_ppod = 0.001

    ### parser of options ###
    parser = argparse.ArgumentParser()
    parser.add_argument('--accuracy_pod', default=[0.0001], nargs=1, type=np.float64)
    parser.add_argument('--python_version', default=True, action='store_true')
    #parser.add_argument('--filesdb', default=dbses, nargs='+', help='files')
    parser.add_argument('--path', default=['./results'], nargs=1, help='files')
    parser.add_argument('--snapshots_path', default=['./snapshots/'], nargs=1, help='files')
    parser.add_argument('--snapshots_pattern', default=['XYZ*.csv'], nargs=1, help='files')
    parser.add_argument('--read_stencils', default=True, action='store_false')
#    parser.add_argument('--read_stencils', default=False, action='store_true')
    parser.add_argument('--fixed_number_of_pods', default=False, action='store_true')
    parser.add_argument('--accuracy_integration', default=[0.001], nargs=1, type=np.float64)
    parser.add_argument('--number_of_pod', default=[10], nargs=1, type=np.int32)
    parser.add_argument('--colinearity_analysis', default=False, action='store_true')
    #parser.add_argument('--colinearity_analysis', default=True, action='store_true')
    parser.add_argument('--read_processed_data_hdf5', default=True, action='store_false')
    #parser.add_argument('--output_information_dimensional_dataset', default=True, action='store_true')
    parser.add_argument('--output_information_dimensional_dataset', default=False, action='store_true')
    #parser.add_argument('--output_information_dimensionless_dataset', default=True, action='store_true')
    parser.add_argument('--output_information_dimensionless_dataset', default=False, action='store_true')
    #parser.add_argument('--output_information_dimensionless_homogeneous_dataset', default=True, action='store_false')
    parser.add_argument('--output_information_dimensionless_homogeneous_dataset', default=False, action='store_true')
    parser.add_argument('--plot_eigenvectors', default=False, action='store_false')
    #parser.add_argument('--plot_eigenvectors', default=True, action='store_false')
    parser.add_argument('--plot_derivative_eigenvectors', default=False, action='store_false')
    #parser.add_argument('--plot_derivative_eigenvectors', default=True, action='store_false')
    parser.add_argument('--plot_eigenvalues', default=True, action='store_false')
    #parser.add_argument('--plotting_derivatives_database', default=True, action='store_true')
    parser.add_argument('--plotting_derivatives_database', default=False, action='store_true')
    parser.add_argument('--reproduce_initial_conditions', default=False, action='store_true')
    parser.add_argument('--high_frequency_fraction', default=[10.0], nargs=1, type=np.float64)
    parser.add_argument('--save_snapshots', default=False, action='store_true')
    parser.add_argument('--single_calculation', default=False, action='store_true')
    parser.add_argument('--epsilon_sovolev', default=[0.001], nargs=1, type=np.float64)
    ### parse the input line ###
    args_program = parser.parse_args()

    ### calculate path

    path = calculate_path(args_program)

    if args_program.fixed_number_of_pods:
        N_pod = args_program.number_of_pod[0]
    else:
        accuracy_ppod = args_program.accuracy_pod[0]

    ### check path ###
    if not os.path.isdir(args_program.path[0]):
        os.makedirs(args_program.path[0])

    print "-------------------------------------------------"
    print "This is processor %d"%(rank)
    print "-------------------------------------------------"

    comm.Barrier()

    ### read datatable or files ###

    [dim_vars,
     coors,
     neigbours,
     boundaries,
     ts_dim,
     new_indexes] = provide_database(args_program, comm, size, rank)

#colinearity analysis
    if args_program.colinearity_analysis:
        colineary_analysis(dim_vars, [0.99, 0.999, 0.9999], path, new_indexes)

### info vars
    if args_program.output_information_dimensional_dataset:
        variables_info(coors, dim_vars, 'dim_vars_', path, new_indexes, args_program.snapshots_path[0], comm, rank, size)


#### make read values dimensionless ####
    dimless_vars = copy.deepcopy(dim_vars)
    ### Free the memory
    #print "Freeing dimensional variables in processor %i"%(rank)
    dim_vars = None


    dimless_vars.make_dimensionless(v0, T0, T1, p0)


### info vars
    if args_program.output_information_dimensionless_dataset:
        variables_info(coors, dimless_vars, 'dimless_vars_', path, new_indexes, args_program.snapshots_path[0], comm, rank, size)


### make coordinates and time dimensioneless ###
    ts = ts_dim.copy(order='C')
    ts = ts / t0

    coors.make_coordinates_dimensioneless(L0)

### make data homogenaous boundary conditions ###

    dimless_homogeneous_vars = copy.deepcopy(dimless_vars)
    dimless_homogeneous_vars.make_homogeneous_bc(coors, Tup_dimless, Tdown_dimless, ymax_dimless, P00, RR)

    dimless_vars = None
### info vars
    if args_program.output_information_dimensionless_homogeneous_dataset:
        variables_info(coors, dimless_homogeneous_vars, 'dimless_homogeneous_', path, new_indexes, args_program.snapshots_path[0], comm, rank, size)

###calculate derivative database
    first_derivative_database = calculate_derivatives_database(args_program,
                                                               args_program.read_stencils,
                                                               args_program.snapshots_path[0],
                                                               dimless_homogeneous_vars,
                                                               coors,
                                                               new_indexes, neigbours,
                                                               rank)

    if args_program.plotting_derivatives_database:
        outputting_and_plotting_derivatives_database(first_derivative_database,
                                                     coors,
                                                     new_indexes,
                                                     path, comm, rank, size)



    if args_program.single_calculation:
        bundle_minimization_procedure(args_program.epsilon_sovolev[0],
                                      args_program,
                                      dimless_homogeneous_vars,
                                      first_derivative_database,
                                      N_pod,
                                      accuracy_ppod,
                                      coors,
                                      new_indexes,
                                      neigbours,
                                      path,
                                      ts,
                                      comm, size, rank
                                      )
    else:
        args_minimization_bundle = (args_program,
                                    dimless_homogeneous_vars,
                                    first_derivative_database,
                                    N_pod,
                                    accuracy_ppod,
                                    coors,
                                    new_indexes,
                                    neigbours,
                                    path,
                                    ts,
                                    comm, size, rank)

        sp.optimize.minimize_scalar(bundle_minimization_procedure,
                                    args=args_minimization_bundle,
                                    method='bounded',
                                    bounds=(0.0, 0.01))