import numpy as np

from mpi4py import MPI
from physical_conditions import Tup_dimless, Tdown_dimless, ymax_dimless, P00, RR
from vectors import Vectors
from rhs_python_version import solve_linear_system, non_linear_cn, pressure_eq_lineal_sovolev, error_non_linear_cn, stiffness, linear_system_components
import copy
from info import save_snapshot, plot_snapshots
from scipy.optimize import fsolve
import time

def save_amplitudes(path, tes, a_solutions):

    a_local=copy.deepcopy(a_solutions)
    a_local.transpose()
    a_local.vstack_vector_lumped(tes)
    a_local.transpose()

    np.savetxt(path + 'aPs.dat',  a_local.P)
    np.savetxt(path + 'aus.dat',  a_local.u)
    np.savetxt(path + 'aTs.dat',  a_local.T)

# def tau_min(taus, params):
#
#     [tau0, tau1]=taus
#     print [tau0, tau1]
#
#     [tensor, a, aPn]=params
#
# #    aint = integration_vectors(a)
#
#     product_phys_local(tau0, tau1, tensor)
#     Press_eq(tau0, tau1, tensor, a)
#
#     aux = (a.P-aPn)*(a.P-aPn)
#
#     return np.sqrt(np.sum(aux))

def time_step(t, ts, dt_0):
    dt = dt_0
    snapshot = False
    if (t < np.max(ts)):
        next_t = ts[np.argmax(ts > t)]
        if (next_t - t <= dt):
            dt = next_t - t
            snapshot = True
    return dt, snapshot

def integrate(t, snaps, dt_0, tensor, a, dtm, accuracy_integration):

    dtmax, snapshot = time_step(t, snaps, dt_0)

    dt=copy.deepcopy(dtmax)

    if dt>dtm*2.:
        dt=dtm*2.

    while True:
        Y, Z = linear_system_components(tensor, a, dt)

        da = solve_linear_system(Y, Z, tensor.alpha.shape[0])

        da_lin=copy.deepcopy(da)
        da_non_lin=copy.deepcopy(da)

        sols=fsolve(non_linear_cn, np.append(da_lin.u, da_lin.T), xtol=1.0e-8, args=(tensor, Y, Z, dt))
        da_non_lin.u=sols[:da.u.shape[0]]
        da_non_lin.T=sols[da.u.shape[0]:]

        a_new=copy.deepcopy(a)
        a_new.add_lumped_separated(da_non_lin)

        error = error_non_linear_cn(a, a_new, tensor, dt)
        # [errora, errorb] = error_non_linear_CN(sols[:a_lin.u.shape[0]],sols[a_lin.u.shape[0]:] , tensor, a.u, a.T, dt)

        errort=np.append(error.u, error.T)
        as_new=np.append(a_new.u, a_new.T)
        errorrel=np.zeros(errort.shape[0], dtype=np.float64)
        for i in range(errort.shape[0]):
            if np.absolute(as_new[i])>1.e-12:
                errorrel[i]=errort[i]/as_new[i]

        maxerror=np.max(np.absolute(errorrel))
    # print errora
    # print errorb
        if maxerror <  accuracy_integration:
            break
        else:
            dt=dt*0.5

    # a.u=sols[:a_lin.u.shape[0]]
    # a.T=sols[a_lin.u.shape[0]:]

    a = a_new

    # print "Max error integration step: %f" % (maxerror)
    # print "Time Step: %f"%(dt)
    if dtmax != dt:
        snapshot=False

    t += dt
    dtm=dt
    return (a, dt, snapshot, t, dtm)

def calculate_error_int_high_modes(path, tes, a_solutions, layer):

    ep=np.zeros(a_solutions.P.shape[0])
    eu=np.zeros(a_solutions.u.shape[0])
    et=np.zeros(a_solutions.u.shape[0])

    high_modes_error = Vectors()

    for i in range(a_solutions.u.shape[0]):
        ep[i]=100.0*np.sqrt(np.dot(a_solutions.P[i, layer.n_p:], a_solutions.P[i, layer.n_p:]))\
              /np.sqrt(np.dot(a_solutions.P[i, :], a_solutions.P[i, :])+1.e-20)
        eu[i]=100.0*np.sqrt(np.dot(a_solutions.u[i, layer.n_v:], a_solutions.u[i, layer.n_v:]))\
              /np.sqrt(np.dot(a_solutions.u[i, :], a_solutions.u[i, :])+1.e-20)
        et[i]=100.0*np.sqrt(np.dot(a_solutions.T[i, layer.n_t:], a_solutions.T[i, layer.n_t:]))\
              /np.sqrt(np.dot(a_solutions.T[i, :], a_solutions.T[i, :])+1.e-20)

    # high_modes_error.P = ep
    # high_modes_error.T = et
    # high_modes_error.u = eu

    eep = np.vstack([tes, ep]).transpose()
    eeu = np.vstack([tes, eu]).transpose()
    eet = np.vstack([tes, et]).transpose()

    np.savetxt(path+"error_int_high_modes_p.dat", eep)
    np.savetxt(path+"error_int_high_modes_u.dat", eeu)
    np.savetxt(path+"error_int_high_modes_t.dat", eet)


def loop_integration_py_patch(tensor,  #dimenionless
                                a0,  #dimenionless
                                a0s,  # dimenionless
                                ts,  #dimenionless
                                coors,  #dimenionless
                                eigenvectors,  #dimenionless
                                dimless_vars,  #dimenionless but non homogeneous
                                layer,
                                error,
                                error_modes,
                                path,
                                snapshots_path,
                                accuracy_integration,
                                new_indexes,
                                save_snapshots,
                                comm, rank, size):

    a=copy.deepcopy(a0)

    dt_0 = (ts[-1] - ts[0]) / 1000.

    snaps=ts-ts[0]

    t = np.float64(0.0)
    tes = np.zeros(1, dtype=np.float64)
    counter = 0

    errorl = Vectors()
    errorl.zeros(1)

    a_solutions = Vectors()
    a_solutions.zeros()

    a_solutions.append_own_class_lumped_separated(a0)

    a_snapshots =  Vectors()
    a_snapshots.append_own_class_lumped_separated(a0)

    dtm=1.0e-10

    if rank==0:
        stiffness(tensor)

    snapshots_files = []
    snapshots_names = []
    while (True):


        (a, dt, snapshot, t, dtm)=integrate(t, snaps, dt_0, tensor, a, dtm,  accuracy_integration)

        a.P = pressure_eq_lineal_sovolev(a, tensor)


        #t += dt
        if rank==0:
            print "Integration processor %i for %f" %(rank, t)
        counter += 1

        # keep track of the amplitudes in each integration step
        a_solutions.vstack_own_class_lumped_separated(a)

        tes=np.append(tes, t)

        # plot and print
        if snapshot:

            ind = np.where(snaps == t)[0][0]

            ### error in terms of the amplitudes
            error_modes.T += np.sum(np.fabs(a.T[:layer.n_t] - a0s.T[ind][:layer.n_t]))
            error_modes.u += np.sum(np.fabs(a.u[:layer.n_v] - a0s.u[ind][:layer.n_v]))
            a_snapshots.vstack_own_class_lumped_separated(a)

            ### generate snapshos of the solutions
            if save_snapshots:
                save_snapshot(ind, a, eigenvectors, coors, path, snapshots_path, new_indexes, snapshots_files, snapshots_names, comm, rank, size)

            ### calulate values of the variables (geometrically)
            fields_local_dimensionless = Vectors()
            fields_local_dimensionless.dot_lumped_separated(eigenvectors, a)
            fields_local_dimensionless.make_non_homogeneous_bc(coors, Tup_dimless, Tdown_dimless, ymax_dimless, P00, RR)

            ### calculate differences between solutions and input snapshots
            aux = dimless_vars.difference_dataset_integration(ind, fields_local_dimensionless)

                ### calculate total difference
            error_aux = Vectors()
            error_aux.dot(aux, aux, new_indexes.local_indices_inner)

            errorl.append_own_class(error_aux)

        if (t >= snaps[-1]):
            break


    ### plot the snapshots collected
    if save_snapshots:
        plot_snapshots(path, snapshots_path, snapshots_files, snapshots_names, comm, rank, size)

    ### save collected amplitudes of a's during the calculation
    if rank == 0:
        save_amplitudes(path, tes, a_solutions)
        calculate_error_int_high_modes(path, tes, a_solutions, layer)

    ### collecte the error asociated to geometrical variables
    errorl2 = Vectors()
    errorl2.zeros_shape(errorl.P.shape, errorl.u.shape, errorl.v.shape, errorl.w.shape, errorl.T.shape)
    errorl2.reduce(errorl, comm, size, rank)

    if rank==0:
        error[0] = np.sum(np.sqrt(errorl2.P))
        error[1] = np.sum(np.sqrt(errorl2.u))
        error[2] = np.sum(np.sqrt(errorl2.v))
        error[3] = np.sum(np.sqrt(errorl2.w))
        error[4] = np.sum(np.sqrt(errorl2.T))

    comm.Bcast([error, MPI.DOUBLE], root=0)


# def loop_integration_py(tensor,#dimenionless
#                         a,#dimenionless
#                         ts,#dimenionless
#                         coors,#dimenionless
#                         eigenvectors,#dimenionless
#                         dimless_vars, #dimenionless but non homogeneous
#                         layer,
#                         error,
#                         path,
#                         accuracy_integration,
#                         comm, rank, size):
#
#     dt_0 = (ts[-1] - ts[0]) / 1000.
#
#     snaps=ts-ts[0]
#
#     t = np.float64(0.0)
#     tes = np.zeros(1, dtype=np.float64)
#     counter = 0
#
#     errorl = vectors()
#     errorl.zeros(1)
#
#     a_solutions = vectors()
#     a_solutions.zeros()
#
#     a_solutions.append_own_class_lumped_separated(a)
#
#     flagini = True
#
#     dtm=1.0e-10
#
#     stiffness(tensor)
#
#     while (True):
#
# #         dt, snapshot = time_step(t, snaps, dt_0)
# #
# #         da = solve_linear_system(tensor, a, dt)
# #
# #         a_lin=copy.deepcopy(a)
# #         a_lin.add_lumped_separated(da)
# #
# # #        errora= np.zeros(a_lin.u.shape[0], dtype=np.float64)
# # #        errorb= np.zeros(a_lin.T.shape[0], dtype=np.float64)
# #
# #         sols=fsolve(non_linear_CN, np.append(a_lin.u, a_lin.T), xtol=1.0e-4, args=(tensor, a.u, a.T, dt))
# #
# #         [errora, errorb] =error_non_linear_CN(sols[:a_lin.u.shape[0]],sols[a_lin.u.shape[0]:] , tensor, a.u, a.T, dt)
# #
# #         print errora
# #         print errorb
# #
# #         a.u=sols[:a_lin.u.shape[0]]
# #         a.T=sols[a_lin.u.shape[0]:]
#
#         (a, dt, snapshot)=integrate(t, snaps, dt_0, tensor, a, dtm,  accuracy_integration)
#
#         a.P = pressure_eq_lineal(a, tensor)
#
#
#         t += dt
#         print "Integration for %f" %(t)
#         counter += 1
#
#         # keep track of the amplitudes in each integration step
#         a_solutions.vstack_own_class_lumped_separated(a)
#
#         tes=np.append(tes, t)
#
#         # plot and print
#         if snapshot:
#
#             ind = np.where(snaps == t)[0][0]
#
#             save_snapshot(ind, a, eigenvectors, coors, path, comm, rank, size)
#
#             fields_local_dimensionless = vectors()
#             fields_local_dimensionless.dot_lumped_separated(eigenvectors, a)
#             fields_local_dimensionless.make_non_homogeneous_bc(coors, Tup_dimless, Tdown_dimless, ymax_dimless, P00, RR)
#
#             aux = dimless_vars.difference_dataset_integration(ind, fields_local_dimensionless)
#
#             error_aux = vectors()
#             error_aux.dot(aux, aux)
#
#             errorl.append_own_class(error_aux)
#
#         if (t >= snaps[-1]):
#             break
#
#         dtm=dt
# # save amplitudes of a's during the calculation
#     if rank == 0:
#         save_amplitudes(path, tes, a_solutions)
#         calculate_error_int_high_modes(path, tes, a_solutions, layer)
#
#     errorl2 = vectors()
#     errorl2.zeros_shape(errorl.P.shape, errorl.u.shape, errorl.v.shape, errorl.w.shape, errorl.T.shape)
#     errorl2.reduce(errorl, comm, size, rank)
#
#     if rank==0:
#         error[0] = np.sum(np.sqrt(errorl2.P))
#         error[1] = np.sum(np.sqrt(errorl2.u))
#         error[2] = np.sum(np.sqrt(errorl2.v))
#         error[3] = np.sum(np.sqrt(errorl2.w))
#         error[4] = np.sum(np.sqrt(errorl2.T))
#
#     comm.Bcast([error, MPI.DOUBLE], root=0)

def integration_patch(tensor,  #dimenionless
                a0,  #dimenionless
                a0s,  # dimenionless
                ts,  #dimenionless
                coors,  #dimenionless
                eigenvectors,  #dimenionless
                dimless_vars,  #dimensionless  non-homogeneous
                layer,
                  path,
                  snapshots_path,
                  comm, rank, size,
                  python_flag,
                  accuracy_integration,
                  new_indexes,
                  save_snapshots
                  ):
    start_time = time.time()

    if rank == 0:
        print "--------------------------------------------"
        print "-----------  NEW CALCULATION  --------------"
        print "--------------------------------------------"

    error = np.zeros(5, dtype=np.float64)

    error_modes = Vectors()
    error_modes.zeros(1)

    if python_flag:
        loop_integration_py_patch(tensor, #dimenionless
                            a0, #dimenionless
                            a0s, #dimenionless
                            ts, #dimenionless
                            coors, #dimenionless
                            eigenvectors, #dimenionless
                            dimless_vars, #dimensionless  non-homogeneous
                            layer,
                            error,
                            error_modes,
                            path,
                            snapshots_path,
                            accuracy_integration,
                            new_indexes,
                            save_snapshots,
                            comm, rank, size)
    else:
        print "kk"

    if rank == 0:
        print("--- %s seconds ---" % (time.time() - start_time))

    if rank == 0:
        print "Total Error P: ", error[0]
        print "Total Error u: ", error[1]
        print "Total Error v: ", error[2]
        print "Total Error w: ", error[3]
        print "Total Error T: ", error[4]
        print "Total Error: %g" % (error[0] + error[1] + error[2] + error[3] + error[4])

    total_error = error[0] + error[1] + error[2] + error[3] + error[4]

    if rank == 0:
        np.savetxt(path + 'total_error.dat', [total_error, ])

#    return error[0] + error[1] + error[2] + error[3] + error[4]
#     return error[1] + error[2] + error[3] + error[4]

#    return error_modes.T * np.max(np.fabs(a0s.u)) / np.max(np.fabs(a0s.T)) + error_modes.u
    return error_modes.T

#
# def integration(tensor, #dimenionless
#                 a0, #dimenionless
#                 ts, #dimenionless
#                 coors, #dimenionless
#                 eigenvectors, #dimenionless
#                 dimless_vars, #dimensionless  non-homogeneous
#                 layer,
#                 path,
#                 comm, rank, size,
#                 python_flag,
#                 accuracy_integration
#                 ):
#     start_time = time.time()
#
#     if rank == 0:
#         print "--------------------------------------------"
#         print "-----------  NEW CALCULATION  --------------"
#         print "--------------------------------------------"
#
#     a = copy.deepcopy(a0)
#
#     error = np.zeros(5, dtype=np.float64)
#
#     if python_flag:
#         loop_integration_py(tensor, #dimenionless
#                             a, #dimenionless
#                             ts, #dimenionless
#                             coors, #dimenionless
#                             eigenvectors, #dimenionless
#                             dimless_vars, #dimensionless  non-homogeneous
#                             layer,
#                             error,
#                             path,
#                             accuracy_integration,
#                             comm, rank, size)
#     else:
#         print "kk"
#
#     if rank == 0:
#         print("--- %s seconds ---" % (time.time() - start_time))
#
#     if rank == 0:
#         print "Total Error P: ", error[0]
#         print "Total Error u: ", error[1]
#         print "Total Error v: ", error[2]
#         print "Total Error w: ", error[3]
#         print "Total Error T: ", error[4]
#         print "Total Error: %g" % (error[0] + error[1] + error[2] + error[3] + error[4])
#
#     total_error = error[0] + error[1] + error[2] + error[3] + error[4]
#
#     np.savetxt(path + 'total_error.dat', [total_error, ])
#
#     return error[0] + error[1] + error[2] + error[3] + error[4]
