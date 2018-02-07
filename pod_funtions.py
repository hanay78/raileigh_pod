
import numpy as np

from auxiliary_pod_calculator import calculate_pod_simple, calculate_pod_simple_accuracy
from auxiliary_classes import Layers
from vectors import Vectors



def pod_calculator_lumped_separated_sovolev(first_derivative_database,
                                            dimless_vars,
                                            number_pod, accuracy, fixed_number_of_pods,
                                            new_indexes,
                                            epsilon_sovolev,
                                            path,
                                            high_frequency_fraction,
                                            comm, size, rank):

    """Function that calculates the pods of all three variables
    considering a fixed number of pods or a given accuracy."""


    if rank == 0:
        print "--------------------------------------"
        print "Calculating PODS "
        print "      "
        print "--------------------------------------"

    auxiliar = Vectors()
    auxiliar.P = dimless_vars.P[new_indexes.local_indices_inner]
    auxiliar.T = dimless_vars.T[new_indexes.local_indices_inner]
    auxiliar.u = dimless_vars.u[new_indexes.local_indices_inner]
    auxiliar.v = dimless_vars.v[new_indexes.local_indices_inner]
    auxiliar.w = dimless_vars.w[new_indexes.local_indices_inner]

    sqrt_epsilon_sovolev = np.sqrt(epsilon_sovolev)
    for i in range(3):

        auxiliar.u = np.vstack((auxiliar.u,
                                sqrt_epsilon_sovolev *
                                first_derivative_database.u[new_indexes.local_indices_inner, :, i]))
        auxiliar.v = np.vstack((auxiliar.v,
                                sqrt_epsilon_sovolev *
                                first_derivative_database.v[new_indexes.local_indices_inner, :, i]))
        auxiliar.w = np.vstack((auxiliar.w,
                                sqrt_epsilon_sovolev *
                                first_derivative_database.w[new_indexes.local_indices_inner, :, i]))
        auxiliar.T = np.vstack((auxiliar.T,
                                sqrt_epsilon_sovolev *
                                first_derivative_database.T[new_indexes.local_indices_inner, :, i]))
        auxiliar.P = np.vstack((auxiliar.P,
                                sqrt_epsilon_sovolev *
                                first_derivative_database.P[new_indexes.local_indices_inner, :, i]))



    vaux = auxiliar.u
    vaux = np.vstack((vaux, auxiliar.v))
    vaux = np.vstack((vaux, auxiliar.w))

    vectors_with_ghost = Vectors()
    vectors_with_ghost.P = dimless_vars.P
    vectors_with_ghost.T = dimless_vars.T

    vaux_ghost = dimless_vars.u
    vaux_ghost = np.vstack((vaux_ghost, dimless_vars.v))
    vaux_ghost = np.vstack((vaux_ghost, dimless_vars.w))

    layer = Layers()



    if fixed_number_of_pods:
        (l_vels, u_vels, umaxerror) = calculate_pod_simple(vaux,
                                                           number_pod,
                                                           comm, size, rank)
        (l_press, u_press, pmaxerror) = calculate_pod_simple(auxiliar.P,
                                                             number_pod,
                                                             comm, size, rank)
        (ltheta, utheta, tmaxerror) = calculate_pod_simple(auxiliar.T,
                                                           number_pod,
                                                           comm, size, rank)
    else:
        if rank == 0:
            print "----------------------"
            print "Calculating POD of P"
            print "----------------------"
        (l_press, u_press, n_p, n_p1) = calculate_pod_simple_accuracy(vectors_with_ghost.P,
                                                                      auxiliar.P,
                                                                      accuracy,
                                                                      high_frequency_fraction,
                                                                      comm, size, rank)
        if rank == 0:
            print "----------------------"
            print "Calculating POD of V"
            print "----------------------"
        (l_vels, u_vels, n_v, n_v1) = calculate_pod_simple_accuracy(vaux_ghost,
                                                                    vaux,
                                                                    accuracy,
                                                                    high_frequency_fraction,
                                                                    comm, size, rank)
        if rank == 0:
            print "----------------------"
            print "Calculating POD of T"
            print "----------------------"
        (ltheta, utheta, n_t, n_t1) = calculate_pod_simple_accuracy(vectors_with_ghost.T,
                                                                    auxiliar.T,
                                                                    accuracy,
                                                                    high_frequency_fraction,
                                                                    comm, size, rank)
        layer.set(n_p, n_p1, n_v, n_v1, n_t, n_t1)




    eigenvectors = Vectors()

    eigenvectors.P = np.zeros((dimless_vars.P.shape[0], u_press.shape[1]))
    eigenvectors.T = np.zeros((dimless_vars.T.shape[0], utheta.shape[1]))
    eigenvectors.u = np.zeros((dimless_vars.u.shape[0], u_vels.shape[1]))
    eigenvectors.v = np.zeros((dimless_vars.u.shape[0], u_vels.shape[1]))
    eigenvectors.w = np.zeros((dimless_vars.u.shape[0], u_vels.shape[1]))

    eigenvectors.u[:] = u_vels[0:dimless_vars.u.shape[0]]
    eigenvectors.v[:] = u_vels[dimless_vars.u.shape[0]:2*dimless_vars.u.shape[0]]
    eigenvectors.w[:] = u_vels[2*dimless_vars.u.shape[0]:]

    # eigenvectors.u[new_indexes.local_indices_inner] =
    # u_vels[0:new_indexes.local_indices_inner.shape[0]]
    # eigenvectors.v[new_indexes.local_indices_inner] =
    # u_vels[4*new_indexes.local_indices_inner.shape[0]:5*new_indexes.local_indices_inner.shape[0]]
    # eigenvectors.w[new_indexes.local_indices_inner] =
    # u_vels[8*new_indexes.local_indices_inner.shape[0]:9*new_indexes.local_indices_inner.shape[0]]

    eigenvectors.P[:] = u_press[:]
    eigenvectors.T[:] = utheta[:]

    # eigenvectors.P[new_indexes.local_indices_inner] =
    # u_press[0:new_indexes.local_indices_inner.shape[0]]
    # eigenvectors.T[new_indexes.local_indices_inner] =
    # utheta[0:new_indexes.local_indices_inner.shape[0]]

    eigenvalues = Vectors()
    eigenvalues.set(l_press, l_vels, l_vels, l_vels, ltheta)

    if rank == 0:
        if fixed_number_of_pods:
            print "------------------------------------"
            print "With %d Pods per variable"%(number_pod)
            print "Meassure of error"
            print "Pressure: %f"%(pmaxerror)
            print "Velocity: %f"%(umaxerror)
            print "Temperature: %f"%(tmaxerror)
            print "------------------------------------"

            writefile = open(path+'info_pods.txt', 'w')

            kk1a = "With %d Pods per variable\n"%(number_pod)
            kk1b = "Meassure of error\n"
            kk1c = "Pressure: %f\n"%(pmaxerror)
            kk1d = "Velocity: %f\n"%(umaxerror)
            kk1e = "Temperature: %f\n"%(tmaxerror)

            writefile.write(kk1a)
            writefile.write(kk1b)
            writefile.write(kk1c)
            writefile.write(kk1d)
            writefile.write(kk1e)

            writefile.close()

        else:
            print "------------------------------------"
            print "For maximum error: %f"%(accuracy)
            print "Are necessary the following number of modes per variable:"
            print ""
            print "Pressure: %d"%(n_p)
            print "Velocity: %d"%(n_v)
            print "Temperature: %d"%(n_t)
            print ""
            print "POD error calulation with 1/%d original mistake"%(high_frequency_fraction)
            print "For maximum error: %f"%(accuracy/100.)
            print "Are necessary the following number of modes per variable:"
            print ""
            print "Pressure: %d"%(n_p1)
            print "Velocity: %d"%(n_v1)
            print "Temperature: %d"%(n_t1)
            print "------------------------------------"

            writefile = open(path+'info_pods.txt', 'w')

            kk1a = "For maximum error: %f\n"%(accuracy)
            kk1b = "Are necessary the following number of modes per variable:\n"
            kk1c = "Pressure: %d\n"%(n_p)
            kk1d = "Velocity: %d\n"%(n_v)
            kk1e = "Temperature: %d\n"%(n_t)
            kk1ff = "POD error calulation with 1/%d original mistake\n"%(high_frequency_fraction)
            kk1f = "Pressure: %d\n"%(n_p1)
            kk1g = "Velocity: %d\n"%(n_v1)
            kk1h = "Temperature: %d\n"%(n_t1)

            writefile.write(kk1a)
            writefile.write(kk1b)
            writefile.write(kk1c)
            writefile.write(kk1d)
            writefile.write(kk1e)
            writefile.write(kk1ff)
            writefile.write(kk1f)
            writefile.write(kk1g)
            writefile.write(kk1h)

            writefile.close()

    return eigenvalues, eigenvectors, layer






#
# def pod_calculator_lumped_separated(dimless_vars, #Pres, Vels, Temp,
#                                     number_pod, accuracy, fixed_number_of_pods,
#                                     comm, size, rank):
#
#
#     vels_aux = np.vstack((dimless_vars.u, dimless_vars.v))
#     vels = np.vstack((vels_aux, dimless_vars.w))
#     #velsP = np.vstack((vels, dimless_vars.P))
#
#
#     # (l_vels, u_vels) = calculate_pod_simple(velsP, number_pod, comm, size, rank)
#
#     layer = Layers()
#     if fixed_number_of_pods:
#         (l_vels, u_vels, umaxerror) = calculate_pod_simple(vels,
#                                                            number_pod,
#                                                            comm, size, rank)
#         (l_press, u_press, pmaxerror) = calculate_pod_simple(dimless_vars.P,
#                                                              number_pod,
#                                                              comm, size, rank)
#         (ltheta, utheta, tmaxerror) = calculate_pod_simple(dimless_vars.T,
#                                                            number_pod,
#                                                            comm, size, rank)
#     else:
#         (l_vels, u_vels, n_v, n_v1) = calculate_pod_simple_accuracy(vels,
#                                                                     accuracy,
#                                                                     comm, size, rank)
#         (l_press, u_press, n_p, n_p1) = calculate_pod_simple_accuracy(dimless_vars.P,
#                                                                       accuracy,
#                                                                       comm, size, rank)
#         (ltheta, utheta, n_t, n_t1) = calculate_pod_simple_accuracy(dimless_vars.T,
#                                                                     accuracy,
#                                                                     comm, size, rank)
#         layer.set(n_p, n_p1, n_v, n_v1, n_t, n_t1)
#
#
#     #eigenvalues = Vectors()
#     eigenvectors = Vectors()
#
#     # eigenvectors.set(u_vels[3*dimless_vars.u.shape[0]:],
#     #                  u_vels[0:dimless_vars.u.shape[0]],
#     #                  u_vels[dimless_vars.u.shape[0]:2*dimless_vars.u.shape[0]],
#     #                  u_vels[2*dimless_vars.u.shape[0]:3*dimless_vars.u.shape[0]],
#     #                  utheta)
#     eigenvectors.set(u_press,
#                      u_vels[0:dimless_vars.u.shape[0]],
#                      u_vels[dimless_vars.u.shape[0]:2*dimless_vars.u.shape[0]],
#                      u_vels[2*dimless_vars.u.shape[0]:],
#                      utheta)
#     eigenvalues = Vectors()
#
#     eigenvalues.set(l_press, l_vels, l_vels, l_vels, ltheta)
#
#     if fixed_number_of_pods:
#         print "------------------------------------"
#         print "With %d Pods per variable"%(number_pod)
#         print "Meassure of error"
#         print "Pressure: %f"%(pmaxerror)
#         print "Velocity: %f"%(umaxerror)
#         print "Temperature: %f"%(tmaxerror)
#         print "------------------------------------"
#
#         filewrite = open('info_pods.txt', 'w')
#
#         kk1a = "With %d Pods per variable\n"%(number_pod)
#         kk1b = "Meassure of error\n"
#         kk1c = "Pressure: %f\n"%(pmaxerror)
#         kk1d = "Velocity: %f\n"%(umaxerror)
#         kk1e = "Temperature: %f\n"%(tmaxerror)
#
#         filewrite.write(kk1a)
#         filewrite.write(kk1b)
#         filewrite.write(kk1c)
#         filewrite.write(kk1d)
#         filewrite.write(kk1e)
#
#         filewrite.close()
#
#     else:
#         print "------------------------------------"
#         print "For maximum error: %f"%(accuracy)
#         print "Are necessary the following number of modes per variable:"
#         print "Pressure: %d"%(n_p)
#         print "Velocity: %d"%(n_v)
#         print "Temperature: %d"%(n_t)
#
#         print "POD error calulation with 1/100 original mistake"
#         print "For maximum error: %f"%(accuracy/100.)
#         print "Are necessary the following number of modes per variable:"
#         print "Pressure: %d"%(n_p1)
#         print "Velocity: %d"%(n_v1)
#         print "Temperature: %d"%(n_t1)
#         print "------------------------------------"
#         filewrite = open('info_pods.txt', 'w')
#
#         kk1a = "For maximum error: %f\n"%(accuracy)
#         kk1b = "Are necessary the following number of modes per variable:\n"
#         kk1c = "Pressure: %d\n"%(n_p)
#         kk1d = "Velocity: %d\n"%(n_v)
#         kk1e = "Temperature: %d\n"%(n_t)
#         kk1ff = "POD error calulation with 1/100 original mistake"
#         kk1f = "Pressure: %d\n"%(n_p1)
#         kk1g = "Velocity: %d\n"%(n_v1)
#         kk1h = "Temperature: %d\n"%(n_t1)
#
#         filewrite.write(kk1a)
#         filewrite.write(kk1b)
#         filewrite.write(kk1c)
#         filewrite.write(kk1d)
#         filewrite.write(kk1e)
#         filewrite.write(kk1ff)
#         filewrite.write(kk1f)
#         filewrite.write(kk1g)
#         filewrite.write(kk1h)
#
#         filewrite.close()
#
#     return eigenvalues, eigenvectors, layer

