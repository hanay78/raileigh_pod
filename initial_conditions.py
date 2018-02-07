
import numpy as np
from mpi4py import MPI
import copy
import os.path

from vectors import Vectors
from vel_plot import plot_velocity_field_files
from physical_conditions import v0, p0, T0, T1, Tup_dimless, Tdown_dimless, ymax_dimless, P00, RR #, epsilon_sovolev
from info import save_txt_parallel

def initial_values_patch(theta, utheta, dtheta, dutheta, new_indexes, epsilon_sovolev, comm, size, rank):

    a_0_local=np.zeros(utheta.shape[1])
    for i in range(utheta.shape[1]):
        a_0_local[i]=np.dot(theta[new_indexes.local_indices_inner, 0], utheta[new_indexes.local_indices_inner, i])
        for j in range(3):
            a_0_local[i]+=epsilon_sovolev*np.dot(dtheta[new_indexes.local_indices_inner, 0, j], dutheta[new_indexes.local_indices_inner, i, j])

    a_0=np.zeros(a_0_local.shape, dtype=np.float64)
    comm.Allreduce([a_0_local, MPI.DOUBLE], [a_0, MPI.DOUBLE], op=MPI.SUM)

    return a_0

def all_values_patch(theta, utheta, dtheta, dutheta, new_indexes, epsilon_sovolev, comm, size, rank):

    a_0_local=np.zeros((theta.shape[1], utheta.shape[1]))
    for n in range(theta.shape[1]):
        for i in range(utheta.shape[1]):
            a_0_local[n][i]=np.dot(theta[new_indexes.local_indices_inner, n], utheta[new_indexes.local_indices_inner, i])
            for j in range(3):
                a_0_local[n][i]+=epsilon_sovolev*np.dot(dtheta[new_indexes.local_indices_inner, n, j], dutheta[new_indexes.local_indices_inner, i, j])

    a_0=np.zeros(a_0_local.shape, dtype=np.float64)
    comm.Allreduce([a_0_local, MPI.DOUBLE], [a_0, MPI.DOUBLE], op=MPI.SUM)

    return a_0

def initial_value_coefficients_patch(first_derivative_database,
                                     first_derivative_eigenvectors,
                                     dimless_homogeneous_vars,
                                     eigenvectors,
                                     new_indexes,
                                     ts,
                                     path,
                                     epsilon_sovolev,
                                     comm, size, rank):


    if rank==0:
        print "--------------------------------------"
        print "Calculating initial value  "
        print "     of coefficients       "
        print "--------------------------------------"

    a0 = Vectors()

    a0.P = initial_values_patch(dimless_homogeneous_vars.P,
                                eigenvectors.P,
                                first_derivative_database.P,
                                first_derivative_eigenvectors.P,
                                new_indexes,
                                epsilon_sovolev,
                                comm, size, rank)
    a0.T = initial_values_patch(dimless_homogeneous_vars.T,
                                eigenvectors.T,
                                first_derivative_database.T,
                                first_derivative_eigenvectors.T,
                                new_indexes,
                                epsilon_sovolev,
                                comm, size, rank)
    a0.u = initial_values_patch(dimless_homogeneous_vars.u,
                                eigenvectors.u,
                                first_derivative_database.u,
                                first_derivative_eigenvectors.u,
                                new_indexes,
                                epsilon_sovolev,
                                comm, size, rank)
    a0.v = initial_values_patch(dimless_homogeneous_vars.v,
                                eigenvectors.v,
                                first_derivative_database.v,
                                first_derivative_eigenvectors.v,
                                new_indexes,
                                epsilon_sovolev,
                                comm, size, rank)
    a0.w = initial_values_patch(dimless_homogeneous_vars.w,
                                eigenvectors.w,
                                first_derivative_database.w,
                                first_derivative_eigenvectors.w,
                                new_indexes,
                                epsilon_sovolev,
                                comm, size, rank)

    a0.u = a0.u + a0.v + a0.w

    a0.v = []
    a0.w = []

    a0s = Vectors()

    a0s.P = all_values_patch(dimless_homogeneous_vars.P,
                             eigenvectors.P,
                             first_derivative_database.P,
                             first_derivative_eigenvectors.P,
                             new_indexes,
                             epsilon_sovolev,
                             comm, size, rank)
    a0s.T = all_values_patch(dimless_homogeneous_vars.T,
                             eigenvectors.T,
                             first_derivative_database.T,
                             first_derivative_eigenvectors.T,
                             new_indexes,
                             epsilon_sovolev,
                             comm, size, rank)
    a0s.u = all_values_patch(dimless_homogeneous_vars.u,
                             eigenvectors.u,
                             first_derivative_database.u,
                             first_derivative_eigenvectors.u,
                             new_indexes,
                             epsilon_sovolev,
                             comm, size, rank)
    a0s.v = all_values_patch(dimless_homogeneous_vars.v,
                             eigenvectors.v,
                             first_derivative_database.v,
                             first_derivative_eigenvectors.v,
                             new_indexes,
                             epsilon_sovolev,
                             comm, size, rank)
    a0s.w = all_values_patch(dimless_homogeneous_vars.w,
                             eigenvectors.w,
                             first_derivative_database.w,
                             first_derivative_eigenvectors.w,
                             new_indexes,
                             epsilon_sovolev,
                             comm, size, rank)

    a0s.u = a0s.u + a0s.v + a0s.w

    a0s.v = []
    a0s.w = []

    if rank==0:

        aa0s = copy.deepcopy(a0s)
        snaps = ts - ts[0]

        aa0s.P = np.vstack((snaps, aa0s.P.transpose())).transpose()
        aa0s.T = np.vstack((snaps, aa0s.T.transpose())).transpose()
        aa0s.u = np.vstack((snaps, aa0s.u.transpose())).transpose()

        np.savetxt(path + 'aP0s.dat', aa0s.P)
        np.savetxt(path + 'aT0s.dat', aa0s.T)
        np.savetxt(path + 'au0s.dat', aa0s.u)

    return [a0, a0s]

def reproduce_initial_conditions_patch(eigenvectors,
                                     a0,
                                     coors,
                                     path,
                                     dimless_vars, #dimenionless but non homogeneous
                                     new_indexes,
                                     snapshots_path,
                                     comm, size, rank):

    if rank==0:
        print "--------------------------------------"
        print " Reproducing initial conditions       "
        print "      "
        print "--------------------------------------"

    ini_fields = Vectors()
    ini_fields.dot_lumped_separated(eigenvectors, a0)

    #ini_fields_dim = Vectors()
    ini_fields_dim_intermediate = copy.deepcopy(ini_fields)

    ini_fields_dim_intermediate.make_non_homogeneous_bc(coors, Tup_dimless, Tdown_dimless, ymax_dimless, P00, RR)
#    ini_fields_dim_intermediate.non_homogeneous_bc()

    aux = dimless_vars.difference_dataset_integration(0, ini_fields_dim_intermediate)

    error_aux = Vectors()
    error_aux.dot(aux, aux, new_indexes.local_indices_inner)

    errorl2 = Vectors()
    errorl2.zeros_shape(1, 1, 1, 1, 1)
    errorl2.reduce(error_aux, comm, size, rank)

    errorl3 = np.zeros(5, dtype=np.float64)
    if rank==0:
        errorl3[0] = np.sum(np.sqrt(errorl2.P))
        errorl3[1] = np.sum(np.sqrt(errorl2.u))
        errorl3[2] = np.sum(np.sqrt(errorl2.v))
        errorl3[3] = np.sum(np.sqrt(errorl2.w))
        errorl3[4] = np.sum(np.sqrt(errorl2.T))

    comm.Bcast([errorl3, MPI.DOUBLE], root=0)

    if rank==0:
        print "------------------------------------"
        print "Errors in the reproduction of the initial conditions"
        print "Error in T: %f"%(errorl3[4] )
        print "Error in V: %f"%(np.sqrt(errorl3[1]**2+errorl3[2]**2+errorl3[3]**2))
        print "Error in P: %f"%(errorl3[0])
        print "------------------------------------"

        f=open(path+'info_errors_initial_conditions.txt', 'w')

        kk1a= "Errors in the reproduction of the initial conditions\n"
        kk1b= "Error in T: %f\n"%(errorl3[4] )
        kk1c= "Error in V: %f\n"%(np.sqrt(errorl3[1]**2+errorl3[2]**2+errorl3[3]**2))
        kk1d= "Error in P: %f\n"%(errorl3[0])

        f.write(kk1a)
        f.write(kk1b)
        f.write(kk1c)
        f.write(kk1d)

        f.close()

    ini_fields_dim_intermediate.make_dimensional(v0, T0, T1, p0)

    ini_fields_dim = Vectors()

    ini_fields_dim.P = np.vstack([ini_fields_dim_intermediate.P[new_indexes.local_indices_inner].transpose(), coors.coors_dim_rounded[new_indexes.local_indices_inner,:].transpose()]).transpose()
    ini_fields_dim.u = np.vstack([ini_fields_dim_intermediate.u[new_indexes.local_indices_inner].transpose(), coors.coors_dim_rounded[new_indexes.local_indices_inner,:].transpose()]).transpose()
    ini_fields_dim.v = np.vstack([ini_fields_dim_intermediate.v[new_indexes.local_indices_inner].transpose(), coors.coors_dim_rounded[new_indexes.local_indices_inner,:].transpose()]).transpose()
    ini_fields_dim.w = np.vstack([ini_fields_dim_intermediate.w[new_indexes.local_indices_inner].transpose(), coors.coors_dim_rounded[new_indexes.local_indices_inner,:].transpose()]).transpose()
    ini_fields_dim.T = np.vstack([ini_fields_dim_intermediate.T[new_indexes.local_indices_inner].transpose(), coors.coors_dim_rounded[new_indexes.local_indices_inner,:].transpose()]).transpose()

    for p in range(size):

        if rank == p:
            if rank == 0:
                ft=open(path+'T_ini.dat','wb')
                np.savetxt(ft, ini_fields_dim.T)
                ft.close()
                fp=open(path+'P_ini.dat','wb')
                np.savetxt(fp, ini_fields_dim.P)
                fp.close()
                fu=open(path+'U_ini.dat','wb')
                np.savetxt(fu, ini_fields_dim.u)
                fu.close()
                fv=open(path+'V_ini.dat','wb')
                np.savetxt(fv, ini_fields_dim.v)
                fv.close()
                fw=open(path+'W_ini.dat','wb')
                np.savetxt(fw, ini_fields_dim.w)
                fw.close()
            else:
                ft=open(path+'T_ini.dat','ab')
                np.savetxt(ft, ini_fields_dim.T)
                ft.close()
                fp=open(path+'P_ini.dat','ab')
                np.savetxt(fp, ini_fields_dim.P)
                fp.close()
                fu=open(path+'U_ini.dat','ab')
                np.savetxt(fu, ini_fields_dim.u)
                fu.close()
                fv=open(path+'V_ini.dat','ab')
                np.savetxt(fv, ini_fields_dim.v)
                fv.close()
                fw=open(path+'W_ini.dat','ab')
                np.savetxt(fw, ini_fields_dim.w)
                fw.close()
        comm.Barrier()

    argumentop='python ./mode_plotter.py --file_pod P_ini.dat --path ' + path[:-1] + ' --name ini_P --pod'
    argumentot='python ./mode_plotter.py --file_pod T_ini.dat --path ' + path[:-1] + ' --name ini_tau --pod'
    argumentou='python ./mode_plotter.py --file_pod U_ini.dat --path ' + path[:-1] + ' --name ini_u --pod'
    argumentov='python ./mode_plotter.py --file_pod V_ini.dat --path ' + path[:-1] + ' --name ini_v --pod'
    argumentow='python ./mode_plotter.py --file_pod W_ini.dat --path ' + path[:-1] + ' --name ini_w --pod'

    argumentos = [argumentop, argumentot, argumentou, argumentov, argumentow]
    comm.Barrier()

    for i in range(len(argumentos)):
        n=i%size
        if rank==n:
            os.system(argumentos[i])

    # os.system(argumentop)
    # os.system(argumentou)
    # os.system(argumentov)
    # os.system(argumentow)
    # os.system(argumentot)

    if rank==len(argumentos)%size:
        plot_velocity_field_files('Vels_reproduced_ini', path, path+'U_ini.dat', path+'V_ini.dat', snapshots_path)

    aux.make_error_dimensional(v0, T0, T1, p0)

    aux_fields_dim = Vectors()

    aux_fields_dim.P = np.vstack([aux.P[new_indexes.local_indices_inner].transpose(),
                                  coors.coors_dim_rounded[new_indexes.local_indices_inner, :].transpose()]).transpose()
    aux_fields_dim.u = np.vstack([aux.u[new_indexes.local_indices_inner].transpose(),
                                  coors.coors_dim_rounded[new_indexes.local_indices_inner, :].transpose()]).transpose()
    aux_fields_dim.v = np.vstack([aux.v[new_indexes.local_indices_inner].transpose(),
                                  coors.coors_dim_rounded[new_indexes.local_indices_inner, :].transpose()]).transpose()
    aux_fields_dim.w = np.vstack([aux.w[new_indexes.local_indices_inner].transpose(),
                                  coors.coors_dim_rounded[new_indexes.local_indices_inner, :].transpose()]).transpose()
    aux_fields_dim.T = np.vstack([aux.T[new_indexes.local_indices_inner].transpose(),
                                  coors.coors_dim_rounded[new_indexes.local_indices_inner, :].transpose()]).transpose()

    save_txt_parallel(path+'error_initial_P.dat', aux_fields_dim.P, comm, rank, size)
    save_txt_parallel(path+'error_initial_u.dat', aux_fields_dim.u, comm, rank, size)
    save_txt_parallel(path+'error_initial_v.dat', aux_fields_dim.v, comm, rank, size)
    save_txt_parallel(path+'error_initial_w.dat', aux_fields_dim.w, comm, rank, size)
    save_txt_parallel(path+'error_initial_T.dat', aux_fields_dim.T, comm, rank, size)

    argumentop='python ./mode_plotter.py --file_pod error_initial_P.dat --path ' + path[:-1] + ' --name error_ini_P --pod'
    argumentot='python ./mode_plotter.py --file_pod error_initial_T.dat --path ' + path[:-1] + ' --name error_ini_tau --pod'
    argumentou='python ./mode_plotter.py --file_pod error_initial_u.dat --path ' + path[:-1] + ' --name error_ini_u --pod'
    argumentov='python ./mode_plotter.py --file_pod error_initial_u.dat --path ' + path[:-1] + ' --name error_ini_v --pod'
    argumentow='python ./mode_plotter.py --file_pod error_initial_u.dat --path ' + path[:-1] + ' --name error_ini_w --pod'

    argumentos = [argumentop, argumentot, argumentou, argumentov, argumentow]
    comm.Barrier()

    for i in range(len(argumentos)):
        n=i%size
        if rank==n:
            os.system(argumentos[i])


