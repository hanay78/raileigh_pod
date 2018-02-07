#!/usr/bin/env python
import sys
import os
import numpy as np
from mpi4py import MPI
from vectors import Vectors
import copy
from physical_conditions import L0, v0,T0, T1, Tup_dimless, Tdown_dimless, ymax_dimless, P00, RR, p0
from vel_plot import plot_velocity_field, plot_velocity_field_files
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
import h5py
import subprocess
from mode_plotter import modeploter_function

def save_snapshot(ind, a, eigenvectors, coors, path, snapshots_path, new_indexes, snapshots_files, snapshots_names, comm, rank, size):

    fields_local = Vectors()

    fields_local.dot_lumped_separated(eigenvectors, a)

    fields_local.make_non_homogeneous_bc(coors, Tup_dimless, Tdown_dimless, ymax_dimless, P00, RR)

    fields_local_dim = copy.deepcopy(fields_local)
    fields_local_dim.make_dimensional(v0, T0, T1, p0)

    #coors_local_dim = coors * L0

    sols_snap = np.vstack((fields_local_dim.P[new_indexes.local_indices_inner], fields_local_dim.u[new_indexes.local_indices_inner]))
    sols_snap = np.vstack((sols_snap, fields_local_dim.v[new_indexes.local_indices_inner]))
    sols_snap = np.vstack((sols_snap, fields_local_dim.w[new_indexes.local_indices_inner]))
    sols_snap = np.vstack((sols_snap, fields_local_dim.T[new_indexes.local_indices_inner]))
    sols_snap = np.vstack((sols_snap, coors.coors_dim_rounded[new_indexes.local_indices_inner, 0]))
    sols_snap = np.vstack((sols_snap, coors.coors_dim_rounded[new_indexes.local_indices_inner, 1]))
    sols_snap = np.vstack((sols_snap, coors.coors_dim_rounded[new_indexes.local_indices_inner, 2]))
    sols_snap = sols_snap.transpose()

    q = "%05i" % (ind)

    nasol_h = path + "Solutions_" + q + ".h5"
    npsol_h = "Solutions_" + q + ".h5"
    namel_h = "Solutions_" + q

    comm.Barrier()
    snap_h = h5py.File(nasol_h, "w", driver='mpio', comm=MPI.COMM_WORLD)
    all = snap_h.create_dataset("All", (coors.coors_global_dimless_rounded.shape[0], sols_snap.shape[1]), dtype=np.float64)
    all[new_indexes.global_indices_inner,:] = sols_snap[:,:]
    snap_h.close()
    comm.Barrier()

    snapshots_files.append(npsol_h)
    snapshots_names.append(namel_h)

#     if rank==0:
#
#         modeploter_function(path, npsol_h, namel_h)
#         plot_velocity_field(path, npsol_h, snapshots_path, True)
#
# #    return fields_local
#     comm.Barrier()

def plot_snapshots(path, snapshots_path, snapshots_files, snapshots_names, comm, rank, size):

    for i in range(len(snapshots_files)):
        if rank==i%size:
            modeploter_function(path, snapshots_files[i], snapshots_names[i])
            plot_velocity_field(path, snapshots_files[i], snapshots_path, True)


def pod_info(coors_dim, eigenvectors, path, comm, rank, size):

    pod = Vectors()

    pod.P = np.vstack([eigenvectors.P.transpose(), coors_dim.transpose()]).transpose()
    pod.u = np.vstack([eigenvectors.u.transpose(), coors_dim.transpose()]).transpose()
    pod.v = np.vstack([eigenvectors.v.transpose(), coors_dim.transpose()]).transpose()
    pod.w = np.vstack([eigenvectors.w.transpose(), coors_dim.transpose()]).transpose()
    pod.T = np.vstack([eigenvectors.T.transpose(), coors_dim.transpose()]).transpose()

#    print "Number of POD obtained: "
#    print "For Pressure: %i" % eigenvalues.P.size
#    print "For U velocity: %i" % eigenvalues.u.size
#    print "For V velocity: %i" % eigenvalues.v.size
#    print "For W velocity: %i" % eigenvalues.w.size
#    print "For Temperature: %i" % eigenvalues.T.size
    print "-----------------------"
    print "--- OUTPUTTING PODS ---"
    print "-----------------------"

    for p in range(size):
        if rank == p:
            if rank == 0:

                fp=open(path+'pod_pi.dat','wb')
                np.savetxt(fp, pod.P)
                fp.close()

                ftau=open(path+'pod_tau.dat','wb')
                np.savetxt(ftau, pod.T)
                ftau.close()

                fu=open(path+'pod_u.dat','wb')
                np.savetxt(fu, pod.u)
                fu.close()

                fv=open(path+'pod_v.dat','wb')
                np.savetxt(fv, pod.v)
                fv.close()

                fw=open(path+'pod_w.dat','wb')
                np.savetxt(fw, pod.w)
                fw.close()

            else:

                fp=open(path+'pod_pi.dat','ab')
                np.savetxt(fp, pod.P)
                fp.close()

                ftau=open(path+'pod_tau.dat','ab')
                np.savetxt(fp, pod.T)
                ftau.close()

                fu=open(path+'pod_u.dat','ab')
                np.savetxt(fp, pod.u)
                fu.close()

                fv=open(path+'pod_v.dat','ab')
                np.savetxt(fp, pod.v)
                fv.close()

                fw=open(path+'pod_w.dat','ab')
                np.savetxt(fp, pod.w)
                fw.close()

        comm.Barrier()

    argumentop='python ./mode_plotter.py --file_pod pod_pi.dat --path ' + path[:-1] + ' --name pod_obtenido_pi --pod'
    argumentot='python ./mode_plotter.py --file_pod pod_tau.dat --path ' + path[:-1] + ' --name pod_obtenido_tau --pod'
    argumentou='python ./mode_plotter.py --file_pod pod_u.dat --path ' + path[:-1] + ' --name pod_obtenido_u --pod'
    argumentov='python ./mode_plotter.py --file_pod pod_v.dat --path ' + path[:-1] + ' --name pod_obtenido_v --pod'
    argumentow='python ./mode_plotter.py --file_pod pod_w.dat --path ' + path[:-1] + ' --name pod_obtenido_w --pod'

    os.system(argumentop)
    os.system(argumentou)
    os.system(argumentov)
    os.system(argumentow)
    os.system(argumentot)

    plot_velocity_field_files('Pods_vectorial_', path, path+'pod_u.dat', path+'pod_v.dat',)

def pod_info_patch(coors, eigenvectors, path, new_indexes, snapshots_path, comm, rank, size):

    pod = Vectors()

    pod.P = np.vstack([eigenvectors.P[new_indexes.local_indices_inner].transpose(), coors.coors_dim_rounded[new_indexes.local_indices_inner].transpose()]).transpose()
    pod.u = np.vstack([eigenvectors.u[new_indexes.local_indices_inner].transpose(), coors.coors_dim_rounded[new_indexes.local_indices_inner].transpose()]).transpose()
    pod.v = np.vstack([eigenvectors.v[new_indexes.local_indices_inner].transpose(), coors.coors_dim_rounded[new_indexes.local_indices_inner].transpose()]).transpose()
    pod.w = np.vstack([eigenvectors.w[new_indexes.local_indices_inner].transpose(), coors.coors_dim_rounded[new_indexes.local_indices_inner].transpose()]).transpose()
    pod.T = np.vstack([eigenvectors.T[new_indexes.local_indices_inner].transpose(), coors.coors_dim_rounded[new_indexes.local_indices_inner].transpose()]).transpose()

#    print "Number of POD obtained: "
#    print "For Pressure: %i" % eigenvalues.P.size
#    print "For U velocity: %i" % eigenvalues.u.size
#    print "For V velocity: %i" % eigenvalues.v.size
#    print "For W velocity: %i" % eigenvalues.w.size
#    print "For Temperature: %i" % eigenvalues.T.size
    if rank==0:
        print "-----------------------"
        print "--- OUTPUTTING PODS ---"
        print "-----------------------"

    for p in range(size):
        if rank == p:
            if rank == 0:

                fp=open(path+'pod_pi.dat','wb')
                np.savetxt(fp, pod.P)
                fp.close()

                ftau=open(path+'pod_tau.dat','wb')
                np.savetxt(ftau, pod.T)
                ftau.close()

                fu=open(path+'pod_u.dat','wb')
                np.savetxt(fu, pod.u)
                fu.close()

                fv=open(path+'pod_v.dat','wb')
                np.savetxt(fv, pod.v)
                fv.close()

                fw=open(path+'pod_w.dat','wb')
                np.savetxt(fw, pod.w)
                fw.close()

            else:

                fp=open(path+'pod_pi.dat','ab')
                np.savetxt(fp, pod.P)
                fp.close()

                ftau=open(path+'pod_tau.dat','ab')
                np.savetxt(ftau, pod.T)
                ftau.close()

                fu=open(path+'pod_u.dat','ab')
                np.savetxt(fu, pod.u)
                fu.close()

                fv=open(path+'pod_v.dat','ab')
                np.savetxt(fv, pod.v)
                fv.close()

                fw=open(path+'pod_w.dat','ab')
                np.savetxt(fw, pod.w)
                fw.close()
        comm.Barrier()
    if rank==0:
        argumentop='python ./mode_plotter.py --file_pod pod_pi.dat --path ' + path[:-1] + ' --name pod_obtenido_pi --pod'
        argumentot='python ./mode_plotter.py --file_pod pod_tau.dat --path ' + path[:-1] + ' --name pod_obtenido_tau --pod'
        argumentou='python ./mode_plotter.py --file_pod pod_u.dat --path ' + path[:-1] + ' --name pod_obtenido_u --pod'
        argumentov='python ./mode_plotter.py --file_pod pod_v.dat --path ' + path[:-1] + ' --name pod_obtenido_v --pod'
        argumentow='python ./mode_plotter.py --file_pod pod_w.dat --path ' + path[:-1] + ' --name pod_obtenido_w --pod'

        os.system(argumentop)
        os.system(argumentou)
        os.system(argumentov)
        os.system(argumentow)
        os.system(argumentot)

        plot_velocity_field_files('Pods_vectorial_', path, path+'pod_u.dat', path+'pod_v.dat', snapshots_path)

def variables_info(coors, dimless_homogeneous_vars, name, path, new_indexes, snapshots_path, comm, rank, size):

    pod = Vectors()

    pod.P = np.vstack([dimless_homogeneous_vars.P[new_indexes.get_local_inner_indices(), :].transpose(),
                       coors.coors_dim_rounded[new_indexes.get_local_inner_indices(), :].transpose()]).transpose()
    pod.u = np.vstack([dimless_homogeneous_vars.u[new_indexes.get_local_inner_indices(), :].transpose(),
                       coors.coors_dim_rounded[new_indexes.get_local_inner_indices(), :].transpose()]).transpose()
    pod.v = np.vstack([dimless_homogeneous_vars.v[new_indexes.get_local_inner_indices(), :].transpose(),
                       coors.coors_dim_rounded[new_indexes.get_local_inner_indices(), :].transpose()]).transpose()
    pod.w = np.vstack([dimless_homogeneous_vars.w[new_indexes.get_local_inner_indices(), :].transpose(),
                       coors.coors_dim_rounded[new_indexes.get_local_inner_indices(), :].transpose()]).transpose()
    pod.T = np.vstack([dimless_homogeneous_vars.T[new_indexes.get_local_inner_indices(), :].transpose(),
                       coors.coors_dim_rounded[new_indexes.get_local_inner_indices(), :].transpose()]).transpose()

#    print "Number of POD obtained: "
#    print "For Pressure: %i" % eigenvalues.P.size
#    print "For U velocity: %i" % eigenvalues.u.size
#    print "For V velocity: %i" % eigenvalues.v.size
#    print "For W velocity: %i" % eigenvalues.w.size
#    print "For Temperature: %i" % eigenvalues.T.size
    if rank==0:
        print "------------------------------------------------"
        print "---            OUTPUTTING VARIABLES          ---"
        print "------------------------------------------------"

    for p in range(size):
        if rank == p:
            if rank == 0:

                fp=open(path+name+'pi.dat','wb')
                np.savetxt(fp, pod.P)
                fp.close()

                ftau=open(path+name+'tau.dat','wb')
                np.savetxt(ftau, pod.T)
                ftau.close()

                fu=open(path+name+'u.dat','wb')
                np.savetxt(fu, pod.u)
                fu.close()

                fv=open(path+name+'v.dat','wb')
                np.savetxt(fv, pod.v)
                fv.close()

                fw=open(path+name+'w.dat','wb')
                np.savetxt(fw, pod.w)
                fw.close()

            else:

                fp=open(path+name+'pi.dat','ab')
                np.savetxt(fp, pod.P)
                fp.close()

                ftau=open(path+name+'tau.dat','ab')
                np.savetxt(ftau, pod.T)
                ftau.close()

                fu=open(path+name+'u.dat','ab')
                np.savetxt(fu, pod.u)
                fu.close()

                fv=open(path+name+'v.dat','ab')
                np.savetxt(fv, pod.v)
                fv.close()

                fw=open(path+name+'w.dat','ab')
                np.savetxt(fw, pod.w)
                fw.close()

        comm.Barrier()

    #modeploter_function(path, file_pod, name, pod_yes=False, dir=2, dx=0.000625):

    files_pod =[name+'pi.dat', name+'tau.dat',name+'u.dat', name+'v.dat', name+'w.dat']
    names =[name+'pi', name+'tau',name+'u', name+'v', name+'w']
    # argumentos.append('python ./mode_plotter.py --file_pod '+name+'pi.dat --path ' + path[:-1] + ' --name '+name+'pi --pod')
    # argumentos.append('python ./mode_plotter.py --file_pod '+name+'tau.dat --path ' + path[:-1] + ' --name '+name+'tau --pod')
    # argumentos.append('python ./mode_plotter.py --file_pod '+name+'u.dat --path ' + path[:-1] + ' --name '+name+'u --pod')
    # argumentos.append('python ./mode_plotter.py --file_pod '+name+'v.dat --path ' + path[:-1] + ' --name '+name+'v --pod')
    # argumentos.append('python ./mode_plotter.py --file_pod '+name+'w.dat --path ' + path[:-1] + ' --name '+name+'w --pod')

    # for p in range(len(files_pod)):
    #     if p%size==rank:
    #         # os.system(argumentos[p])
    #         modeploter_function(path, files_pod[p], names[p], True, False)
    #     # os.system(argumentop)
    #     # os.system(argumentou)
    #     # os.system(argumentov)
    #     # os.system(argumentow)
    #     # os.system(argumentot)
    #
    # if rank==len(files_pod)%size:
    #     plot_velocity_field_files('Vels_database_', path, path+name+'u.dat', path+name+'v.dat', snapshots_path)

    for p in range(len(files_pod)):
        if rank==0:
            # os.system(argumentos[p])
            modeploter_function(path, files_pod[p], names[p], True, False)
            plot_velocity_field_files('Vels_database_', path, path+name+'u.dat', path+name+'v.dat', snapshots_path)
        # os.system(argumentop)
        # os.system(argumentou)
        # os.system(argumentov)
        # os.system(argumentow)
        # os.system(argumentot)




def outputting_and_plotting_derivatives_database(first_derivative,
                                        coors,
                                        new_indexes,
                                        path, comm, rank, size):

    namesf00=[]
    namesf0=[]
    print "-----------------------------------"
    print "Outputting and plotting derivatives"
    print "-----------------------------------"
    for s in range(first_derivative.P.shape[1]):
        for i in range(3):
            fields_plot = np.vstack([first_derivative.P[new_indexes.local_indices_inner,s,i].transpose(),
                                     coors.coors_dim_rounded[new_indexes.local_indices_inner,:].transpose()]).transpose()

            namesf00.append('P_snapshot_'+str(s)+'_dir_'+str(i))
            namesf0.append(namesf00[-1]+'.dat')
            namesf=path+namesf0[-1]

            for p in range(size):
                if rank == p:
                    if rank == 0:
                        ft=open(namesf,'wb')
                        np.savetxt(ft, fields_plot)
                        ft.close()
                    else:
                        ft=open(namesf,'ab')
                        np.savetxt(ft, fields_plot)
                        ft.close()
                comm.Barrier()
            comm.Barrier()
            # if rank==0:
            #     argumento='python ./mode_plotter.py --file_pod '+ namesf0 +' --path ' + path[:-1] + ' --name '+ namesf00 +' --pod'
            #     os.system(argumento)

    for s in range(first_derivative.u.shape[1]):
        for i in range(3):
            fields_plot = np.vstack([first_derivative.u[new_indexes.local_indices_inner,s,i].transpose(),
                                     coors.coors_dim_rounded[new_indexes.local_indices_inner,:].transpose()]).transpose()

            namesf00.append('U_snapshot_'+str(s)+'_dir_'+str(i))
            namesf0.append(namesf00[-1]+'.dat')
            namesf=path+namesf0[-1]
            for p in range(size):
                if rank == p:
                    if rank == 0:
                        ft=open(namesf,'wb')
                        np.savetxt(ft, fields_plot)
                        ft.close()
                    else:
                        ft=open(namesf,'ab')
                        np.savetxt(ft, fields_plot)
                        ft.close()
                comm.Barrier()
            # if rank==0:
            #     argumento='python ./mode_plotter.py --file_pod '+ namesf0 +' --path ' + path[:-1] + ' --name '+ namesf00 +' --pod'
            #     os.system(argumento)


    for s in range(first_derivative.v.shape[1]):
        for i in range(3):
            fields_plot = np.vstack([first_derivative.v[new_indexes.local_indices_inner,s,i].transpose(),
                                     coors.coors_dim_rounded[new_indexes.local_indices_inner,:].transpose()]).transpose()

            namesf00.append('V_snapshot_'+str(s)+'_dir_'+str(i))
            namesf0.append(namesf00[-1]+'.dat')
            namesf=path+namesf0[-1]
            for p in range(size):
                if rank == p:
                    if rank == 0:
                        ft=open(namesf,'wb')
                        np.savetxt(ft, fields_plot)
                        ft.close()
                    else:
                        ft=open(namesf,'ab')
                        np.savetxt(ft, fields_plot)
                        ft.close()
                comm.Barrier()
            # if rank==0:
            #     argumento='python ./mode_plotter.py --file_pod '+ namesf0 +' --path ' + path[:-1] + ' --name '+ namesf00 +' --pod'
            #     os.system(argumento)

    for s in range(first_derivative.w.shape[1]):
        for i in range(3):
            fields_plot = np.vstack([first_derivative.w[new_indexes.local_indices_inner,s,i].transpose(),
                                     coors.coors_dim_rounded[new_indexes.local_indices_inner,:].transpose()]).transpose()

            namesf00.append('W_snapshot_'+str(s)+'_dir_'+str(i))
            namesf0.append(namesf00[-1]+'.dat')
            namesf=path+namesf0[-1]

            for p in range(size):
                if rank == p:
                    if rank == 0:
                        ft=open(namesf,'wb')
                        np.savetxt(ft, fields_plot)
                        ft.close()
                    else:
                        ft=open(namesf,'ab')
                        np.savetxt(ft, fields_plot)
                        ft.close()
                comm.Barrier()
            # if rank==0:
            #     argumento='python ./mode_plotter.py --file_pod '+ namesf0 +' --path ' + path[:-1] + ' --name '+ namesf00 +' --pod'
            #     os.system(argumento)


    for s in range(first_derivative.T.shape[1]):
        for i in range(3):
            fields_plot = np.vstack([first_derivative.T[new_indexes.local_indices_inner,s,i].transpose(),
                                     coors.coors_dim_rounded[new_indexes.local_indices_inner,:].transpose()]).transpose()

            namesf00.append('T_snapshot_'+str(s)+'_dir_'+str(i))
            namesf0.append(namesf00[-1]+'.dat')
            namesf=path+namesf0[-1]

            for p in range(size):
                if rank == p:
                    if rank == 0:
                        ft=open(namesf,'wb')
                        np.savetxt(ft, fields_plot)
                        ft.close()
                    else:
                        ft=open(namesf,'ab')
                        np.savetxt(ft, fields_plot)
                        ft.close()
                comm.Barrier()
            # if rank==0:
            #     argumento='python ./mode_plotter.py --file_pod '+ namesf0 +' --path ' + path[:-1] + ' --name '+ namesf00 +' --pod'
            #     os.system(argumento)

    for p in range(len(namesf0)):
        n=p%size
        if rank==n:
            argumento='python ./mode_plotter.py --file_pod '+ namesf0[p] +' --path ' + path[:-1] + ' --name '+ namesf00[p] +' --pod'
            os.system(argumento)
#
# def outputting_and_plotting_derivatives(eigenvectors_global,
#                                         first_derivative,
#                                         coors_global_dim_rounded,
#                                         path):
#
#     print "Outputting and plotting derivatives"
#     for s in range(eigenvectors_global.P.shape[1]):
#         for i in range(3):
#             fields_plot = np.vstack([first_derivative.P[s,i].transpose(), coors_global_dim_rounded.transpose()]).transpose()
#
#             namesf00='P_mode_'+str(s)+'_dir_'+str(i)
#             namesf0=namesf00+'.dat'
#             namesf=path+namesf0
#
#             ft=open(namesf,'wb')
#             np.savetxt(ft, fields_plot)
#             ft.close()
#
#             argumento='python ./mode_plotter.py --file_pod '+ namesf0 +' --path ' + path[:-1] + ' --name '+ namesf00 +' --pod'
#             os.system(argumento)
#
#     for s in range(eigenvectors_global.u.shape[1]):
#         for i in range(3):
#             fields_plot = np.vstack([first_derivative.u[s,i].transpose(), coors_global_dim_rounded.transpose()]).transpose()
#
#             namesf00='U_mode_'+str(s)+'_dir_'+str(i)
#             namesf0=namesf00+'.dat'
#             namesf=path+namesf0
#
#             ft=open(namesf,'wb')
#             np.savetxt(ft, fields_plot)
#             ft.close()
#
#             argumento='python ./mode_plotter.py --file_pod '+ namesf0 +' --path ' + path[:-1] + ' --name '+ namesf00 +' --pod'
#             os.system(argumento)
#
#
#     for s in range(eigenvectors_global.v.shape[1]):
#         for i in range(3):
#             fields_plot = np.vstack([first_derivative.v[s,i].transpose(), coors_global_dim_rounded.transpose()]).transpose()
#
#             namesf00='V_mode_'+str(s)+'_dir_'+str(i)
#             namesf0=namesf00+'.dat'
#             namesf=path+namesf0
#
#             ft=open(namesf,'wb')
#             np.savetxt(ft, fields_plot)
#             ft.close()
#
#             argumento='python ./mode_plotter.py --file_pod '+ namesf0 +' --path ' + path[:-1] + ' --name '+ namesf00 +' --pod'
#             os.system(argumento)
#
#     for s in range(eigenvectors_global.w.shape[1]):
#         for i in range(3):
#             fields_plot = np.vstack([first_derivative.w[s,i].transpose(), coors_global_dim_rounded.transpose()]).transpose()
#
#             namesf00='W_mode_'+str(s)+'_dir_'+str(i)
#             namesf0=namesf00+'.dat'
#             namesf=path+namesf0
#
#             ft=open(namesf,'wb')
#             np.savetxt(ft, fields_plot)
#             ft.close()
#
#             argumento='python ./mode_plotter.py --file_pod '+ namesf0 +' --path ' + path[:-1] + ' --name '+ namesf00 +' --pod'
#             os.system(argumento)
#
#
#     for s in range(eigenvectors_global.T.shape[1]):
#         for i in range(3):
#             fields_plot = np.vstack([first_derivative.T[s,i].transpose(), coors_global_dim_rounded.transpose()]).transpose()
#
#             namesf00='T_mode_'+str(s)+'_dir_'+str(i)
#             namesf0=namesf00+'.dat'
#             namesf=path+namesf0
#
#             ft=open(namesf,'wb')
#             np.savetxt(ft, fields_plot)
#             ft.close()
#
#             argumento='python ./mode_plotter.py --file_pod '+ namesf0 +' --path ' + path[:-1] + ' --name '+ namesf00 +' --pod'
#             os.system(argumento)

def save_txt_parallel(namesf, fields_plot, comm, rank, size):

    for p in range(size):
        if rank == p:
            if rank == 0:
                ft=open(namesf,'wb')
                np.savetxt(ft, fields_plot)
                ft.close()
            else:
                ft=open(namesf,'ab')
                np.savetxt(ft, fields_plot)
                ft.close()
        comm.Barrier()
    comm.Barrier()

def outputting_and_plotting_derivatives_eigenvectors(
                                        first_derivative, second_derivative,
                                        coors,
                                        new_indexes,
                                        path, comm, rank, size):

    namesf00=[]
    namesf0=[]
    if rank==0:
        print "------------------------------------------------"
        print "Outputting and plotting derivatives eigenvectors"
        print "------------------------------------------------"
    for s in range(first_derivative.P.shape[1]):
        for i in range(3):
            fields_plot = np.vstack([first_derivative.P[new_indexes.local_indices_inner,s,i].transpose(),
                                     coors.coors_dim_rounded[new_indexes.local_indices_inner,:].transpose()]).transpose()

            namesf00.append('eigenvectors_P_'+str(s)+'_dir_'+str(i))
            namesf0.append(namesf00[-1]+'.dat')
            namesf=path+namesf0[-1]
            save_txt_parallel(namesf, fields_plot, comm, rank, size)
            for j in range(3):
                fields_plot = np.vstack([second_derivative.P[new_indexes.local_indices_inner,s,i,j].transpose(),
                                         coors.coors_dim_rounded[new_indexes.local_indices_inner,:].transpose()]).transpose()

                namesf00.append('eigenvectors_P_'+str(s)+'_dir_'+str(i)+str(j))
                namesf0.append(namesf00[-1]+'.dat')
                namesf=path+namesf0[-1]
                save_txt_parallel(namesf, fields_plot, comm, rank, size)

    for s in range(first_derivative.u.shape[1]):
        for i in range(3):
            fields_plot = np.vstack([first_derivative.u[new_indexes.local_indices_inner,s,i].transpose(),
                                     coors.coors_dim_rounded[new_indexes.local_indices_inner,:].transpose()]).transpose()

            namesf00.append('eigenvectors_u_'+str(s)+'_dir_'+str(i))
            namesf0.append(namesf00[-1]+'.dat')
            namesf=path+namesf0[-1]
            save_txt_parallel(namesf, fields_plot, comm, rank, size)
            for j in range(3):
                fields_plot = np.vstack([second_derivative.u[new_indexes.local_indices_inner,s,i,j].transpose(),
                                         coors.coors_dim_rounded[new_indexes.local_indices_inner,:].transpose()]).transpose()

                namesf00.append('eigenvectors_u_'+str(s)+'_dir_'+str(i)+str(j))
                namesf0.append(namesf00[-1]+'.dat')
                namesf=path+namesf0[-1]
                save_txt_parallel(namesf, fields_plot, comm, rank, size)

    for s in range(first_derivative.v.shape[1]):
        for i in range(3):
            fields_plot = np.vstack([first_derivative.v[new_indexes.local_indices_inner, s, i].transpose(),
                                     coors.coors_dim_rounded[new_indexes.local_indices_inner,:].transpose()]).transpose()

            namesf00.append('eigenvectors_v_' + str(s) + '_dir_' + str(i))
            namesf0.append(namesf00[-1] + '.dat')
            namesf = path + namesf0[-1]
            save_txt_parallel(namesf, fields_plot, comm, rank, size)
            for j in range(3):
                fields_plot = np.vstack([second_derivative.v[new_indexes.local_indices_inner, s, i, j].transpose(),
                                         coors.coors_dim_rounded[new_indexes.local_indices_inner,:].transpose()]).transpose()

                namesf00.append('eigenvectors_v_' + str(s) + '_dir_' + str(i) + str(j))
                namesf0.append(namesf00[-1] + '.dat')
                namesf = path + namesf0[-1]
                save_txt_parallel(namesf, fields_plot, comm, rank, size)

    for s in range(first_derivative.w.shape[1]):
        for i in range(3):
            fields_plot = np.vstack([first_derivative.w[new_indexes.local_indices_inner, s, i].transpose(),
                                     coors.coors_dim_rounded[new_indexes.local_indices_inner,:].transpose()]).transpose()

            namesf00.append('eigenvectors_w_' + str(s) + '_dir_' + str(i))
            namesf0.append(namesf00[-1] + '.dat')
            namesf = path + namesf0[-1]
            save_txt_parallel(namesf, fields_plot, comm, rank, size)
            for j in range(3):
                fields_plot = np.vstack([second_derivative.w[new_indexes.local_indices_inner, s, i, j].transpose(),
                                         coors.coors_dim_rounded[new_indexes.local_indices_inner,
                                         :].transpose()]).transpose()

                namesf00.append('eigenvectors_w_' + str(s) + '_dir_' + str(i) + str(j))
                namesf0.append(namesf00[-1] + '.dat')
                namesf = path + namesf0[-1]
                save_txt_parallel(namesf, fields_plot, comm, rank, size)

    for s in range(first_derivative.T.shape[1]):
        for i in range(3):
            fields_plot = np.vstack([first_derivative.T[new_indexes.local_indices_inner, s, i].transpose(),
                                     coors.coors_dim_rounded[new_indexes.local_indices_inner,
                                     :].transpose()]).transpose()

            namesf00.append('eigenvectors_T_' + str(s) + '_dir_' + str(i))
            namesf0.append(namesf00[-1] + '.dat')
            namesf = path + namesf0[-1]
            save_txt_parallel(namesf, fields_plot, comm, rank, size)
            for j in range(3):
                fields_plot = np.vstack([second_derivative.T[new_indexes.local_indices_inner, s, i, j].transpose(),
                                         coors.coors_dim_rounded[new_indexes.local_indices_inner,
                                         :].transpose()]).transpose()

                namesf00.append('eigenvectors_T_' + str(s) + '_dir_' + str(i) + str(j))
                namesf0.append(namesf00[-1] + '.dat')
                namesf = path + namesf0[-1]
                save_txt_parallel(namesf, fields_plot, comm, rank, size)


    for p in range(len(namesf0)):
        n=p%size
        if rank==n:
            argumento='python ./mode_plotter.py --file_pod '+ namesf0[p] +' --path ' + path[:-1] + ' --name '+ namesf00[p] +' --pod'
            os.system(argumento)

def plot_eigenvalues(path, eigenvalues):
    x1 = np.arange(eigenvalues.u.shape[0])
    x2 = np.arange(eigenvalues.T.shape[0])

    plt.figure()
#    plt.set_yscale('log')
    plt.semilogy(x1, eigenvalues.u, 'bo')
    plt.savefig(path+'Eigenvectors_velocity.png')
    plt.close()

    plt.figure()
#    plt.set_yscale('log')
    plt.semilogy(x2, eigenvalues.T, 'ro')
    plt.savefig(path+'Eigenvectors_temperature.png')
    plt.close()

def colineary_analysis(dim_vars, correlation_coef, path, new_indexes):
#    ck=np.vstack((np.ones(dim_vars.T.shape[0]),dim_vars.T.transpose())).transpose()

#    vif_results = [vif(ck, i) for i in range(ck.shape[1])]

    for cf in correlation_coef:

        independent = np.vstack((np.ones(dim_vars.T[new_indexes.get_local_inner_indices()].shape[0]), dim_vars.T[:, 0]))
        independent = np.vstack((independent, dim_vars.T[new_indexes.get_local_inner_indices(), -1]))
        ind_ind = np.array([0, dim_vars.T.shape[1]])
        independent_t = independent.transpose()

        for i in range(1,dim_vars.T.shape[1]-2):
            prueba= np.vstack((independent_t.transpose(), dim_vars.T[new_indexes.get_local_inner_indices(),i])).transpose()
            vif_r = vif(prueba, prueba.shape[1]-1)
            if vif_r>1./(1.-cf):
                continue
            else:
                independent_t=prueba
                ind_ind=np.append(ind_ind, i)
        ind_ind.sort()

        print '----------------------------------------'
        print 'Analysis of colinearity'
        print 'For a correlation coefficient %f'%(cf)
        print 'the non-colinear snapshots are'
        print ind_ind
        print '----------------------------------------'
        np.savetxt(path+'independent_snapschots_correlation_0_'+str(cf)+'.dat', ind_ind)