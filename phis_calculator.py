import numpy as np
from mpi4py import MPI
from auxiliary_classes import Neigbors, Boundary, Matrices, round_coordinates
from vectors import Vectors
import copy
from physical_conditions import St, Re_inv, Pr_inv,  RR, Ty #, epsilon_sovolev
import producto


def calculate_neighbors(c, comm, size, rank):
    if rank==0:
        print "-------------------------------------"
        print "--- CALCULATING NEIGBOURING CELLS ---"
        print "-------------------------------------"

    neigbours = Neigbors()

    ## value 0.000625 is one foruth of the basis of the grid ###
    #c = round_coordinates(coors, round_cell_size)
    #    c=np.round(coors/0.000625)*0.000625
    coors_rounded = copy.deepcopy(c)
    q=np.arange(0,coors_rounded.shape[0], dtype=np.int32)
    #c=np.vstack((c.transpose(),q)).transpose()

    xs = np.unique(c[:, 0])
    ys = np.unique(c[:, 1])
    zs = np.unique(c[:, 2])

    neigbours._I = np.zeros(c.shape[0], dtype=np.int32)
    neigbours._I.fill(-1)

    neigbours.I_ = np.zeros(c.shape[0], dtype=np.int32)
    neigbours.I_.fill(-1)

    neigbours._J = np.zeros(c.shape[0], dtype=np.int32)
    neigbours._J.fill(-1)

    neigbours.J_ = np.zeros(c.shape[0], dtype=np.int32)
    neigbours.J_.fill(-1)

    neigbours._K = np.zeros(c.shape[0], dtype=np.int32)
    neigbours._K.fill(-1)

    neigbours.K_ = np.zeros(c.shape[0], dtype=np.int32)
    neigbours.K_.fill(-1)



    for x in xs:
        iyz=np.argwhere(c[:,0]==x)
        if iyz.size:
            cx = c[iyz[:,0]]
            qx = q[iyz[:,0]]
            for y in ys:
                iz = np.argwhere(cx[:,1]==y)
                if iz.size:
                    cx_ordered=cx[iz[:,0]]
                    qx_ordered=qx[iz[:,0]]
                    #cx_ordered=cx_ordered[cx_ordered[:,2].argsort()]
                    qx_ordered=qx_ordered[cx_ordered[:,2].argsort()]
                    #global_coors=cx_ordered[:,3].round().astype(int)
                    global_coors=qx_ordered
                    neigbours.K_[global_coors[:-1]]=global_coors[1:]
                    neigbours._K[global_coors[1:]]=global_coors[:-1]
            for z in zs:
                iy = np.argwhere(cx[:,2]==z)
                if iy.size:
                    cx_ordered=cx[iy[:,0]]
                    qx_ordered=qx[iy[:,0]]
                    #cx_ordered=cx_ordered[cx_ordered[:,1].argsort()]
                    qx_ordered=qx_ordered[cx_ordered[:,1].argsort()]
                    #global_coors=cx_ordered[:,3].round().astype(int)
                    global_coors=qx_ordered
                    neigbours.J_[global_coors[:-1]]=global_coors[1:]
                    neigbours._J[global_coors[1:]]=global_coors[:-1]

    for y in ys:
        ixz=np.argwhere(c[:,1]==y)
        if ixz.size:
            cy = c[ixz[:,0]]
            qy = q[ixz[:,0]]
            for z in zs:
                ix = np.argwhere(cy[:,2]==z)
                if ix.size:
                    cy_ordered=cy[ix[:,0]]
                    qy_ordered=qy[ix[:,0]]
                    #cy_ordered=cy_ordered[cy_ordered[:,0].argsort()]
                    qy_ordered=qy_ordered[cy_ordered[:,0].argsort()]
                    #global_coors=cy_ordered[:,3].round().astype(int)
                    global_coors=qy_ordered
                    neigbours.I_[global_coors[:-1]]=global_coors[1:]
                    neigbours._I[global_coors[1:]]=global_coors[:-1]

    blx_aux = c[np.argwhere(neigbours._I==-1)][:, 0, :]
    blx_aux_q = q[np.argwhere(neigbours._I==-1)][:,0]

    brx_aux = c[np.argwhere(neigbours.I_==-1)][:, 0, :]
    brx_aux_q = q[np.argwhere(neigbours.I_==-1)][:,0]

    bly_aux = c[np.argwhere(neigbours._J==-1)][:, 0, :]
    bly_aux_q = q[np.argwhere(neigbours._J==-1)][:,0]

    bry_aux = c[np.argwhere(neigbours.J_==-1)][:, 0, :]
    bry_aux_q = q[np.argwhere(neigbours.J_==-1)][:,0]

    blz_aux = c[np.argwhere(neigbours._K==-1)][:, 0, :]
    blz_aux_q = q[np.argwhere(neigbours._K==-1)][:,0]

    brz_aux = c[np.argwhere(neigbours.K_==-1)][:, 0, :]
    brz_aux_q = q[np.argwhere(neigbours.K_==-1)][:,0]

    blx_aux_aux = blx_aux[:,0]
    brx_aux_aux = brx_aux[:,0]

    bly_aux_aux = bly_aux[:,1]
    bry_aux_aux = bry_aux[:,1]

    blz_aux_aux = blz_aux[:,2]
    brz_aux_aux = brz_aux[:,2]


    max_x=np.max(c[:,0])
    min_x=np.min(c[:,0])

    max_y=np.max(c[:,1])
    min_y=np.min(c[:,1])

    max_z=np.max(c[:,2])
    min_z=np.min(c[:,2])

    arg_blx=np.argwhere(blx_aux_aux==min_x)[:,0]
    arg_bly=np.argwhere(bly_aux_aux==min_y)[:,0]
    arg_blz=np.argwhere(blz_aux_aux==min_z)[:,0]

    arg_brx=np.argwhere(brx_aux_aux==max_x)[:,0]
    arg_bry=np.argwhere(bry_aux_aux==max_y)[:,0]
    arg_brz=np.argwhere(brz_aux_aux==max_z)[:,0]

    boundaries = Boundary()

    boundaries.blx=blx_aux_q[arg_blx]
    boundaries.bly=bly_aux_q[arg_bly]
    boundaries.blz=blz_aux_q[arg_blz]

    boundaries.brx=brx_aux_q[arg_brx]
    boundaries.bry=bry_aux_q[arg_bry]
    boundaries.brz=brz_aux_q[arg_brz]

    # boundaries.blx=blx_aux[arg_blx][:,0,3]
    # boundaries.bly=bly_aux[arg_bly][:,0,3]
    # boundaries.blz=blz_aux[arg_blz][:,0,3]
    #
    # boundaries.brx=brx_aux[arg_brx][:,0,3]
    # boundaries.bry=bry_aux[arg_bry][:,0,3]
    # boundaries.brz=brz_aux[arg_brz][:,0,3]
    #
    # boundaries.blx=boundaries.blx.round().astype(int)
    # boundaries.bly=boundaries.bly.round().astype(int)
    # boundaries.blz=boundaries.blz.round().astype(int)
    #
    # boundaries.brx=boundaries.brx.round().astype(int)
    # boundaries.bry=boundaries.bry.round().astype(int)
    # boundaries.brz=boundaries.brz.round().astype(int)

    return [neigbours, boundaries]

# def calculate_global_eigenvectors(eigenvectors, comm, size, rank):
#
#     ### calculating shape ###
#     shape = np.zeros(size, dtype=np.int32)
#
#     shape_local = np.zeros(1, dtype=np.int32)
#     shape_local[0] = eigenvectors.T.shape[0]
#
#     comm.Allgather([shape_local, MPI.INT], [shape, MPI.INT])
#
#     count = vectors()
#     count_local = vectors()
#
#     ### calculating counts ###
#
#     count.zeros(size)
#
#     count_local.zeros(1)
#
#     count_local.T[0] = eigenvectors.T.size
#     count_local.P[0] = eigenvectors.P.size
#     count_local.u[0] = eigenvectors.u.size
#     count_local.v[0] = eigenvectors.v.size
#     count_local.w[0] = eigenvectors.w.size
#
# #    count_local.set(eigenvectors.P.size, eigenvectors.u.size, eigenvectors.v.size, eigenvectors.w.size, eigenvectors.T.size)
#
#     comm.Allgather([count_local.T, MPI.INT], [count.T, MPI.INT])
#     comm.Allgather([count_local.P, MPI.INT], [count.P, MPI.INT])
#     comm.Allgather([count_local.u, MPI.INT], [count.u, MPI.INT])
#     comm.Allgather([count_local.v, MPI.INT], [count.v, MPI.INT])
#     comm.Allgather([count_local.w, MPI.INT], [count.w, MPI.INT])
#
#     ### calculating displacements ###
#
#     desp = vectors()
#     desp.zeros_shape(count.P.size,
#                      count.u.size, count.v.size, count.w.size,
#                      count.T.size)
#
#     for j in range(desp.T.size):
#         if j > 0:
#             desp.T[j] = desp.T[j - 1] + count.T[j - 1]
#
#     for j in range(desp.P.size):
#         if j > 0:
#             desp.P[j] = desp.P[j - 1] + count.P[j - 1]
#
#     for j in range(desp.u.size):
#         if j > 0:
#             desp.u[j] = desp.u[j - 1] + count.u[j - 1]
#
#     for j in range(desp.v.size):
#         if j > 0:
#             desp.v[j] = desp.v[j - 1] + count.v[j - 1]
#
#     for j in range(desp.w.size):
#         if j > 0:
#             desp.w[j] = desp.w[j - 1] + count.w[j - 1]
#
#     eigenvectors_global =vectors()
#     eigenvectors_global.zeros_shape((np.sum(shape), eigenvectors.P.shape[1]),
#                                     (np.sum(shape), eigenvectors.u.shape[1]),
#                                     (np.sum(shape), eigenvectors.v.shape[1]),
#                                     (np.sum(shape), eigenvectors.w.shape[1]),
#                                     (np.sum(shape), eigenvectors.T.shape[1])
#                                     )
#
#     comm.Allgatherv([eigenvectors.P, MPI.DOUBLE], [eigenvectors_global.P, tuple(count.P), tuple(desp.P), MPI.DOUBLE])
#     comm.Allgatherv([eigenvectors.u, MPI.DOUBLE], [eigenvectors_global.u, tuple(count.u), tuple(desp.u), MPI.DOUBLE])
#     comm.Allgatherv([eigenvectors.v, MPI.DOUBLE], [eigenvectors_global.v, tuple(count.v), tuple(desp.v), MPI.DOUBLE])
#     comm.Allgatherv([eigenvectors.w, MPI.DOUBLE], [eigenvectors_global.w, tuple(count.w), tuple(desp.w), MPI.DOUBLE])
#     comm.Allgatherv([eigenvectors.T, MPI.DOUBLE], [eigenvectors_global.T, tuple(count.T), tuple(desp.T), MPI.DOUBLE])
#
#     return eigenvectors_global


def product_phis_universal_star_ccm_patch(first_derivative_eigenvectors,
                                          second_derivative_eigenvectors,
                                          eigenvectors,
                                          new_indexes,
                                          epsilon_sovolev,
                                          comm,
                                          size,
                                          rank):


    if rank==0:
        print "--------------------------------------"
        print "        Calculating Tensors  "
        print "--------------------------------------"

    f_d = Vectors()
    s_d = Vectors()

    f_d.P=np.swapaxes(np.swapaxes(first_derivative_eigenvectors.P, 0, 2), 0,1)
    f_d.T=np.swapaxes(np.swapaxes(first_derivative_eigenvectors.T, 0, 2), 0,1)
    f_d.u=np.swapaxes(np.swapaxes(first_derivative_eigenvectors.u, 0, 2), 0,1)
    f_d.v=np.swapaxes(np.swapaxes(first_derivative_eigenvectors.v, 0, 2), 0,1)
    f_d.w=np.swapaxes(np.swapaxes(first_derivative_eigenvectors.w, 0, 2), 0,1)

    s_d.P=np.swapaxes(np.swapaxes(np.swapaxes(second_derivative_eigenvectors.P, 0, 1), 1, 2), 2, 3)
    s_d.T=np.swapaxes(np.swapaxes(np.swapaxes(second_derivative_eigenvectors.T, 0, 1), 1, 2), 2, 3)
    s_d.u=np.swapaxes(np.swapaxes(np.swapaxes(second_derivative_eigenvectors.u, 0, 1), 1, 2), 2, 3)
    s_d.v=np.swapaxes(np.swapaxes(np.swapaxes(second_derivative_eigenvectors.v, 0, 1), 1, 2), 2, 3)
    s_d.w=np.swapaxes(np.swapaxes(np.swapaxes(second_derivative_eigenvectors.w, 0, 1), 1, 2), 2, 3)

    tensor=Matrices()
    #tensor_global=matrices()



    # [pis, us, taus]=calculate_derivatives(read_stencils, path, coors, coors_global_dim_rounded, eigenvectors_global, neigbours)


    vus = np.array([eigenvectors.u.transpose()[:, new_indexes.local_indices_inner],
                    eigenvectors.v.transpose()[:, new_indexes.local_indices_inner],
                    eigenvectors.w.transpose()[:, new_indexes.local_indices_inner]])

    dvus = np.array([f_d.u[:, :, new_indexes.local_indices_inner],
                     f_d.v[:, :, new_indexes.local_indices_inner],
                     f_d.w[:, :, new_indexes.local_indices_inner]])

    ddvus = np.array([s_d.u[:, :, :, new_indexes.local_indices_inner],
                      s_d.v[:, :, :, new_indexes.local_indices_inner],
                      s_d.w[:, :, :, new_indexes.local_indices_inner]])

    taus = eigenvectors.T.transpose()[:, new_indexes.local_indices_inner]
    dtaus = f_d.T[:, :, new_indexes.local_indices_inner]
    ddtaus = s_d.T[:, :, :, new_indexes.local_indices_inner]

    pis = eigenvectors.P.transpose()[:, new_indexes.local_indices_inner]
    dpis = f_d.P[:, :, new_indexes.local_indices_inner]
    ddpis = s_d.P[:, :, :, new_indexes.local_indices_inner]

    tensor.A=np.zeros((vus[0].shape[0], vus[0].shape[0]))
    producto.calculo_A(vus, tensor.A)

    tensor.B=np.zeros((vus[0].shape[0], vus[0].shape[0]))
    producto.calculo_B(dvus, tensor.B)

    tensor.S=np.zeros((vus[0].shape[0], vus[0].shape[0]))
    producto.calculo_S(ddvus, tensor.S)

    tensor.C=np.zeros((vus[0].shape[0], taus.shape[0]))
    producto.calculo_C(vus, taus, tensor.C)

    tensor.G=np.zeros((vus[0].shape[0], taus.shape[0]))
    producto.calculo_G(dvus, dtaus, tensor.G)

    tensor.Omega=np.zeros((vus[0].shape[0], vus[0].shape[0], vus[0].shape[0]))
    producto.calculo_Omega(vus, dvus, tensor.Omega)

    tensor.psi=np.zeros((vus[0].shape[0], vus[0].shape[0], vus[0].shape[0]))
    producto.calculo_psi(dvus, tensor.psi)

    tensor.chi = np.zeros((vus[0].shape[0], vus[0].shape[0], vus[0].shape[0]))
    producto.calculo_chi(vus, dvus, ddvus, tensor.chi)

    tensor.H=np.zeros((taus.shape[0], taus.shape[0]))
    producto.calculo_H(taus, tensor.H)

    tensor.J=np.zeros((taus.shape[0], vus[0].shape[0]))
    producto.calculo_J(vus, taus, tensor.J)

    tensor.L=np.zeros((taus.shape[0], taus.shape[0]))
    producto.calculo_L(dtaus, tensor.L)

    tensor.N=np.zeros((taus.shape[0], taus.shape[0]))
    producto.calculo_N(ddtaus, tensor.N)

    tensor.M=np.zeros((taus.shape[0], vus[0].shape[0]))
    producto.calculo_M(dvus, dtaus, tensor.M)

    tensor.Kappa=np.zeros((taus.shape[0], vus[0].shape[0], taus.shape[0]))
    producto.calculo_Kappa(vus, taus, dtaus, tensor.Kappa)

    tensor.upsilon=np.zeros((taus.shape[0], vus[0].shape[0], taus.shape[0]))
    producto.calculo_upsilon(dvus, dtaus, tensor.upsilon)

    tensor.omicron=np.zeros((taus.shape[0], vus[0].shape[0], taus.shape[0]))
    producto.calculo_omicron(vus, dtaus, ddtaus, tensor.omicron)


# chequear esto cuando empiece con la presion


    tensor.E=np.zeros((pis.shape[0], pis.shape[0]))
    producto.calculo_E(dpis, tensor.E)

    tensor.E_star = np.zeros((pis.shape[0], pis.shape[0]))
    producto.calculo_E_star(ddpis, tensor.E_star)

    tensor.F=np.zeros((pis.shape[0], taus.shape[0]))
    producto.calculo_F(pis, dtaus, tensor.F)

    tensor.F_star = np.zeros((pis.shape[0], taus.shape[0]))
    producto.calculo_F_star(dpis, ddtaus, tensor.F_star)

    tensor.Lambda=np.zeros((pis.shape[0], vus[0].shape[0], vus[0].shape[0]))
    producto.calculo_Lambda(pis, dvus, tensor.Lambda)

    tensor.Lambda_star=np.zeros((pis.shape[0], vus[0].shape[0], vus[0].shape[0]))
    producto.calculo_Lambda_star(ddpis, dvus, tensor.Lambda_star)

    # for k in range(pis[0].shape[0]):
    #     for l in range(pis[0].shape[0]):
    #         for i in range(3):
    #             tensor.E[k][l]+=np.dot(pis[1][k][i], pis[1][l][i])


    # for k in range(pis[0].shape[0]):
    #     for l in range(taus[0].shape[0]):
    #         tensor.F[k][l]=np.dot(pis[0][k], taus[1][l][1])



    # for i in range(3):
    #     for j in range(3):
    #         for k in range(pis[0].shape[0]):
    #             for l in range(us[0][0].shape[0]):
    #                 for m in range(us[0][0].shape[0]):
    #                    tensor.Lambda[k][l][m] += np.dot(pis[0][k], us[j][1][l][i]*us[i][1][m][j])

    tensor.A = comm.allreduce(tensor.A)
    tensor.B = comm.allreduce(tensor.B)
    tensor.C = comm.allreduce(tensor.C)

    tensor.E = comm.allreduce(tensor.E)
    tensor.F = comm.allreduce(tensor.F)

    tensor.E_star = comm.allreduce(tensor.E_star)
    tensor.F_star = comm.allreduce(tensor.F_star)

    tensor.J = comm.allreduce(tensor.J)
    tensor.L = comm.allreduce(tensor.L)

    tensor.Omega = comm.allreduce(tensor.Omega)
    tensor.Kappa = comm.allreduce(tensor.Kappa)

    tensor.Lambda = comm.allreduce(tensor.Lambda)
    tensor.Lambda_star = comm.allreduce(tensor.Lambda_star)

    tensor.S = comm.allreduce(tensor.S)
    tensor.G = comm.allreduce(tensor.G)
    tensor.psi = comm.allreduce(tensor.psi)
    tensor.chi = comm.allreduce(tensor.chi)
    tensor.H = comm.allreduce(tensor.H)
    tensor.N = comm.allreduce(tensor.N)
    tensor.M = comm.allreduce(tensor.M)
    tensor.upsilon = comm.allreduce(tensor.upsilon)
    tensor.omicron = comm.allreduce(tensor.omicron)

    new_tensor = Matrices()

    new_tensor.alpha = St * tensor.A + St * epsilon_sovolev * tensor.B
    new_tensor.beta = Re_inv * tensor.B + Re_inv * epsilon_sovolev * tensor.S
    new_tensor.nu = RR * tensor.C + RR * epsilon_sovolev * tensor.G
    new_tensor.zeta =  tensor.Omega + epsilon_sovolev * tensor.psi + epsilon_sovolev * tensor.chi
    new_tensor.eta = St * tensor.H + St * epsilon_sovolev * tensor.L
    new_tensor.iota = Ty * (tensor.J + epsilon_sovolev * tensor.M)
    new_tensor.phi = Re_inv * Pr_inv * (tensor.L + epsilon_sovolev * tensor.N)
    new_tensor.mu = tensor.Kappa + epsilon_sovolev * tensor.upsilon + epsilon_sovolev * tensor.omicron

    new_tensor.E_prima = tensor.E + epsilon_sovolev * tensor.E_star
    new_tensor.F_prima = RR * (tensor.F + epsilon_sovolev * tensor.F_star)
    new_tensor.Lambda_prima = tensor.Lambda - epsilon_sovolev * tensor.Lambda_star

    return new_tensor #, tensor
