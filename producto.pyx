import numpy as np
cimport numpy as np

ctypedef np.float_t DTYPE_t

cdef DTYPE_t my_dot(np.ndarray[DTYPE_t, ndim=1] a, np.ndarray[DTYPE_t, ndim=1] b):
    cdef double r
    r=0.0
    for i in range(a.shape[0]):
        r+=a[i]*b[i]
    return r

def calculo_A(np.ndarray[DTYPE_t, ndim=3] vus, np.ndarray[DTYPE_t, ndim=2]  A):

    cdef int k, l, i

    for k in range(vus[0].shape[0]):
        for l in range(vus[0].shape[0]):
            for i in range(3):
                A[k][l] += np.dot(vus[i][k], vus[i][l])

def calculo_B(np.ndarray[DTYPE_t, ndim=4] dvus, np.ndarray[DTYPE_t, ndim=2] B):

    cdef int i, j, k, l

    for i in range(3):
        for j in range(3):
            for k in range(dvus[0].shape[0]):
                for l in range(dvus[0].shape[0]):
                    B[k][l] += np.dot(dvus[i][k][j],dvus[i][l][j])

def calculo_S(np.ndarray[DTYPE_t, ndim=5] ddvus, np.ndarray[DTYPE_t, ndim=2] S):

    cdef int i, j, m, k, l

    for i in range(3):
        for j in range(3):
            for m in range(3):
                for k in range(ddvus[0].shape[0]):
                    for l in range(ddvus[0].shape[0]):
                        S[k][l] += np.dot(ddvus[i][k][j][m],ddvus[i][l][j][m])

def calculo_C(np.ndarray[DTYPE_t, ndim=3] vus, np.ndarray[DTYPE_t, ndim=2] taus, np.ndarray[DTYPE_t, ndim=2] C):

    cdef int k, l

    for k in range(vus[0].shape[0]):
        for l in range(taus.shape[0]):
            C[k][l] = np.dot(vus[1][k], taus[l])

def calculo_G(np.ndarray[DTYPE_t, ndim=4] dvus, np.ndarray[DTYPE_t, ndim=3] dtaus, np.ndarray[DTYPE_t, ndim=2] G):

    cdef int k, l, j

    for k in range(dvus[0].shape[0]):
        for l in range(dtaus.shape[0]):
            for j in range(3):
                G[k][l] += np.dot(dvus[1][k][j], dtaus[l][j])

def calculo_Omega(np.ndarray[DTYPE_t, ndim=3] vus, np.ndarray[DTYPE_t, ndim=4] dvus, np.ndarray[DTYPE_t, ndim=3] Omega):

    cdef int i, j, k, l, m

    for i in range(3):
        for j in range(3):
            for k in range(vus[0].shape[0]):
                for l in range(vus[0].shape[0]):
                    for m in range(dvus[0].shape[0]):

                       Omega[k][l][m] += np.dot(vus[i][k], vus[j][l]*dvus[i][m][j])

def calculo_psi(np.ndarray[DTYPE_t, ndim=4] dvus, np.ndarray[DTYPE_t, ndim=3] psi):

    cdef int n, i, j, k, l, m

    for n in range(3):
        for i in range(3):
            for j in range(3):
                for k in range(dvus[0].shape[0]):
                    for l in range(dvus[0].shape[0]):
                        for m in range(dvus[0].shape[0]):

                            psi[k][l][m] += np.dot(dvus[i][k][n], dvus[j][l][n] * dvus[i][m][j])

def calculo_chi(np.ndarray[DTYPE_t, ndim=3] vus, np.ndarray[DTYPE_t, ndim=4] dvus, np.ndarray[DTYPE_t, ndim=5] ddvus, np.ndarray[DTYPE_t, ndim=3] chi):

    cdef int n, i, j, k, l, m

    for n in range(3):
        for i in range(3):
            for j in range(3):
                for k in range(dvus[0].shape[0]):
                    for l in range(vus[0].shape[0]):
                        for m in range(ddvus[0].shape[0]):

                            chi[k][l][m] += np.dot(dvus[i][k][n], vus[j][l] * ddvus[i][m][n][j])

def calculo_H(np.ndarray[DTYPE_t, ndim=2] taus, np.ndarray[DTYPE_t, ndim=2] H):

    cdef int k, l

    for k in range(taus.shape[0]):
        for l in range(taus.shape[0]):

            H[k][l] = np.dot(taus[k], taus[l])

def calculo_J(np.ndarray[DTYPE_t, ndim=3] vus, np.ndarray[DTYPE_t, ndim=2] taus, np.ndarray[DTYPE_t, ndim=2] J):

    cdef int k, l

    for k in range(taus.shape[0]):
        for l in range(vus[0].shape[0]):
            J[k][l] = np.dot(taus[k], vus[1][l])

def calculo_L(np.ndarray[DTYPE_t, ndim=3] dtaus, np.ndarray[DTYPE_t, ndim=2] L):

    cdef int k, l, j

    for k in range(dtaus.shape[0]):
        for l in range(dtaus.shape[0]):
            for j in range(3):
                L[k][l]+= np.dot(dtaus[k][j], dtaus[l][j])

def calculo_N(np.ndarray[DTYPE_t, ndim=4] ddtaus, np.ndarray[DTYPE_t, ndim=2] N):

    cdef int k, l, i, j

    for i in range(3):
        for j in range(3):
            for k in range(ddtaus.shape[0]):
                for l in range(ddtaus.shape[0]):
                    N[k][l]+= np.dot(ddtaus[k][i][j], ddtaus[l][i][j])

def calculo_M(np.ndarray[DTYPE_t, ndim=4] dvus, np.ndarray[DTYPE_t, ndim=3] dtaus, np.ndarray[DTYPE_t, ndim=2] M):

    cdef int k, l, j

    for k in range(dtaus.shape[0]):
        for l in range(dvus[0].shape[0]):
            for j in range(3):
                M[k][l]= np.dot(dtaus[k][j], dvus[1][l][j])

def calculo_Kappa(np.ndarray[DTYPE_t, ndim=3] vus, np.ndarray[DTYPE_t, ndim=2] taus, np.ndarray[DTYPE_t, ndim=3] dtaus, np.ndarray[DTYPE_t, ndim=3] Kappa):

    cdef int k, l, m, i

    for k in range(taus.shape[0]):
        for l in range(vus[0].shape[0]):
            for m in range(taus.shape[0]):
                for i in range(3):
                    Kappa[k][l][m]+= np.dot(taus[k], vus[i][l]*dtaus[m][i])

def calculo_upsilon(np.ndarray[DTYPE_t, ndim=4] dvus, np.ndarray[DTYPE_t, ndim=3] dtaus, np.ndarray[DTYPE_t, ndim=3] upsilon):

    cdef int n, i, k, l, m

    for n in range(3):
        for i in range(3):
            for k in range(dtaus.shape[0]):
                for l in range(dvus[0].shape[0]):
                    for m in range(dtaus.shape[0]):
                        upsilon[k][l][m] += np.dot(dtaus[k][n], dvus[i][l][n]*dtaus[m][i])

def calculo_omicron(np.ndarray[DTYPE_t, ndim=3] vus, np.ndarray[DTYPE_t, ndim=3] dtaus, np.ndarray[DTYPE_t, ndim=4] ddtaus, np.ndarray[DTYPE_t, ndim=3] omicron):

    cdef int n, i, k, l, m

    for n in range(3):
        for i in range(3):
            for k in range(dtaus.shape[0]):
                for l in range(vus[0].shape[0]):
                    for m in range(ddtaus.shape[0]):
                        omicron[k][l][m] += np.dot(dtaus[k][n], vus[i][l] * ddtaus[m][n][i])

def calculo_E(np.ndarray[DTYPE_t, ndim=3] dpis, np.ndarray[DTYPE_t, ndim=2] E):

    cdef int k, l, i

    for k in range(dpis.shape[0]):
        for l in range(dpis.shape[0]):
            for i in range(3):

                E[k][l]+=np.dot(dpis[k][i], dpis[l][i])

def calculo_E_star(np.ndarray[DTYPE_t, ndim=4] ddpis, np.ndarray[DTYPE_t, ndim=2] E_star):

    cdef int k, l, i, j

    for k in range(ddpis.shape[0]):
        for l in range(ddpis.shape[0]):
            for i in range(3):
                for j in range(3):

                    E_star[k][l] += np.dot(ddpis[k][i][i], ddpis[l][j][j])

def calculo_F(np.ndarray[DTYPE_t, ndim=2] pis, np.ndarray[DTYPE_t, ndim=3] dtaus, np.ndarray[DTYPE_t, ndim=2] F):

    cdef int k, l

    for k in range(pis.shape[0]):
        for l in range(dtaus.shape[0]):

            F[k][l] = np.dot(pis[k], dtaus[l][1])


def calculo_F_star(np.ndarray[DTYPE_t, ndim=3] dpis, np.ndarray[DTYPE_t, ndim=4] ddtaus, np.ndarray[DTYPE_t, ndim=2] F_star):

    cdef int k, l, i

    for k in range(dpis.shape[0]):
        for l in range(ddtaus.shape[0]):
            for i in range(3):
                # ver si la i es el segundo o tercer inidice en ddtaus
                F_star[k][l] += np.dot(dpis[k][i], ddtaus[l][i][1])



def calculo_Lambda(np.ndarray[DTYPE_t, ndim=2] pis, np.ndarray[DTYPE_t, ndim=4] dvus, np.ndarray[DTYPE_t, ndim=3] Lambda):

    cdef int i, j, k, l, m

    for i in range(3):
        for j in range(3):
            for k in range(pis.shape[0]):
                for l in range(dvus[0].shape[0]):
                    for m in range(dvus[0].shape[0]):
                       Lambda[k][l][m] += np.dot(pis[k], dvus[j][l][i]*dvus[i][m][j])

def calculo_Lambda_star(np.ndarray[DTYPE_t, ndim=4] ddpis, np.ndarray[DTYPE_t, ndim=4] dvus, np.ndarray[DTYPE_t, ndim=3] Lambda_star):

    cdef int i, j, n, k, l, m

    for i in range(3):
        for j in range(3):
            for n in range(3):
                for k in range(ddpis.shape[0]):
                    for l in range(dvus[0].shape[0]):
                        for m in range(dvus[0].shape[0]):
                            Lambda_star[k][l][m] += np.dot(ddpis[k][n][n], dvus[j][l][i]*dvus[i][m][j])