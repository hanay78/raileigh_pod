import numpy as np
#from scipy.sparse.linalg import eigsh, svds
#from scipy.linalg import eig, eigh, svd
from scipy.sparse.linalg import eigsh
from scipy.linalg import eig
from mpi4py import MPI


def substract_global_mean(t_variable, comm, size):

    """calculates the meanles variable and the mean"""

    t_mean_local = np.zeros(1, dtype=np.float64)
    t_mean = np.zeros(size, dtype=np.float64)

    t_mean_local[0] = np.mean(t_variable)

    comm.Allgather([t_mean_local, MPI.DOUBLE], [t_mean, MPI.DOUBLE])

    return (t_variable-np.mean(t_mean), np.mean(t_mean))

def calculate_autocorrelation_matrix(theta, comm):

    """calculates the autocorrelation matrix"""

    theta_t = theta.transpose()
    theta_t_theta_local = np.dot(theta_t, theta)
    theta_t_theta = np.zeros(theta_t_theta_local.shape)

    comm.Allreduce([theta_t_theta_local, MPI.DOUBLE], [theta_t_theta, MPI.DOUBLE], op=MPI.SUM)

    return theta_t_theta


def calculate_pod_representative(theta_ghost,
                                 theta_t_theta,
                                 max_err_pod,
                                 high_frequency_fraction):

    """calculates the pod for a threshold of signification
    variable number of components
    """

    # ### initial number of eigenvectors values ###
    # NumEv = 1
    #
    # ### iterate for accuracy serched for ###
    # while(True):
    #     ### calculate eigenvalues and eigenvectors ###
    #     (ltheta, vtheta) = eigsh(theta_t_theta, NumEv, which='LM')
    #
    #     ### estimate error ###
    #     podmaxerr = (theta_t_theta.shape[0]-NumEv)*ltheta[0]/ltheta.sum()
    #
    #     ### check error ###
    #     if(podmaxerr<max_err_pod):
    #
    #         ### check there are not mistakes due to very small eigenvalues ###
    #         while (ltheta[0] < 0.):
    #             ll = ltheta[1:]
    #             vt = vtheta[:, 1:]
    #             (ltheta, vtheta) = (ll, vt)
    #         break
    #
    #     ### increase the number of modes and iterate again ###
    #     NumEv+=1
    #
    # comm.Barrier()
    #
    # #ltheta_inv=1./np.sqrt(ltheta[:])
    #
    # vtheta=np.fliplr(vtheta)
    # ltheta=np.flipud(ltheta)
    #
    # ### calculate Y=UDV, resolving YVtD-1=U ###
    # utheta=np.multiply(np.dot(theta,vtheta),1./np.sqrt(ltheta[:]))
    #
    #
    # ### calculate norm of vectors ###
    # uaux_local = np.zeros(utheta.shape[1])
    # uthetaT=utheta.transpose()
    # for i in range(uaux_local.size):
    #     uaux_local[i]=np.dot(uthetaT[i,:], utheta[:,i])
    #
    # uaux = np.zeros(uaux_local.shape)
    # comm.Allreduce([uaux_local, MPI.DOUBLE], [uaux, MPI.DOUBLE], op=MPI.SUM)
    #
    # ptheta = np.sqrt(uaux)
    #
    # ### make unitary ###
    # uu_theta=utheta/ptheta
    # return (ltheta, uu_theta, NumEv, NumEv)
    #


    #################################################################
    eigenvalues, eigenvectors = eig(theta_t_theta)
    eigenvalues = np.ascontiguousarray(np.real(eigenvalues), dtype=np.float32)
    eigenvectors = np.ascontiguousarray(np.real(eigenvectors), dtype=np.float32)
    high_frequency_number = eigenvalues.shape[0]
    low_frequency_number = eigenvalues.shape[0]
    suml = eigenvalues.sum()
    while True:
        high_frequency_number = high_frequency_number - 1
        error_p = np.sum(eigenvalues[high_frequency_number:])/suml
        if error_p > max_err_pod/high_frequency_fraction:
            break
    while True:
        low_frequency_number = low_frequency_number -1
        error_p = np.sum(eigenvalues[low_frequency_number:])/suml
        if error_p > max_err_pod:
            break

    eigenvalues1 = eigenvalues[:high_frequency_number+1]
    eigenvectors1 = eigenvectors[:, :high_frequency_number+1]
    u11 = np.multiply(np.dot(theta_ghost, eigenvectors1), 1./np.sqrt(eigenvalues1[:]))
    #u11=np.multiply(np.dot(theta,eigenvectors1), 1./np.sqrt(eigenvalues1[:]))
    return (eigenvalues1, u11, low_frequency_number+1, high_frequency_number+1)

    #################################################################

    #
    # l2, v2 = eigh(theta_t_theta)
    # n2=0
    # suml2=l2.sum()
    # while(True):
    #     n2 = n2 + 1
    #     error_p= np.sum(l2[:n2])/suml2
    #     if error_p>max_err_pod:
    #         break
    #
    # l21=l2[n2:]
    # v21=v2[:, n2:]
    # u21=np.multiply(np.dot(theta,v21), 1./np.sqrt(l21[:]))
    #
    # u21=np.fliplr(u21)
    # v21=np.fliplr(v21)
    # l21=np.flipud(l21)


    # NumEv = 1
    # NumEv1 = 1
    # while(True):
    #     u, d, v = svds(theta, k=NumEv1)
    #     podmaxerr = (theta.shape[1]-NumEv1)*d[0]**2/np.dot(d,d)
    #     if(podmaxerr<max_err_pod/100):
    #         break
    #     NumEv1=NumEv1+1
    #     if(podmaxerr>=max_err_pod):
    #         NumEv=NumEv+1
    # # ### return values ###
    # # return (ltheta, uu_theta)
    # return (np.flipud(d),np.fliplr(u),NumEv,NumEv1)


def calculate_number_pods(theta, theta_t_theta, n_pod, comm):
    """calculate pods for a fixed numbers of components"""

    (ltheta, vtheta) = eigsh(theta_t_theta, n_pod, which='LM')
    while ltheta[0] < 0.:
        llth = ltheta[1:]
        vtth = vtheta[:, 1:]
        (ltheta, vtheta) = (llth, vtth)

    comm.Barrier()
    ltheta_inv = 1./np.sqrt(ltheta[:])

    utheta = np.multiply(np.dot(theta, vtheta), ltheta_inv)

    uaux_local = np.zeros(utheta.shape[1])
    utheta_t = utheta.transpose()
    for i in range(uaux_local.size):
        uaux_local[i] = np.dot(utheta_t[i, :], utheta[:, i])
    #uaux_local = np.dot(utheta.transpose(), utheta)
    uaux = np.zeros(uaux_local.shape)
    comm.Allreduce([uaux_local, MPI.DOUBLE], [uaux, MPI.DOUBLE], op=MPI.SUM)

    ptheta = np.sqrt(uaux)
    #ptheta = LA.norm(utheta, axis=0)

    uu_theta = utheta/ptheta

    podmaxerr = (theta_t_theta.shape[0]-n_pod)*ltheta[0]/ltheta.sum()

    return (ltheta, uu_theta, podmaxerr)



# def calculate_pod(T, max_err_pod, comm, size, rank):
#     (theta, meanT) = substract_global_mean(T, comm, size, rank)
#     theta_t_theta = calculate_autocorrelation_matrix(theta, comm, size, rank)
#     (ltheta, utheta)=calculate_pod_representative(theta,
                                                    # theta_t_theta, max_err_pod,
                                                    # comm, size, rank)
#     return (meanT, theta, ltheta, utheta)
#
def calculate_pod_simple(variable, n_pod, comm, size, rank):

    """autocrorelation matrix and pod calculation"""
    theta_t_theta = calculate_autocorrelation_matrix(variable, comm)
    (ltheta, utheta, podmaxerr) = calculate_number_pods(variable, theta_t_theta, n_pod, comm)
    return (ltheta, utheta, podmaxerr)

def calculate_pod_simple_accuracy(var_ghost, var, accuracy, high_frequency_fraction, comm, size, rank):

    """autocrorelation matrix and pod calculation"""

    theta_t_theta = calculate_autocorrelation_matrix(var, comm)
    (ltheta, utheta, low_n, high_n) = calculate_pod_representative(var_ghost,
                                                                   theta_t_theta,
                                                                   accuracy,
                                                                   high_frequency_fraction)
    return (ltheta, utheta, low_n, high_n)

# def calculate_pod_simple_accuracy_patch(T, accuracy, new_indexes, comm, size, rank):
#
#     theta_t_theta = calculate_autocorrelation_matrix_patch(T, new_indexes, comm, size, rank)
#     (ltheta, utheta, n, n1)=calculate_pod_representative(T,
    # theta_t_theta,
    # accuracy,
    # comm, size, rank)
#     return (ltheta, utheta, n, n1)
