import numpy as np
from vectors import Vectors

def y_calculator(tensor, amplitude, d_time):

    """Calculated the Y matrix of the linear system"""

    y11 = tensor.alpha + d_time/2.0*(tensor.beta +
                                     np.tensordot(tensor.zeta, amplitude.u, axes=([2], [0])) +
                                     np.tensordot(tensor.zeta, amplitude.u, axes=([1], [0]))
                                    )
    y12 = -d_time/2.0*tensor.nu

    y21 = d_time/2.0*(tensor.iota+
                      np.tensordot(tensor.mu, amplitude.T, axes=([2], [0])))

    y22 = tensor.eta + d_time/2.0 * (tensor.phi +
                                     np.tensordot(tensor.mu, amplitude.u, axes=([1], [0])))

    y_1 = np.hstack((y11, y12))
    y_2 = np.hstack((y21, y22))
    y_sol = np.vstack((y_1, y_2))

    return y_sol

def z_calculator(tensor, amplitude, d_time):

    """Calculates the Z vector of the linear system"""

    z_1 = d_time * (- np.dot(tensor.beta, amplitude.u)
                    + np.dot(tensor.nu, amplitude.T)
                    - np.tensordot(np.tensordot(tensor.zeta,
                                                amplitude.u,
                                                axes=([2], [0])),
                                   amplitude.u, axes=([1], [0])))

    z_2 = d_time * (- np.dot(tensor.iota, amplitude.u)
                    - np.dot(tensor.phi, amplitude.T)
                    - np.tensordot(np.tensordot(tensor.mu,
                                                amplitude.T,
                                                axes=([2], [0])),
                                   amplitude.u, axes=([1], [0])))

    z_sol = np.append(z_1, z_2)

    return z_sol

def linear_system_components(tensor, amplitude, d_time):

    """Organizes the calculation of the components of the linear system"""

    y_sol = y_calculator(tensor, amplitude, d_time)
    z_sol = z_calculator(tensor, amplitude, d_time)

    return y_sol, z_sol



def solve_linear_system(matrix_y, matrix_z, shape):

    """Solves and esturn the linar system"""

    matrix_x = np.linalg.solve(matrix_y, matrix_z)

    d_amplitudes = Vectors()

    d_amplitudes.u = matrix_x[:shape]
    d_amplitudes.T = matrix_x[shape:]

    return d_amplitudes

def non_linear_cn(sols, tensor, matrix_y, matrix_z, d_time):

    """Calculates the non linar parts of the system"""

    d_vels = sols[:tensor.alpha.shape[0]]
    d_temp = matrix_z[tensor.alpha.shape[0]:]

    matrix_x1 = np.tensordot(np.tensordot(tensor.zeta,
                                          d_vels,
                                          axes=([2], [0])),
                             d_vels,
                             axes=([1], [0]))
    matrix_x2 = np.tensordot(np.tensordot(tensor.mu,
                                          d_temp,
                                          axes=([2], [0])),
                             d_vels,
                             axes=([1], [0]))

    matrix_x = np.append(matrix_x1, matrix_x2)

    return np.dot(matrix_y, sols) + d_time/2.0*matrix_x - matrix_z

def error_non_linear_cn(amplitudes, a_new, tensor, d_time):

    """Calculates the numerical error of the system"""

    matrix_z_n = z_calculator(tensor, amplitudes, d_time/2.)
    matrix_z_n_1 = z_calculator(tensor, a_new, d_time/2.)

    e_n_1 = matrix_z_n_1-matrix_z_n

    error = Vectors()
    error.u = np.dot(np.linalg.inv(tensor.alpha), e_n_1[:tensor.alpha.shape[0]])
    error.T = np.dot(np.linalg.inv(tensor.eta), e_n_1[tensor.alpha.shape[0]:])

    return error
#
# def pressure_eq_lineal(a, tensor):
#
#     vectb = np.tensordot(np.tensordot(tensor.Lambda, a.u, axes=([2], [0])),
# a.u, axes=([1], [0])) - RR*np.dot(tensor.F, a.T)
#
#     X=np.linalg.solve(tensor.E, vectb)
#
#     return X[:]

def pressure_eq_lineal_sovolev(amplitudes, tensor):

    """Resolve the pressure equation. Pressure is a dependent magnitude"""

    matrix_x = np.linalg.solve(tensor.E_prima,
                               np.tensordot(np.tensordot(tensor.Lambda_prima,
                                                         amplitudes.u,
                                                         axes=([2], [0])),
                                            amplitudes.u, axes=([1], [0]))
                               - np.dot(tensor.F_prima, amplitudes.T))

    return matrix_x[:]

def stiffness(tensor):

    """Calculates the stiffness of the system"""

    matrix_m1_1 = np.hstack((tensor.alpha, np.zeros((tensor.alpha.shape[0], tensor.eta.shape[1]))))
    matrix_m1_2 = np.hstack((np.zeros((tensor.eta.shape[0], tensor.alpha.shape[1])), tensor.eta))
    matrix_m1 = np.vstack((matrix_m1_1, matrix_m1_2))

    matrix_m2_1 = np.hstack((tensor.beta, -tensor.nu))
    matrix_m2_2 = np.hstack((tensor.iota, tensor.phi))
    matrix_m2 = np.vstack((matrix_m2_1, matrix_m2_2))

    matrix_mlin = np.dot(np.linalg.inv(matrix_m1), matrix_m2)

    veig, weig = np.linalg.eig(matrix_mlin)

    vveig = np.fabs(np.real(veig))
    vvmax = np.max(vveig)
    vvmin = np.min(vveig)

    print " "
    print "-----------------------------------------------------------------"
    print "The eigenvalues of the system are:"
    print np.real(veig)
    print "The relation of eigenvalues of the system of equations are:"
    print "Max: %f"%(vvmax)
    print "Min: %f"%(vvmin)
    print "The relation max/min is: %f"%(vvmax/vvmin)

    print "-----------------------------------------------------------------"
