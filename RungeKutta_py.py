
#from small_python_auxiliary_functions import *
import numpy as np
#from RHS_python_version import RHS

def time_integration(comm, rank, size, linear_theta, linear_K, all_limits, X, t, dt, avT, avK, a_theta, a_K, utheta, uK, tau1, tau2):
    (d_theta_1, d_K_1) = \
        RHS(comm, rank, size, linear_theta, linear_K,all_limits, X, t, avT, avK, a_theta, a_K,
            utheta, uK, tau1, tau2)
    a_theta_1 = a_theta + dt / 2. * d_theta_1
    a_K_1 = a_K + dt / 2. * d_K_1

    (d_theta_2, d_K_2) = \
        RHS(comm, rank, size, linear_theta, linear_K,all_limits, X, t, avT, avK, a_theta_1, a_K_1,
            utheta, uK, tau1, tau2)
    a_theta_2 = a_theta_1 + dt / 2. * d_theta_2
    a_K_2 = a_K_1 + dt / 2. * d_K_2

    (d_theta_3, d_K_3) = \
        RHS(comm, rank, size, linear_theta, linear_K,all_limits, X, t, avT, avK, a_theta_2, a_K_2,
            utheta, uK, tau1, tau2)
    a_theta_3 = a_theta_2 + dt * d_theta_3
    a_K_3 = a_K_2 + dt * d_K_3

    (d_theta_4, d_K_4) = \
        RHS(comm, rank, size, linear_theta, linear_K, all_limits, X, t, avT, avK, a_theta_3, a_K_3,
            utheta, uK, tau1, tau2)
    a_theta += dt / 6. * (d_theta_1 + 2. * d_theta_2 + 2. * d_theta_3 + d_theta_4)
    a_K += dt / 6. * (d_K_1 + 2. * d_K_2 + 2. * d_K_3 + d_K_4)


def time_integration_3(comm, rank, size, linear_theta, linear_K, all_limits, X, t, dt, avT, avK, a_theta, a_K, utheta, uK, tau1, tau2, eT, eK):
    k_theta = np.zeros((5, a_theta.shape[0]))
    k_K = np.zeros((5, a_K.shape[0]))

    y_theta = np.zeros((5, a_theta.shape[0]))
    y_K = np.zeros((5, a_K.shape[0]))

    y_theta[1] = a_theta
    y_K[1] = a_K

    [k_theta[1], k_K[1]] = RHS(comm, rank, size, linear_theta, linear_K, all_limits, X, t, avT, avK, y_theta[1], y_K[1], utheta, uK, tau1, tau2)

    y_theta[2] = y_theta[1] + dt * 1. / 2. * k_theta[1]
    y_K[2] = y_K[1] + dt * 1. / 2. * k_K[1]
    [k_theta[2], k_K[2]] = RHS(comm, rank, size, linear_theta, linear_K, all_limits, X, t, avT, avK, y_theta[2], y_K[2], utheta, uK, tau1, tau2)

    y_theta[3] = y_theta[1] + dt * (0. * k_theta[1] + 3. / 4. * k_theta[2])
    y_K[3] = y_K[1] + dt * (0. * k_K[1] + 3. / 4. * k_K[2])
    [k_theta[3], k_K[3]] = RHS(comm, rank, size, linear_theta, linear_K, all_limits, X, t, avT, avK, y_theta[3], y_K[3], utheta, uK, tau1, tau2)

    y_theta[4] = y_theta[1] + dt * (2. / 9. * k_theta[1] + 1. / 3. * k_theta[2] + 4. / 9. * k_theta[3])
    y_K[4] = y_K[1] + dt * (2. / 9. * k_K[1] + 1. / 3. * k_K[2] + 4. / 9. * k_K[3])
    [k_theta[4], k_K[4]] = RHS(comm, rank, size, linear_theta, linear_K, all_limits, X, t, avT, avK, y_theta[4], y_K[4], utheta, uK, tau1, tau2)

    a_theta_star = a_theta + dt * (
        7. / 24. * k_theta[1] + 1. / 4. * k_theta[2] + 1. / 3. * k_theta[3] + 1. / 8. * k_theta[4])
    a_K_star = a_K + dt * (7. / 24. * k_K[1] + 1. / 4. * k_K[2] + 1. / 3. * k_K[3] + 1. / 8. * k_K[4])

    a_theta += dt * (2. / 9. * k_theta[1] + 1. / 3. * k_theta[2] + 4. / 9. * k_theta[3] + 0. * k_theta[4])
    a_K += dt * (2. / 9. * k_K[1] + 1. / 3. * k_K[2] + 4. / 9. * k_K[3] + 0. * k_K[4])

    eT += a_theta - a_theta_star
    eK += a_K - a_K_star


def time_integration_5(comm, rank, size, linear_theta, linear_K, all_limits, X, t, dt, avT, avK, a_theta, a_K, utheta, uK, tau1, tau2, eT, eK):
    k_theta = np.zeros((8, a_theta.shape[0]))
    k_K = np.zeros((8, a_K.shape[0]))

    y_theta = np.zeros((8, a_theta.shape[0]))
    y_K = np.zeros((8, a_K.shape[0]))

    y_theta[1] = a_theta
    y_K[1] = a_K
    (k_theta[1], k_K[1]) = RHS(comm, rank, size, linear_theta, linear_K, all_limits, X, t, avT, avK, y_theta[1], y_K[1], utheta, uK, tau1, tau2)

    y_theta[2] = y_theta[1] + dt * 1. / 5. * k_theta[1]
    y_K[2] = y_K[1] + dt * 1. / 5. * k_K[1]
    (k_theta[2], k_K[2]) = RHS(comm, rank, size, linear_theta, linear_K, all_limits, X, t, avT, avK, y_theta[2], y_K[2], utheta, uK, tau1, tau2)

    y_theta[3] = y_theta[1] + dt * (3. / 40. * k_theta[1] + 9. / 40. * k_theta[2])
    y_K[3] = y_K[1] + dt * (3. / 40. * k_K[1] + 9. / 40. * k_K[2])
    (k_theta[3], k_K[3]) = RHS(comm, rank, size, linear_theta, linear_K, all_limits, X, t, avT, avK, y_theta[3], y_K[3], utheta, uK, tau1, tau2)

    y_theta[4] = y_theta[1] + dt * (44. / 45. * k_theta[1] - 56. / 15. * k_theta[2] + 32. / 9. * k_theta[3])
    y_K[4] = y_K[1] + dt * (44. / 45. * k_K[1] - 56. / 15. * k_K[2] + 32. / 9. * k_K[3])
    (k_theta[4], k_K[4]) = RHS(comm, rank, size, linear_theta, linear_K, all_limits, X, t, avT, avK, y_theta[4], y_K[4], utheta, uK, tau1, tau2)

    y_theta[5] = y_theta[1] + dt * (
        19372. / 6561. * k_theta[1] - 25360. / 2187. * k_theta[2] + 64448. / 6561. * k_theta[3] - 212. / 729. * k_theta[
            4])
    y_K[5] = y_K[1] + dt * (
        19372. / 6561. * k_K[1] - 25360. / 2187. * k_K[2] + 64448. / 6561. * k_K[3] - 212. / 729. * k_K[4])
    (k_theta[5], k_K[5]) = RHS(comm, rank, size, linear_theta, linear_K, all_limits, X, t, avT, avK, y_theta[5], y_K[5], utheta, uK, tau1, tau2)

    y_theta[6] = y_theta[1] + dt * (
        9017. / 3168. * k_theta[1] - 355. / 33. * k_theta[2] + 46732. / 5247. * k_theta[3] + 49. / 176. * k_theta[4]
        - 5103. / 18656. * k_theta[5])
    y_K[6] = y_K[1] + dt * (
        9017. / 3168. * k_K[1] - 355. / 33. * k_K[2] + 46732. / 5247. * k_K[3] + 49. / 176. * k_K[4]
        - 5103. / 18656. * k_K[5])
    (k_theta[6], k_K[6]) = RHS(comm, rank, size, linear_theta, linear_K, all_limits, X, t, avT, avK, y_theta[6], y_K[6], utheta, uK, tau1, tau2)

    y_theta[7] = y_theta[1] + dt * (
        35. / 384. * k_theta[1] - 0. * k_theta[2] + 500. / 1113. * k_theta[3] \
        + 125. / 192. * k_theta[4] - 2187. / 6784. * k_theta[5] + 11. / 84. * k_theta[6])
    y_K[7] = y_K[1] + dt * (
        35. / 384. * k_K[1] - 0. * k_K[2] + 500. / 1113. * k_K[3] \
        + 125. / 192. * k_K[4] - 2187. / 6784. * k_K[5] + 11. / 84. * k_K[6])
    (k_theta[7], k_K[7]) = RHS(comm, rank, size, linear_theta, linear_K, all_limits, X, t, avT, avK, y_theta[7], y_K[7], utheta, uK, tau1, tau2)

    a_theta_star = a_theta + dt * (5179. / 57600. * k_theta[1] + 0. * k_theta[2] + 7571. / 16695. * k_theta[3] \
                                   + 393. / 640. * k_theta[4] - 92097. / 339200. * k_theta[5] \
                                   + 187. / 2100. * k_theta[6] + 1. / 40. * k_theta[7])
    a_K_star = a_K + dt * (5179. / 57600. * k_K[1] + 0. * k_K[2] + 7571. / 16695. * k_K[3] \
                           + 393. / 640. * k_K[4] - 92097. / 339200. * k_K[5] \
                           + 187. / 2100. * k_K[6] + 1. / 40. * k_K[7])

    a_theta += dt * (35. / 384. * k_theta[1] + 0. * k_theta[2] + 500. / 1113. * k_theta[3] \
                     + 125. / 192. * k_theta[4] - 2187. / 6784. * k_theta[5] + 11. / 84. * k_theta[6] + 0. * k_theta[7])
    a_K += dt * (35. / 384. * k_K[1] + 0. * k_K[2] + 500. / 1113. * k_K[3] \
                 + 125. / 192. * k_K[4] - 2187. / 6784. * k_K[5] + 11. / 84. * k_K[6] + 0. * k_K[7])

    eT += a_theta - a_theta_star
    eK += a_K - a_K_star

