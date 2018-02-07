import numpy as np
from mpi4py import MPI

class Vectors:

    """Class that is created to collect the variables in an  ordered way.
    Pressure Temperature and velocities are stored toghether
    in a convenient way."""
    def __init__(self):

        self.P = np.empty(0, dtype=np.float64)
        self.u = np.empty(0, dtype=np.float64)
        self.v = np.empty(0, dtype=np.float64)
        self.w = np.empty(0, dtype=np.float64)
        self.T = np.empty(0, dtype=np.float64)

        self._min_temperature = np.empty(0, dtype=np.float64)
        self._max_temperature = np.empty(0, dtype=np.float64)

        self._min_pressure = np.empty(0, dtype=np.float64)
        self._max_pressure = np.empty(0, dtype=np.float64)

        self._min_u = np.empty(0, dtype=np.float64)
        self._max_u = np.empty(0, dtype=np.float64)

        self._min_v = np.empty(0, dtype=np.float64)
        self._max_v = np.empty(0, dtype=np.float64)

        self._min_w = np.empty(0, dtype=np.float64)
        self._max_w = np.empty(0, dtype=np.float64)

        self.coors = np.empty(0, dtype=np.float64)

        self.temperature_up = np.empty(0, dtype=np.float64)
        self.temperature_down = np.empty(0, dtype=np.float64)
        self.ymax = np.empty(0, dtype=np.float64)
        self.pressure_00 = np.empty(0, dtype=np.float64)
        self.RR = np.empty(0, dtype=np.float64)

        self.factors = np.empty(0, dtype=np.float64)

    def zeros(self, sizen=0):
        """Intialize al variables to:
           ---a vector of equal size 0 (by default)
           ---to a vector of to sizen size."""
        self.P = np.zeros(sizen, dtype=np.float64)
        self.u = np.zeros(sizen, dtype=np.float64)
        self.v = np.zeros(sizen, dtype=np.float64)
        self.w = np.zeros(sizen, dtype=np.float64)
        self.T = np.zeros(sizen, dtype=np.float64)

    def zeros_shape(self, size_p, size_u, size_v, size_w, size_t):

        """initilaize vectors to a diferent size"""
        self.P = np.zeros(size_p, dtype=np.float64)
        self.u = np.zeros(size_u, dtype=np.float64)
        self.v = np.zeros(size_v, dtype=np.float64)
        self.w = np.zeros(size_w, dtype=np.float64)
        self.T = np.zeros(size_t, dtype=np.float64)

    def set(self, p_local, u_local, v_local, w_local, t_local):

        """Set vectors of the class to given vectors"""

        self.P = p_local
        self.u = u_local
        self.v = v_local
        self.w = w_local
        self.T = t_local

    def set_own_class(self, amplitudes):

        """Set all vectors to the vectors of a given instance"""

        self.P = amplitudes.P
        self.u = amplitudes.u
        self.v = amplitudes.v
        self.w = amplitudes.w
        self.T = amplitudes.T

    def append_own_class(self, instance_b):

        """append the vectors of a given instance"""

        self.P = np.append(self.P, instance_b.P)
        self.u = np.append(self.u, instance_b.u)
        self.v = np.append(self.v, instance_b.v)
        self.w = np.append(self.w, instance_b.w)
        self.T = np.append(self.T, instance_b.T)

    def append_own_class_lumped_separated(self, instance_b):

        """appedn the vectors of a given instance in terms of P-T-u.
        All other velocities are considered to be coupled with u"""

        self.P = np.append(self.P, instance_b.P)
        self.u = np.append(self.u, instance_b.u)
        self.T = np.append(self.T, instance_b.T)

    def append(self, pressure, velocity_u, velocity_v, velocity_w, temperature):

        """append to class vectors given separatedly"""
        self.P = np.append(self.P, pressure)
        self.u = np.append(self.u, velocity_u)
        self.v = np.append(self.v, velocity_v)
        self.w = np.append(self.w, velocity_w)
        self.T = np.append(self.T, temperature)

    def vstack_own_class(self, instance_b):

        """vstack all vectors of a given class"""

        self.P = np.vstack([self.P, instance_b.P])
        self.u = np.vstack([self.u, instance_b.u])
        self.v = np.vstack([self.v, instance_b.v])
        self.w = np.vstack([self.w, instance_b.w])
        self.T = np.vstack([self.T, instance_b.T])

    def vstack_own_class_lumped_separated(self, instance_b):

        """vstack all vectors of a given class.
        Velocities are coupled with u"""

        self.P = np.vstack([self.P, instance_b.P])
        self.u = np.vstack([self.u, instance_b.u])
        self.T = np.vstack([self.T, instance_b.T])

    def vstack(self, pressure, velocity_u, velocity_v, velocity_w, temperature):

        """vstak in an instance vectors given separatedly"""
        self.P = np.vstack([self.P, pressure])
        self.u = np.vstack([self.u, velocity_u])
        self.v = np.vstack([self.v, velocity_v])
        self.w = np.vstack([self.w, velocity_w])
        self.T = np.vstack([self.T, temperature])

    def vstack_vector(self, tes):

        """vstak same vector to all vectors"""
        self.P = np.vstack([tes, self.P])
        self.u = np.vstack([tes, self.u])
        self.v = np.vstack([tes, self.v])
        self.w = np.vstack([tes, self.w])
        self.T = np.vstack([tes, self.T])

    def vstack_vector_lumped(self, tes):
        """vstak same vector to all vectors
        considering velocities coupled"""
        self.P = np.vstack([tes, self.P])
        self.u = np.vstack([tes, self.u])
#        self.v = np.vstack([tes, self.v])
#        self.w = np.vstack([tes, self.w])
        self.T = np.vstack([tes, self.T])

    def make_dimensionless(self, velocity_0, temperature_0, temperature_1, pressure_0):

        """made vectors dimensionless"""

        self.u = self.u / velocity_0
        self.v = self.v / velocity_0
        self.w = self.w / velocity_0
        self.P = self.P / pressure_0
        self.T = (self.T-temperature_0)/(temperature_1-temperature_0)

    def make_dimensional(self, velocity_0, temperature_0, temperature_1, pressure_0):

        """make vectors dimensional"""
        self.u = self.u * velocity_0
        self.v = self.v * velocity_0
        self.w = self.w * velocity_0
        self.P = self.P * pressure_0
        self.T = self.T*(temperature_1-temperature_0)+temperature_0

    def make_error_dimensional(self, velocity_0, temperature_0, temperature_1, pressure_0):

        """Make errors dismensional. note temperature is not shifted to 300k"""
        self.u = self.u * velocity_0
        self.v = self.v * velocity_0
        self.w = self.w * velocity_0
        self.P = self.P * pressure_0
        self.T = self.T*(temperature_1-temperature_0)

    def transpose(self):

        """transpose all vectors"""
        self.u = self.u.transpose()
        self.v = self.v.transpose()
        self.w = self.w.transpose()
        self.P = self.P.transpose()
        self.T = self.T.transpose()

    def minmax(self, comm, size, rank):

        """calculate min and max of the vectors"""

        self._max_temperature = np.zeros(1, dtype=np.float64)
        self._min_temperature = np.zeros(1, dtype=np.float64)

        self._max_pressure = np.zeros(1, dtype=np.float64)
        self._min_pressure = np.zeros(1, dtype=np.float64)

        self._max_u = np.zeros(1, dtype=np.float64)
        self._min_u = np.zeros(1, dtype=np.float64)
        self._max_v = np.zeros(1, dtype=np.float64)
        self._min_v = np.zeros(1, dtype=np.float64)
        self._max_w = np.zeros(1, dtype=np.float64)
        self._min_w = np.zeros(1, dtype=np.float64)

        comm.Allreduce(np.max(self.T), self._max_temperature, op=MPI.MAX)
        comm.Allreduce(np.min(self.T), self._min_temperature, op=MPI.MIN)
        comm.Allreduce(np.max(self.P), self._max_pressure, op=MPI.MAX)
        comm.Allreduce(np.min(self.P), self._min_pressure, op=MPI.MIN)

        comm.Allreduce(np.max(self.u), self._max_u, op=MPI.MAX)
        comm.Allreduce(np.min(self.u), self._min_u, op=MPI.MIN)
        comm.Allreduce(np.max(self.v), self._max_v, op=MPI.MAX)
        comm.Allreduce(np.min(self.v), self._min_v, op=MPI.MIN)
        comm.Allreduce(np.max(self.w), self._max_w, op=MPI.MAX)
        comm.Allreduce(np.min(self.w), self._min_w, op=MPI.MIN)

    def reduce(self, amplitudes, comm, size, rank):

        """reduce values of all vectors"""

        comm.Reduce([amplitudes.T, MPI.DOUBLE], [self.T, MPI.DOUBLE], root=0, op=MPI.SUM)
        comm.Reduce([amplitudes.P, MPI.DOUBLE], [self.P, MPI.DOUBLE], root=0, op=MPI.SUM)
        comm.Reduce([amplitudes.u, MPI.DOUBLE], [self.u, MPI.DOUBLE], root=0, op=MPI.SUM)
        comm.Reduce([amplitudes.v, MPI.DOUBLE], [self.v, MPI.DOUBLE], root=0, op=MPI.SUM)
        comm.Reduce([amplitudes.w, MPI.DOUBLE], [self.w, MPI.DOUBLE], root=0, op=MPI.SUM)

    def calculate_factors(self):

        """calculate reescaling factors"""
        self.factors = np.array([self._max_pressure - self._min_pressure,
                                 self._max_u - self._min_u,
                                 self._max_v - self._min_v,
                                 self._max_w - self._min_w,
                                 self._max_temperature - self._min_temperature],
                                dtype=np.float64)
        self.factors = np.fabs(self.factors)

    def dot(self, instance_a, instance_b, indices):

        """make numpy dot in all vectors"""

        self.T = np.dot(instance_a.T[indices], instance_b.T[indices])
        self.u = np.dot(instance_a.u[indices], instance_b.u[indices])
        self.v = np.dot(instance_a.v[indices], instance_b.v[indices])
        self.w = np.dot(instance_a.w[indices], instance_b.w[indices])
        self.P = np.dot(instance_a.P[indices], instance_b.P[indices])

    def dot_lumped_separated(self, instance_a, instance_b):

        """make numpy dot considering velocities coupled"""
        self.T = np.dot(instance_a.T, instance_b.T)
        self.u = np.dot(instance_a.u, instance_b.u)
        self.v = np.dot(instance_a.v, instance_b.u)
        self.w = np.dot(instance_a.w, instance_b.u)
        self.P = np.dot(instance_a.P, instance_b.P)

    def difference_dataset_integration(self, ind, fields_local):

        """calculates differences between the vectors of two instances"""

        aux = Vectors()

        aux.T = self.T[:, ind] - fields_local.T
        aux.P = self.P[:, ind] - fields_local.P
        aux.u = self.u[:, ind] - fields_local.u
        aux.v = self.v[:, ind] - fields_local.v
        aux.w = self.w[:, ind] - fields_local.w

        return aux

    def __add__(self, other):
        aux = Vectors()
        aux.set(self.P + other.P,
                self.u + other.u,
                self.v + other.v,
                self.w + other.w,
                self.T + other.T)
        return aux

    def add_lumped_separated(self, d_a):

        """add two instances considering just T and u. u's are considering toghether"""

        self.u = self.u + d_a.u
        self.T = self.T + d_a.T

    def make_homogeneous_bc(self, coorsss, temperature_up, temperature_down, ymax, pressure_00, RR):

        """make boundry conditions homogeneous"""

        self.coors = coorsss.coors_dimless_rounded

        self.temperature_up = temperature_up
        self.temperature_down = temperature_down
        self.ymax = ymax
        self.pressure_00 = pressure_00
        self.RR = RR

        correction_t = ((temperature_up-temperature_down)/ymax*self.coors[:, 1]+temperature_down)
        correction_p = (pressure_00
                        + RR *
                        (0.5*(temperature_up-temperature_down) / ymax *
                         self.coors[:, 1] * self.coors[:, 1]
                         + temperature_down * self.coors[:, 1])
                       )


        self.T = (self.T.transpose() - correction_t).transpose()
        self.P = (self.P.transpose() - correction_p).transpose()

    def non_homogeneous_bc(self):

        """make non homogenaous boundary conditions"""
        ### check for repetitio of this mehtod
        self.T = self.T + ((self.temperature_up-self.temperature_down)
                           /
                           self.ymax*self.coors[:, 1]+self.temperature_down)
        self.P = self.P + (self.pressure_00
                           + self.RR * (0.5*(self.temperature_up-self.temperature_down) / self.ymax
                                        * self.coors[:, 1] * self.coors[:, 1]
                                        + self.temperature_down * self.coors[:, 1])
                          )

    def make_non_homogeneous_bc(self,
                                coorsss,
                                temperature_up,
                                temperature_down,
                                ymax,
                                pressure_00,
                                RR):

        """make non homogenaous boundary conditions"""
        ### check for repetitio of this mehtod

        coors = coorsss.coors_dimless_rounded

        correction_t = ((temperature_up-temperature_down)/ymax*coors[:, 1]+temperature_down)
        correction_p = (pressure_00
                        + RR * (0.5*(temperature_up-temperature_down)
                                /
                                ymax * coors[:, 1] * coors[:, 1]
                                + temperature_down * coors[:, 1])
                       )

        self.T = (self.T.transpose() + correction_t).transpose()
        self.P = (self.P.transpose() + correction_p).transpose()

    def save_in_hdf5_file(self, my_file, globalsize, global_indices_inner, local_indices_inner):

        """ save vectors in a hdf5 file"""
        grc = my_file.create_group(self.__class__.__name__)

        #grc.create_dataset("T", data=self.T)
        if len(self.T.shape) == 1:
            tgrc = grc.create_dataset("T", (globalsize,), dtype=np.float64)
            tgrc[global_indices_inner] = self.T[local_indices_inner]
        else:
            tgrc = grc.create_dataset("T", (globalsize, self.T.shape[1]), dtype=np.float64)
            tgrc[global_indices_inner, :] = self.T[local_indices_inner, :]
        #grc.create_dataset("T", ())
        if len(self.P.shape) == 1:
            pgrc = grc.create_dataset("P", (globalsize,), dtype=np.float64)
            pgrc[global_indices_inner] = self.P[local_indices_inner]
        else:
            pgrc = grc.create_dataset("P", (globalsize, self.P.shape[1]), dtype=np.float64)
            pgrc[global_indices_inner, :] = self.P[local_indices_inner, :]
        #grc.create_dataset("P", data=self.P)
        if len(self.u.shape) == 1:
            ugrc = grc.create_dataset("u", (globalsize,), dtype=np.float64)
            ugrc[global_indices_inner] = self.P[local_indices_inner]
        else:
            ugrc = grc.create_dataset("u", (globalsize, self.u.shape[1]), dtype=np.float64)
            ugrc[global_indices_inner, :] = self.u[local_indices_inner, :]
        #grc.create_dataset("u", data=self.u)
        if len(self.v.shape) == 1:
            vgrc = grc.create_dataset("v", (globalsize,), dtype=np.float64)
            vgrc[global_indices_inner] = self.v[local_indices_inner]
        else:
            vgrc = grc.create_dataset("v", (globalsize, self.v.shape[1]), dtype=np.float64)
            vgrc[global_indices_inner, :] = self.v[local_indices_inner, :]
        #grc.create_dataset("v", data=self.v)
        if len(self.w.shape) == 1:
            wgrc = grc.create_dataset("w", (globalsize,), dtype=np.float64)
            wgrc[global_indices_inner] = self.w[local_indices_inner]
        else:
            wgrc = grc.create_dataset("w", (globalsize, self.w.shape[1]), dtype=np.float64)
            wgrc[global_indices_inner, :] = self.w[local_indices_inner, :]

    def save_derivatives_in_hdf5_file(self,
                                      my_file,
                                      globalsize,
                                      global_indices_inner,
                                      local_indices_inner):

        """ save derivatives in a hdf5 file"""

        grc = my_file.create_group(self.__class__.__name__)

        tgrc = grc.create_dataset("T", (globalsize, self.T.shape[1], 3), dtype=np.float64)
        tgrc[global_indices_inner, :, :] = self.T[local_indices_inner, :, :]

        pgrc = grc.create_dataset("P", (globalsize, self.P.shape[1], 3), dtype=np.float64)
        pgrc[global_indices_inner, :, :] = self.P[local_indices_inner, :, :]

        ugrc = grc.create_dataset("u", (globalsize, self.u.shape[1], 3), dtype=np.float64)
        ugrc[global_indices_inner, :, :] = self.u[local_indices_inner, :, :]

        vgrc = grc.create_dataset("v", (globalsize, self.v.shape[1], 3), dtype=np.float64)
        vgrc[global_indices_inner, :, :] = self.v[local_indices_inner, :, :]

        wgrc = grc.create_dataset("w", (globalsize, self.w.shape[1], 3), dtype=np.float64)
        wgrc[global_indices_inner, :, :] = self.w[local_indices_inner, :, :]




    def read_in_hdf5_file(self, my_file, patch_indexes_ghosh_cells):

        """ read vectors from a hdf5 file"""
        grc = my_file[self.__class__.__name__]

        self.T = grc["T"][patch_indexes_ghosh_cells, :]
        self.P = grc["P"][patch_indexes_ghosh_cells, :]
        self.u = grc["u"][patch_indexes_ghosh_cells, :]
        self.v = grc["v"][patch_indexes_ghosh_cells, :]
        self.w = grc["w"][patch_indexes_ghosh_cells, :]


    def read_derivatives_in_hdf5_file(self, my_file, patch_indexes_ghosh_cells):

        """ read derivatives from a hdf5 file"""

        grc = my_file[self.__class__.__name__]

        self.T = grc["T"][patch_indexes_ghosh_cells, :, :]
        self.P = grc["P"][patch_indexes_ghosh_cells, :, :]
        self.u = grc["u"][patch_indexes_ghosh_cells, :, :]
        self.v = grc["v"][patch_indexes_ghosh_cells, :, :]
        self.w = grc["w"][patch_indexes_ghosh_cells, :, :]
