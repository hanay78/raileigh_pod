import numpy as np
import copy
from physical_conditions import round_cell_size

def round_coordinates(coors, rnum):

    """coordinates o be located in rouded possitions"""
    cloc = copy.deepcopy(coors)
    return np.round(cloc / rnum) * rnum

class Neigbors:

    """routine that calculates the number of the cells located arround the cell considered"""

    def __init__(self):

        self._I = np.empty(0)
        self.I_ = np.empty(0)

        self._J = np.empty(0)
        self.J_ = np.empty(0)

        self._K = np.empty(0)
        self.K_ = np.empty(0)

    def save_in_hdf5_file(self, my_file, globalsize, rank):

        """save neighbouring cells in hdf5 files"""
        grc = my_file.create_group(self.__class__.__name__)

        _Igrc = grc.create_dataset("_I", (globalsize,))
        if rank == 0:
            _Igrc[:] = self._I[:]
        #grc.create_dataset("_I", data=self._I)

        I_grc = grc.create_dataset("I_", (globalsize,))
        if rank == 0:
            I_grc[:] = self.I_[:]
        #grc.create_dataset("I_", data=self.I_)

        _Jgrc = grc.create_dataset("_J", (globalsize,))
        if rank == 0:
            _Jgrc[:] = self._J[:]
        #grc.create_dataset("_J", data=self._J)

        J_grc = grc.create_dataset("J_", (globalsize,))
        if rank == 0:
            J_grc[:] = self.J_[:]
        #grc.create_dataset("J_", data=self.J_)

        _Kgrc = grc.create_dataset("_K", (globalsize,))
        if rank == 0:
            _Kgrc[:] = self._K[:]
        #grc.create_dataset("_K", data=self._K)

        K_grc = grc.create_dataset("K_", (globalsize,))
        if rank == 0:
            K_grc[:] = self.K_[:]
        #grc.create_dataset("K_", data=self.K_)

    def read_in_hdf5_file(self, my_file, ini, fin):

        """read neighbouring cells in hdf5 files"""

        grc = my_file[self.__class__.__name__]

        self._I = grc["_I"][ini:fin]
        self.I_ = grc["I_"][ini:fin]

        self._J = grc["_J"][ini:fin]
        self.J_ = grc["J_"][ini:fin]

        self._K = grc["_K"][ini:fin]
        self.K_ = grc["K_"][ini:fin]



class Boundary:

    """find out which cells are located in the boundary"""

    def __init__(self):
        self.blx = np.empty(0)
        self.bly = np.empty(0)
        self.blz = np.empty(0)

        self.brx = np.empty(0)
        self.bry = np.empty(0)
        self.brz = np.empty(0)

    def save_in_hdf5_file(self, my_file, rank):

        """save boundaries in hdf4 file"""

        grc = my_file.create_group(self.__class__.__name__)

        lx = grc.create_dataset("blx", (self.blx.shape[0],))
        ly = grc.create_dataset("bly", (self.bly.shape[0],))
        lz = grc.create_dataset("blz", (self.blz.shape[0],))

        rx = grc.create_dataset("brx", (self.brx.shape[0],))
        ry = grc.create_dataset("bry", (self.bry.shape[0],))
        rz = grc.create_dataset("brz", (self.brz.shape[0],))

        if rank == 0:
            lx[:] = self.blx[:]
            ly[:] = self.bly[:]
            lz[:] = self.blz[:]

            rx[:] = self.brx[:]
            ry[:] = self.bry[:]
            rz[:] = self.brz[:]

        #
        # grc.create_dataset("blx", data=self.blx)
        # grc.create_dataset("bly", data=self.bly)
        # grc.create_dataset("blz", data=self.blz)
        #
        # grc.create_dataset("brx", data=self.brx)
        # grc.create_dataset("bry", data=self.bry)
        # grc.create_dataset("brz", data=self.brz)

    def read_in_hdf5_file(self, my_file, ini, fin):

        """read boundaries in hdf4 file"""

        grc = my_file[self.__class__.__name__]

        self.blx = grc["blx"][ini:fin]
        self.bly = grc["bly"][ini:fin]
        self.blz = grc["blz"][ini:fin]

        self.brx = grc["brx"][ini:fin]
        self.bry = grc["bry"][ini:fin]
        self.brz = grc["brz"][ini:fin]


class Matrices:

    """grouping the matrices for the calculation of the simplified method"""

    def __init__(self):
    #    A = np.empty(0)
        self.B = np.empty(0)
        self.C = np.empty(0)
        self.Omega = np.empty(0)

    #    H = np.empty(0)
        self.J = np.empty(0)
        self.L = np.empty(0)
        self.Kappa = np.empty(0)

        self.E = np.empty(0)
        self.F = np.empty(0)
        self.Lambda = np.empty(0)

        self.E_star = np.empty(0)
        self.F_star = np.empty(0)
        self.Lambda_star = np.empty(0)

        self.E_prima = np.empty(0)
        self.F_prima = np.empty(0)
        self.Lambda_prima = np.empty(0)

        self.A = np.empty(0)
        self.S = np.empty(0)
        self.G = np.empty(0)
        self.H = np.empty(0)
        self.M = np.empty(0)
        self.N = np.empty(0)

        self.psi = np.empty(0)
        self.chi = np.empty(0)
        self.upsilon = np.empty(0)
        self.omicron = np.empty(0)

        self.alpha = np.empty(0)
        self.beta = np.empty(0)
        self.nu = np.empty(0)
        self.zeta = np.empty(0)
        self.eta = np.empty(0)
        self.iota = np.empty(0)
        self.phi = np.empty(0)
        self.mu = np.empty(0)



class Layers:

    """define which modes are low and high frequency"""

    def __init__(self):
        self.n_p = np.empty(0)
        self.n_p1 = np.empty(0)

        self.n_v = np.empty(0)
        self.n_v1 = np.empty(0)

        self.n_t = np.empty(0)
        self.n_t1 = np.empty(0)

    def set(self, n_pp, n_pp1, n_vv, n_vv1, n_tt, n_tt1):
        self.n_p = n_pp
        self.n_p1 = n_pp1

        self.n_v = n_vv
        self.n_v1 = n_vv1

        self.n_t = n_tt
        self.n_t1 = n_tt1

class Intexes:

    def __init__(self):
        self.comm = []
        self.size = []
        self.rank = []

        self.inew = np.empty(0)
        self.split_direction = np.empty(0)

        self.indexes_inner = []
        self.indexes_ghosh_cells = []
        self.indexes_ghosh_cells_minus_1 = []
        self.indexes_ghosh_cells_minus_2 = []

        self.patch_indexes_inner = []
        self.patch_indexes_ghosh_cells = []
        self.patch_indexes_ghosh_cells_minus_1 = []
        self.patch_indexes_ghosh_cells_minus_2 = []

        self.local_indices_inner = np.empty(0, dtype=np.int32)
        self.local_indices_ghosh_cells = np.empty(0, dtype=np.int32)
        self.local_indices_ghosh_cells_minus_1 = np.empty(0, dtype=np.int32)
        self.local_indices_ghosh_cells_minus_2 = np.empty(0, dtype=np.int32)

        self.global_indices_inner = np.empty(0, dtype=np.int32)
        self.global_indices_ghosh_cells = np.empty(0, dtype=np.int32)
        self.global_indices_ghosh_cells_minus_1 = np.empty(0, dtype=np.int32)
        self.global_indices_ghosh_cells_minus_2 = np.empty(0, dtype=np.int32)

    def set_csr(self, comm_, size_, rank_):
        self.comm = comm_
        self.size = size_
        self.rank = rank_

    def set_index(self, c):

        unique = [np.unique(c[:, 0]), np.unique(c[:, 1]), np.unique(c[:, 2])]

        mm = np.max([unique[0].shape[0], unique[1].shape[0], unique[2].shape[0]])

        div = [0, 1, 2]
        if mm == unique[1].shape[0]:
            div = [1, 2, 0]
        if mm == unique[2].shape[0]:
            div = [2, 0, 1]

        self.split_direction = div[0]
        self.inew = np.arange(c[:, div[0]].shape[0])

    def re_order(self, coors):

        c = round_coordinates(coors, round_cell_size)

        # c = copy.deepcopy(np.round(coors/0.000625)*0.000625)
        q = np.arange(0, c.shape[0], dtype=np.int32)

        jnew = np.zeros(c.shape[0], dtype=np.int32)
        jnew.fill(-1)

        unique = [np.unique(c[:, 0]), np.unique(c[:, 1]), np.unique(c[:, 2])]

        mm = np.max([unique[0].shape[0], unique[1].shape[0], unique[2].shape[0]])

        div = [0, 1, 2]
        if mm == unique[1].shape[0]:
            div = [1, 2, 0]
        if mm == unique[2].shape[0]:
            div = [2, 0, 1]

        n = 0
        for x in unique[div[0]]:
            ix = np.argwhere(c[:, div[0]] == x)

            if ix.size:
                cx = c[ix[:, 0]]
                qx = q[ix[:, 0]]

                for y in unique[div[1]]:
                    iy = np.argwhere(cx[:, div[1]] == y)

                    if iy.size:
                        cy = cx[iy[:, 0]]
                        qy = qx[iy[:, 0]]

                        for z in unique[div[2]]:
                            iz = np.argwhere(cy[:, div[2]] == z)

                            if iz.size:
                                jnew[n] = qy[iz[:, 0]]
                                n = n + 1
        self.split_direction = div[0]
        self.inew = jnew


#        return inew, div[0]



    def organize_split_vectors(self, coors_round, ghost_cells):


        ### unique indexes in the direction of splitting ###
        x = np.unique(coors_round[:, self.split_direction])

        ### number of phicial coordinates pro-patch ###
        vectsize = x.shape[0]
        nsi = vectsize / self.size
        lsi = vectsize % (self.size)

        fv = np.zeros(self.size, dtype=np.int32)
        fv.fill(nsi)
        fv[:lsi] = nsi + 1


        ### distribution of the coordinates pro patch ###
        gv = np.zeros(self.size + 1, dtype=np.int32)
        for i in range(1, gv.shape[0]):
            gv[i] = gv[i - 1] + fv[i - 1]


        ### phical coordinates in each patch ###
            ### global coordinates with ghosh cells ###
        tr = []
        if self.size > 1:
            tr.append(x[0:gv[1] + ghost_cells])
            for i in range(1, gv.shape[0] - 2):
                tr.append(x[gv[i] - ghost_cells:gv[i + 1] + ghost_cells])
            tr.append(x[gv[self.size - 1] - ghost_cells:gv[self.size]])
        else:
            tr.append(x[:])

            ### indexes that correspond to the coordinates ###
        for i in range(len(tr)):
            indexpach = np.empty(0, dtype=np.int32)
            for j in range(len(tr[i])):
                ix = np.argwhere(coors_round[:, self.split_direction] == tr[i][j])
                indexpach = np.append(indexpach, ix[:, 0])
            self.indexes_ghosh_cells.append(indexpach)


            ### global coordinates in inner domain ###
        trinnner = []
        for i in range(0, gv.shape[0] - 1):
            trinnner.append(x[gv[i]:gv[i + 1]])

            ### check if coordinates also in inner domain or in ghosh area ###
        tr_i = copy.deepcopy(tr)
        for i in range(len(tr_i)):
            for j in range(len(tr_i[i])):
                fll = True
                for k in range(len(trinnner[i])):
                    if tr_i[i][j] == trinnner[i][k]:
                        fll = False
                if fll == True:
                    tr_i[i][j] = -1

            ### indexes that corresponds to the coordinates in the inner domain ###

        for i in range(len(tr_i)):
            indexpach = np.empty(0, dtype=np.int32)
            for j in range(len(tr_i[i])):
                ix = np.argwhere(coors_round[:, self.split_direction] == tr[i][j])
                if tr_i[i][j] == -1:
                    menun = np.zeros(ix[:, 0].shape[0])
                    menun.fill(-1)
                    indexpach = np.append(indexpach, menun)
                else:
                    indexpach = np.append(indexpach, ix[:, 0])
            self.indexes_inner.append(indexpach)

            ### global coordinates with ghosh cells minus 1###
        tr_minus_1 = []
        if self.size > 1:
            tr_minus_1.append(x[0:gv[1] + ghost_cells-1])
            for i in range(1, gv.shape[0] - 2):
                tr_minus_1.append(x[gv[i] - ghost_cells+1:gv[i + 1] + ghost_cells-1])
            tr_minus_1.append(x[gv[self.size - 1] - ghost_cells+1:gv[self.size]])
        else:
            tr_minus_1.append(x[:])

           ### check if coordinates also in domain -1 or in ghosh area ###
        tr_i = copy.deepcopy(tr)
        for i in range(len(tr_i)):
            for j in range(len(tr_i[i])):
                fll = True
                for k in range(len(tr_minus_1[i])):
                    if tr_i[i][j] == tr_minus_1[i][k]:
                        fll = False
                if fll == True:
                    tr_i[i][j] = -1

            ### indexes that corresponds to the coordinates in the inner domain -1###

        for i in range(len(tr_i)):
            indexpach = np.empty(0, dtype=np.int32)
            for j in range(len(tr_i[i])):
                ix = np.argwhere(coors_round[:, self.split_direction] == tr[i][j])
                if tr_i[i][j] == -1:
                    menun = np.zeros(ix[:, 0].shape[0])
                    menun.fill(-1)
                    indexpach = np.append(indexpach, menun)
                else:
                    indexpach = np.append(indexpach, ix[:, 0])
            self.indexes_ghosh_cells_minus_1.append(indexpach)


            ### global coordinates with ghosh cells minus 2###
        tr_minus_2 = []
        if self.size > 1:
            tr_minus_2.append(x[0:gv[1] + ghost_cells-2])
            for i in range(1, gv.shape[0] - 2):
                tr_minus_2.append(x[gv[i] - ghost_cells+2:gv[i + 1] + ghost_cells-2])
            tr_minus_2.append(x[gv[self.size - 1] - ghost_cells + 2:gv[self.size]])
        else:
            tr_minus_2.append(x[:])

           ### check if coordinates also in domain -1 or in ghosh area ###
        tr_i = copy.deepcopy(tr)
        for i in range(len(tr_i)):
            for j in range(len(tr_i[i])):
                fll = True
                for k in range(len(tr_minus_2[i])):
                    if tr_i[i][j] == tr_minus_2[i][k]:
                        fll = False
                if fll == True:
                    tr_i[i][j] = -1

            ### indexes that corresponds to the coordinates in the inner domain -1###

        for i in range(len(tr_i)):
            indexpach = np.empty(0, dtype=np.int32)
            for j in range(len(tr_i[i])):
                ix = np.argwhere(coors_round[:, self.split_direction] == tr[i][j])
                if tr_i[i][j] == -1:
                    menun = np.zeros(ix[:, 0].shape[0])
                    menun.fill(-1)
                    indexpach = np.append(indexpach, menun)
                else:
                    indexpach = np.append(indexpach, ix[:, 0])
            self.indexes_ghosh_cells_minus_2.append(indexpach)

        self.distribute_indexes()
        self.local_indices()

    def split_vector(self, vector):

        vector_splited = []
        for i in range(len(self.indexes_ghosh_cells)):
            vector_splited.append(vector[self.indexes_ghosh_cells[i]])
        return vector_splited

    def distribute_indexes(self):
        self.patch_indexes_inner = np.array(self.comm.scatter(copy.deepcopy(self.indexes_inner),
                                                              root=0),
                                            dtype=np.int32)
        self.patch_indexes_ghosh_cells = np.array(self.comm.scatter(copy.deepcopy(self.indexes_ghosh_cells),
                                                                    root=0),
                                                  dtype=np.int32)
        self.patch_indexes_ghosh_cells_minus_1 = np.array(self.comm.scatter(copy.deepcopy(self.indexes_ghosh_cells_minus_1),
                                                                            root=0),
                                                          dtype=np.int32)
        self.patch_indexes_ghosh_cells_minus_2 = np.array(self.comm.scatter(copy.deepcopy(self.indexes_ghosh_cells_minus_2),
                                                                            root=0),
                                                          dtype=np.int32)

    def local_indices(self):
        #self.local_inner_indices = np.argwhere(self.patch_indexes_inner != -1)[:,0]
        self.local_indices_inner = np.argwhere(self.patch_indexes_inner != -1)[:, 0]
        self.local_indices_ghosh_cells = np.argwhere(self.patch_indexes_ghosh_cells != -1)[:, 0]
        self.local_indices_ghosh_cells_minus_1 = np.argwhere(self.patch_indexes_ghosh_cells_minus_1 != -1)[:, 0]
        self.local_indices_ghosh_cells_minus_2 = np.argwhere(self.patch_indexes_ghosh_cells_minus_2 != -1)[:, 0]

        self.global_indices_inner = self.patch_indexes_inner[self.local_indices_inner]
        self.global_indices_ghosh_cells = self.patch_indexes_ghosh_cells
        self.global_indices_ghosh_cells_minus_1 = self.patch_indexes_ghosh_cells_minus_1[self.local_indices_ghosh_cells_minus_1]
        self.global_indices_ghosh_cells_minus_2 = self.patch_indexes_ghosh_cells_minus_2[self.local_indices_ghosh_cells_minus_2]

    def get_local_inner_indices(self):
        return self.local_indices_inner


    def scatter_vector(self, vector):
        return np.array(self.comm.scatter(self.split_vector(vector), root=0))

class coordinates:
    def __init__(self):
        self.coors_dim_rounded = np.empty(0)
        self.coors_global_dim_rounded = np.empty(0)

        self.coors_dimless_rounded = np.empty(0)
        self.coors_global_dimless_rounded = np.empty(0)

    def make_coordinates_dimensioneless(self, L0):
        self.coors_dimless_rounded = self.coors_dim_rounded / L0
        self.coors_global_dimless_rounded = self.coors_global_dim_rounded / L0
