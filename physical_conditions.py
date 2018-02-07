#!/usr/bin/env python

import numpy as np

#v0=np.float64(1.)
L0 = np.float64(0.1)
rho0 = np.float64(1.14)
lamb = np.float64(0.0269)
cp = np.float64(1000)
kappa = lamb/cp/rho0
t0 = L0*L0/kappa
v0 = L0/t0
p0 = rho0*v0*v0
T1 = np.float64(600.)
T0 = np.float64(300.)
nu = np.float64(0.00023525915) #mu or nu?????????
mu = rho0*nu
beta = np.float64(0.00001)

g = np.float64(74.)

St = L0/(v0*t0)
Re = rho0*v0*L0/mu
Re_inv = 1./Re
Pr = nu/kappa
Pr_inv = 1./Pr

Ra = rho0*rho0*g*beta*(T1-T0)*L0*L0*L0/mu/mu*Pr

RR = Ra/(Re*Re*Pr)

gvect = np.array([0, 1, 0], dtype=np.float64)

Tup_dimless = np.float64(0.)
Tdown_dimless = np.float64(1.)
ymax_dimless = np.float64(1.)
P00 = np.float64(100000.0) / p0

Ty = (Tup_dimless-Tdown_dimless)/ymax_dimless

round_cell_size = np.float64(0.000625)
number_ghosh_cells = np.int32(4)

#epsilon_sovolev = np.float64(0.27)


#Tini=np.float64(1.)
#Cini=np.float64(1.)

#Pe_h=np.float64(5.0)
#Pe_m=np.float64(5.0)
#Le=np.float64(1.0)
#Da=np.float64(0.875)
#nu=np.float64(0.840)
#gamma=np.float64(15.0)
#mu=np.float64(13.0)