import os
import numpy as np


###this is a code created to make a couple of trisl with the ppossivility of making
### calculations of derivatives using fft

data = np.loadtxt('/home/yanez/src/python/pod_raileigh_14/results/kk.dat')

x = data[:, -3]*10.0
y = data[:, -2]*10.0
z = data[:, -1]*10.0

ux = np.unique(x)
uy = np.unique(y)
uz = np.unique(z)

xx, yy, zz = np.meshgrid(ux, uy, uz, indexing='ij')

S = np.zeros(xx.shape)

n = 0
for i in range(ux.shape[0]):
    for j in range(uy.shape[0]):
        for k in range(uz.shape[0]):

            S[i, j, k] = data[n, 0]
            #S[i, j, k] = np.cos(xx[i,j,k]/np.max(xx)*2.0*np.pi)
            n = n+1

# #### option A
# SS=np.zeros((2*(S.shape[0]), S.shape[1], S.shape[2]))
#
# SS[:(S.shape[0]),:,:]=S[:,:,:]
# SS[(S.shape[0]):,:,:]=S[::-1,:,:]
#
# #### option B
# SS=np.zeros((2*(S.shape[0]-2), S.shape[1], S.shape[2]))
#
# SS[:(S.shape[0]-2),:,:]=S[1:-1,:,:]
# SS[(S.shape[0]-2):,:,:]=S[-2:0:-1,:,:]

#### option C
SS = np.zeros((2*(S.shape[0]-2)+1, S.shape[1], S.shape[2]))

SS[:(S.shape[0]-2), :, :] = S[1:-1, :, :]
SS[(S.shape[0]-2+1):, :, :] = S[-2:0:-1, :, :]

# SS=np.zeros((S.shape[0], S.shape[1], S.shape[2]))
#
# SS[:S.shape[0],:,:]=S[:,:,:]


fftdata = np.fft.rfftn(SS)


auxx = np.fft.fftfreq(fftdata.shape[0], 2.0*np.max(ux)-np.min(ux))
auxy = np.fft.fftfreq(fftdata.shape[1], np.max(uy)-np.min(uy))
auxz = np.fft.fftfreq(fftdata.shape[2], np.max(uz)-np.min(uz))

# auxx2=np.abs(auxx)
# auxy2=np.abs(auxy)
# auxz2=np.abs(auxz)
#
# propor=0.5
#
# indexx=np.argwhere(auxx2>propor*np.max(auxx2))[:,0]
# indexy=np.argwhere(auxy2>propor*np.max(auxy2))[:,0]
# indexz=np.argwhere(auxz2>propor*np.max(auxz2))[:,0]

# indexx=np.arange(auxx.shape[0])
# indexy=np.arange(auxy.shape[0])
# indexz=np.arange(auxz.shape[0])

# for i in indexx:
#     for j in indexy:
#         for k in indexz:
#             fftdata[i,j,k]=0.0+0.0j

# for i in range(auxx.shape[0]):
#     for j in  range(auxy.shape[0]):
#         for k in indexz:
#             fftdata[i,j,k]=0.0+0.0j
#
# for i in range(auxx.shape[0]):
#     for j in indexy:
#         for k in range(auxz.shape[0]):
#             fftdata[i,j,k]=0.0+0.0j

# for i in indexx:
#     for j in range(auxy.shape[0]):
#         for k in range(auxz.shape[0]):
#             fftdata[i,j,k]=0.0+0.0j

aux = auxx

der = np.zeros(fftdata.shape, dtype=np.complex_)

for i in range(fftdata.shape[0]):
    der[i] = -fftdata[i]*4.0*np.pi**2*aux[i]**2*SS.shape[0]**2

derp = np.fft.irfftn(der)

data_der = np.zeros((x.shape[0], 4))

n = 0
for i in range(ux.shape[0]):
    for j in range(uy.shape[0]):
        for k in range(uz.shape[0]):

            data_der[n, 0] = np.real(derp[i, j, k])
            n = n+1

data_der[:, 1:4] = data[:, -3:]

np.savetxt('/home/yanez/src/python/pod_raileigh_14/results/der_kk.dat', data_der)

argumentop = 'python ./mode_plotter.py --file_pod der_kk.dat --path ./results --name pod_obtenido_double_kk --pod'
os.system(argumentop)
