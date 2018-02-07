import numpy as np
import copy
import os


# data=np.loadtxt('/home/yanez/src/python/pod_raileigh_14/results/kk.dat')
#
# x=data[:,-3]
# y=data[:,-2]
# z=data[:,-1]

x=np.arange(0.0,2.0*np.pi,0.001)
y=np.arange(0.0,2.0*np.pi,0.001)
z=np.arange(0.0,2.0*np.pi,0.001)

ux=np.unique(x)
uy=np.unique(y)
uz=np.unique(z)

uux = np.linspace(np.min(ux), np.max(ux))
uuy = np.linspace(np.min(uy), np.max(uy))
uuz = np.linspace(np.min(uz), np.max(uz))
xx, yy, zz = np.meshgrid(uux, uuy, uuz, indexing='ij')

S=copy.deepcopy(xx)

n=0
# for i in range(ux.shape[0]):
#     for j in range(uy.shape[0]):
#         for k in range(uz.shape[0]):
#
#             S[i,j,k]=data[n,0]
#             n=n+1
#
# for i in range(ux.shape[0]):
#     for j in range(uy.shape[0]):
#         for k in range(uz.shape[0]):
#
#             S[i,j,k]=np.cos(xx)
#             n=n+1
S=np.cos(xx)

#fftdata=np.fft.fftn(S)*S.shape[0]*S.shape[1]*S.shape[2]
fftdata=np.fft.fftn(S)#*(S.shape[0]+S.shape[1]+S.shape[2])

der = np.zeros(fftdata.shape, dtype=np.complex_)

auxx=np.fft.fftfreq(fftdata.shape[0], np.max(uux)-np.min(uux))
auxy=np.fft.fftfreq(fftdata.shape[1], np.max(uuy)-np.min(uuy))
auxz=np.fft.fftfreq(fftdata.shape[2], np.max(uuz)-np.min(uuz))

auxx2=np.abs(auxx)
auxy2=np.abs(auxy)
auxz2=np.abs(auxz)

indexx=np.argwhere(auxx2>0.3*np.max(auxx2))[:,0]
indexy=np.argwhere(auxy2>0.3*np.max(auxy2))[:,0]
indexz=np.argwhere(auxz2>0.3*np.max(auxz2))[:,0]
#
# for i in indexx:
#     for j in indexy:
#         for k in indexz:
#             fftdata[i,j,k]=0.0+0.0j

aux=auxx



for i in range(fftdata.shape[0]):
    der[i]=-fftdata[i]*4.0*np.pi**2*aux[i]**2*S.shape[0]**2

#der = fftdata

#der=fftdata*4.0*np.pi**2*np.fft.fftfreq(fftdata.shape[0])**2

#derp=np.fft.ifftn(der)*S.shape[0]*S.shape[1]*S.shape[2]
#derp=np.fft.ifftn(der)#*(S.shape[0]+S.shape[1]+S.shape[2])
derp=np.fft.ifftn(der)

data_der=np.zeros((uux.shape[0]*uuy.shape[0]*uuz.shape[0], 4))

n=0
for i in range(uux.shape[0]):
    for j in range(uuy.shape[0]):
        for k in range(uuz.shape[0]):

            data_der[n,0]=np.real(derp[i,j,k])
            data_der[n,1]=uux[i]
            data_der[n,2]=uuy[j]
            data_der[n,3]=uuz[k]
            n=n+1

# data_der[:,1:4]=data[:,-3:]

np.savetxt('/home/yanez/src/python/pod_raileigh_14/results/der_kk.dat', data_der)

argumentop='python ./mode_plotter.py --file_pod der_kk.dat --path ./results --name pod_obtenido_kk --pod'
os.system(argumentop)
