import numpy as np


data = np.loadtxt('/home/yanez/src/python/pod_raileigh_14/results/kk.dat')

x = np.arange(0, 2*np.pi, 0.001)

cx = np.cos(x)

fftdata = np.fft.fft(cx)#*np.size(x)
#fftdata=np.fft.fft(cx, norm="ortho")
#der = np.zeros(fftdata.shape)/np.size(x)
der = np.zeros(fftdata.shape)

#der=-fftdata*(np.fft.fftfreq(fftdata.shape[0], np.max(x)-np.min(x)))**2
der = -fftdata*4*np.pi**2*(np.fft.fftfreq(fftdata.shape[0], np.max(x)-np.min(x)))**2

#der=fftdata
#der=fftdata*(np.fft.fftfreq(fftdata.shape[0], np.max(x)-np.min(x)))**2
#der=-fftdata*4*np.pi**2*(np.fft.fftfreq(fftdata.shape[0], np.max(x)-np.min(x)))**2

derp = np.fft.ifft(der)*np.size(x)*np.size(x)
#derp=np.fft.ifftn(der, norm="ortho")

np.savetxt('kk_if.dat', np.vstack((x, np.real(derp))).transpose())
