# -*- coding: utf-8 -*-
"""
@author: Sina
"""
# Simple tool to use to compare the denoising DWT can achieve for different applications

import pywt as pw
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.fft import fft
from scipy import signal

def butter_lowpass(cutoff, fs, order=5):    #butterworth lowpass filter coeffs
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
def butter_lowpass_filter(data, cutoff, fs, order=5):   #butterworth lowpass filter returned signal function
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_highpass(cutoff, fs, order=5):   #high pass filter coeffs
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
def butter_highpass_filter(data, cutoff, fs, order=5):  #high pass filter output
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def fastFourier(N,T,y,title):       #Fast Fourier Transform
    # Number of sample points = N
    # sample spacing = T
    #sample values = y
    x = np.linspace(0.0, N*T, N)
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    plt.subplot(211)
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    plt.title(title)
    #plt.ylim(0,2.5)
    plt.subplot(212)
    plt.plot(y)
    plt.grid()
    plt.show()
    return xf,abs(yf)

names = pw.wavelist(kind='discrete')    #different mother wavelets supported by pywt

#creating the sample signals
t = np.linspace(0,1,1000)
x = 0.001*np.sin(2*np.pi*10*t)
xr = 0.001*np.sin(2*np.pi*10*t)
extra = np.sin(2*np.pi*20*t)
for i in range(1000):   #loop to add random noise to data
    x[i] += (np.random.random()*2-1)*0.1

#plt.subplot(212)
plt.plot(t,x)
plt.plot(t,xr,'r')
plt.title('raw')
plt.show()

filraw = butter_lowpass_filter(x, 15, 1000,order=5)
filraw = butter_highpass_filter(filraw, 5, 1000, order=5)

plt.plot(t,filraw)
plt.plot(t,xr,'r')
plt.title('raw+filter')
plt.show()

motherWave = 'sym9'
l = pywt.dwt_max_level(1000, motherWave)
coeffs = pywt.wavedec(x, motherWave, level=l)
for i in range(l):
    coeffs[i] = pywt.threshold_firm(coeffs[i], value_low=0.1, value_high=1)
rec = pywt.waverec(coeffs, motherWave)

#plt.subplot(211)
plt.plot(t,xx)
plt.plot(t,xr,'r')
plt.title('re-con')
plt.show()


fildwt = butter_lowpass_filter(rec, 15, 1000,order=5)
fildwt = butter_highpass_filter(fildwt, 5, 1000, order=5)

plt.plot(t,fildwt)
plt.plot(t,xr,'r')
plt.title('dwt+filter')
plt.show()


#calculating error
e1 = 0
for i in range(1000):
    e1 += (filraw[i]-xr[i])**2
e2 = 0
for i in range(1000):
    e2 += (fildwt[i]-xr[i])**2

    
print('vale for error in raw+filter noisu is ' + str(e1))
print('value for eror in recon+filter is ' + str(e2))
