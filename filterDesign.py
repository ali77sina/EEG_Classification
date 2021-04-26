# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 17:02:37 2020

@author: Sina
"""
import numpy as np
import matplotlib.pyplot as plt
from pywt import wavedec
from pywt import dwt_max_level
from pywt import dwt
from pywt import threshold_firm
from pywt import idwt
from pywt import waverec
from pywt import threshold
from scipy.fft import fft
from scipy import signal
from scipy.integrate import quad



def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def Implement_Notch_Filter(time, band, freq, ripple, order, filter_type, data):
    from scipy.signal import iirfilter,lfilter
    fs   = 1/time
    nyq  = fs/2.0
    low  = freq - band/2.0
    high = freq + band/2.0
    low  = low/nyq
    high = high/nyq
    b, a = iirfilter(order, [low, high], rp=ripple, btype='bandstop',
                     analog=False, ftype=filter_type)
    filtered_data = lfilter(b, a, data)
    return filtered_data

def Implement_Bandpass_Filter(time, band, freq, ripple, order, filter_type, data):
    from scipy.signal import iirfilter,lfilter
    fs   = 1/time
    nyq  = fs/2.0
    low  = freq - band/2.0
    high = freq + band/2.0
    low  = low/nyq
    high = high/nyq
    b, a = iirfilter(order, [low, high], rp=ripple, btype='band',
                     analog=False, ftype=filter_type)
    filtered_data = lfilter(b, a, data)
    return filtered_data

def fastFourier(N,T,y):
    print('y is actually ')
    print(type(y))
    # Number of sample points = N
    # sample spacing = T
    #sample values = y
    x = np.linspace(0.0, N*T, N)
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    plt.subplot(211)
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    # plt.subplot(212)
    # plt.plot(y)
    #centered_data = meanTran(y)
    # plt.subplot(221)
    plt.subplot(212)
    plt.plot(y)
    plt.grid()
    plt.show()
    
def psd(sig):
    f, Pxx_den = signal.welch(sig, 1000, nperseg=len(sig))
    plt.semilogy(f, Pxx_den)
    #plt.ylim([0.5e-3, 1])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()
    return [f, Pxx_den]
    
def band_power(y, lower, higher, dx):
    xx = np.trapz(y[lower:higher], dx = dx)
    return xx