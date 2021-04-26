# -*- coding: utf-8 -*-
"""
@author: Sina
"""

#simple function to plot CWT transformation of 1D time-series data

import obspy
from obspy.imaging.cm import obspy_sequential
from obspy.signal.tf_misfit import cwt
import numpy as np
import matplotlib.pyplot as plt

def scalogram(data,npts,dt,f_min,f_max,title):
    
    t = np.linspace(0, dt * npts, npts) #creating the time ticks
    
    
    scalogram = cwt(data, dt, 8, f_min, f_max)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    x, y = np.meshgrid(
        t,
        np.logspace(np.log10(f_min), np.log10(f_max), scalogram.shape[0]))
    
    ax.pcolormesh(x, y, np.abs(scalogram), cmap=obspy_sequential)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_yscale('log')
    ax.set_ylim(f_min, f_max)
    ax.set_title(title)
    plt.show()