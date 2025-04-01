# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 19:28:55 2021

@author: Sina
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy import signal
from obspy.signal.tf_misfit import cwt
from obspy.imaging.cm import obspy_sequential
import tensorflow as tf
import pywt

# hello 

#CNN model
def generate_model():
    model = tf.keras.Sequential([
    # first convolutional layer
    tf.keras.Input(shape=(64,100,3000)),
    tf.keras.layers.Conv2D(3, (3, 3), strides=(1, 1), activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=1, strides=1),
    
    # second convolutional layer
    tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    
    #fully connected classifier
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid') #2 outputs
    ])
    return model


mod = generate_model()
mod.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


#Signal Processing Functions

def perform_cwt(signal1, npts, dt):
    f_min = 1
    f_max = 50
    scalogram = cwt(signal1, dt, 8, f_min, f_max)
    return np.abs(scalogram)

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def fastFourier(N,T,y,title):
    # Number of sample points = N
    # sample spacing = T
    #sample values = y
    x = np.linspace(0.0, N*T, N)
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    plt.subplot(211)
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    plt.title(title)
    plt.ylim(0,1)
    plt.subplot(212)
    plt.plot(y)
    plt.grid()
    plt.show()

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

def baseLine(x):
    m = np.mean(x)
    for i in range(len(x)):
        x[i] = x[i] - m
    return x



def extAlpha(x):    #extracting alpha band data
    x = Implement_Notch_Filter(1/1000, 3, 50, 0.1, 3, 'butter', x)
    x = butter_lowpass_filter(x, 8, 1000)
    x = butter_highpass_filter(x, 13, 1000)
    return x

def extBeta(x): #extracting beta band data
    x = Implement_Notch_Filter(1/1000, 3, 50, 0.1, 3, 'butter', x)
    x = butter_lowpass_filter(x, 14, 1000)
    x = butter_highpass_filter(x, 32, 1000)
    return x


#############################################################################
#The third BCI competition data is used in this script
if __name__ == "__main__":
    f = open('trainData.txt', 'r')
    data = f.read()
    data = data.split('\n')
    data.pop(-1)
    f.close()
    
    data_2 = []
    
    for i in range(len(data)):
        temp = data[i].split('  ')
        temp.pop(0)
        for j in range(len(temp)):
            temp[j] = float(temp[j])
        data_2.append(temp)
    data_2 = np.array(data_2)
    data_2 = np.array_split(data_2, 278)
    data_2 = np.array(data_2)
    
    f = open('trainLab.txt', 'r') 
    label = f.read()
    label = label.split('\n')
    label.pop(-1)
    label = np.array(label, dtype=np.int32)
    f.close()
    
    #converting the labels into one hot vector
    hotVec = []
    for i in label:
        if i == -1:
            hotVec.append([0,1])
        else:
            hotVec.append([1,0])
    
    hotVec = np.array(hotVec)     
    # mod.fit(data_2, hotVec)
            
    alpha_band = []
    for i in range(278):
        temp = []
        for j in range(64):
            temp.append(extAlpha(data_2[i][j]))
        alpha_band.append(temp)
    alpha_band = np.array(alpha_band)
    
    #cwt vector is to be constructed as feature vector
    cwtVec = []
    for i in range(100):
        temp = []
        for j in range(64):
            temp.append(perform_cwt(alpha_band[i][j], 3000, 1/1000))
            print("cwt "+str(i) + ' out of 100 channel ' + str(j) + ' out of 64')
        cwtVec.append(temp)
    
    cwtVec = np.array(cwtVec)
    #only doing 250 to see the perfoermence as it consumes time
    history = mod.fit(alpha_band[0:250], hotVec[0:250], epochs = 10, batch_size = 50)   
    


##############################################################################
