# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 03:20:27 2021

@author: Sina
"""

import numpy as np 

# CSP: defined as calculating the set of spatial filters Wcsp:
#   
#                   Wcsp*C(class 1)*Wcsp.T
#   Wcsp = argmax --------------------------   
#                   Wcsp*C(class 2)*Wcsp.T
#
# C(class i) = class average covariance 
#
# C(class i) = 1/Q * Sum(Cq) from q = 1 to Q
#
#
#               Mq * Mq.T
#   Cq = ---------------------------
#           trace(Mq * Mq).T
#
#   Cc (class average cov) = C(class 1) + C(class 2) + ...
#   Cc (class average covariance) = Vc * λc * Vc.T
#
#   whitening tranform : 
#   P = λc^-0.5 * Vc.T

def covar(data):        #funtion to calculate covariance of a non-square matrix
    num = data@data.T
    val = 0
    for i in range(len(num)):
        val += num[i,i]
    if val!=0:
        return num/val
    else:
        raise ValueError('devided by 0, singular matrix')

def posAverageCov(eeg, label_array, lab):   #average class covariance
    length = 0
    pos_cov = np.zeros((eeg[0].shape[0],eeg[0].shape[0]))
    for i in range(len(eeg)):
        if label_array[i] == lab:
            pos_cov += covar(eeg[i])
            length += 1
        else:
            continue
    return pos_cov/length

def negAverageCov(eeg, label_array, lab):   #average covariance of other classes
    labels = set(label_array)               #done in a one-v-else manner
    for i in labels:
        if i == lab:
            labels.pop(i)
    neg_cov = np.zeros((eeg[0].shape[0],eeg[0].shape[0]))
    for i in labels:
        neg_cov += posAverageCov(eeg,i)
    return neg_cov/len(labels)

def compCov(eeg, label_array):      #computing the composite covariance
    labels = set(label_array)
    comp_cov = np.zeros((eeg[0].shape[0],eeg[0].shape[0]))
    for i in labels:
        comp_cov += posAverageCov(eeg, label_array, i)
    return comp_cov

def whitten(compCov):                       #Whitenning transform
    w,eigenVector = np.linalg.eig(compCov)
    eigenvalue = np.diag(w)
    P = np.linalg.inv(eigenvalue)**(0.5)@eigenVector
    return P

