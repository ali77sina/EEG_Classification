# -*- coding: utf-8 -*-
"""
@author: Sina
"""

#simple code to visualise MRI slices

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from obspy.imaging.cm import obspy_sequential
import os
import pydicom
aaa

# img = nib.load(r'C:\Users\Sina\Desktop\head images\mni_icbm152_t2_tal_nlin_asym_09c.mnc')
img = nib.load(r'C:\Users\Sina\Desktop\multiEEGMRI\ucl\sMRI\smri.img')
data = img.get_fdata()
print(data.mean())
print(data.max())
print(data.min())


for i in range(193):      #loop to show sliced photos
    plt.pcolormesh(data[i], cmap = 'gray')
    plt.show()

# added this for test
