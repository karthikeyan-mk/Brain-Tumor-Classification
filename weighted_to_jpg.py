import numpy as np
import cv2 
import os
from glob import glob
import matplotlib.pyplot as plt
import h5py
import scipy.io
from PIL import Image
import scipy.misc
import imageio

# To change from .mat to .jpeg format
f = h5py.File('data/1.mat','r')
data = f.get('cjdata/image')
data = np.array(data)

plt.imshow(data)
plt.show()

#scipy.misc.imsave('sample.jpg', data)


imageio.imwrite('sample.jpg', data)

# TO name the images as their respective classes
cnt1 = 0
cnt2 = 0
cnt3 = 0
for i in range(3064):
    if i == 0: 
        continue
    
    f = h5py.File('data/'+str(i)+'.mat','r')
    data = f.get('cjdata/image')
    data = np.array(data)
    
    label = f.get('cjdata/label')
    
    if(label.value == 1):
        imageio.imwrite('images/1/'+'Meningioma'+str(cnt1)+'.jpg', data)
        cnt1 = cnt1 + 1
    
    elif(label.value == 2):
        imageio.imwrite('images/2/'+'Glioma'+str(cnt2)+'.jpg', data)
        cnt2 = cnt2 + 1
        
    elif(label.value == 3):
        imageio.imwrite('images/3/'+'Pituitary Tumor'+str(cnt3)+'.jpg', data)
        cnt3 = cnt3 + 1