from sklearn.feature_extraction import image
import numpy as np
from numpy.random import seed
from numpy.random import randint
import torch.nn

def add_gaussian_noise(array,snr=20):
    noisy=np.empty(array.shape)
    if isinstance(snr,list)==True:
        for i in range(0,array.shape[0]):
            #level=np.array(array,'uint8')
            r=randint(snr[0],snr[1],1)
            std=np.mean(array[i,:,:,0])/(10**(0.1*r))
            noise=np.random.normal(0, std,(array.shape[1],array.shape[2]))
            noisy[i,:,:,0] = array[i,:,:,0] + noise
    else:
        for i in range(0,array.shape[0]):
            r=snr
            std=np.mean(array[i,:,:,0])/(10**(0.1*r))
            noise=np.random.normal(0,std,(array.shape[1],array.shape[2]))
            noisy[i,:,:,0] = array[i,:,:,0] + noise
    return noisy

def add_impulse_noise(array,percent=10):
    noisy=np.empty(array.shape)
    if isinstance(percent,list)==True:
        for i in range(0,array.shape[0]):
            r=0.01*randint(percent[0],percent[1],1)
            noise=np.random.uniform(0, 1,(array.shape[1],array.shape[2]))
            noisy[i,:,:,0] = (array[i,:,:,0] + r*noise)/(1+r)
    else:
        for i in range(0,array.shape[0]):
            r=0.01*percent
            noise=np.random.uniform(0,1,(array.shape[1],array.shape[2]))
            noisy[i,:,:,0] = (array[i,:,:,0] + r*noise)/(1+r)
    return noisy

def add_gaussian_noise3D(array,snr=20):
    noisy=np.empty(array.shape)
    if isinstance(snr,list)==True:
        for i in range(0,array.shape[0]):
            #level=np.array(array,'uint8')
            r=randint(snr[0],snr[1],1)
            std=np.mean(array[i,:,:,:,0])/(10**(0.1*r))
            noise=np.random.normal(0, std,(array.shape[1], array.shape[2], array.shape[3]))
            noisy[i,:,:,:,0] = array[i,:,:,:,0] + noise
    else:
        for i in range(0,array.shape[0]):
            r=snr
            std=np.mean(array[i,:,:,:,0])/(10**(0.1*r))
            noise=np.random.normal(0,std,(array.shape[1],array.shape[2], array.shape[3]))
            noisy[i,:,:,:,0] = array[i,:,:,:,0] + noise
    return noisy

def add_impulse_noise3D(array,percent=10):
    noisy=np.empty(array.shape)
    if isinstance(percent,list)==True:
        for i in range(0,array.shape[0]):
            r=0.01*randint(percent[0],percent[1],1)
            noise=np.random.uniform(0, 1,(array.shape[1],array.shape[2],array.shape[3]))
            noisy[i,:,:,:,0] = (array[i,:,:,:,0] + r*noise)/(1+r)
    else:
        for i in range(0,array.shape[0]):
            r=0.01*percent
            noise=np.random.uniform(0,1,(array.shape[1],array.shape[2],array.shape[3]))
            noisy[i,:,:,:,0] = (array[i,:,:,:,0] + r*noise)/(1+r)
    return noisy
