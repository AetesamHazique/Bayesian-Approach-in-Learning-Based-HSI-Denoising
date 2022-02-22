from sklearn.feature_extraction import image
import numpy as np
from numpy.random import seed
from numpy.random import randint
import torch.nn

def mri_2d_extract(array3d,patch_size=40,max_patch_per_layer=10):
    # Extract total number of 2D images
    slices=array3d.shape[2]
    print(slices)
    print(array3d.shape)
    #patches=image.extract_patches_2d(array3d[:,:,0], (patch_size, patch_size),max_patches=max_patch_per_layer)
    if isinstance(patch_size,tuple):
        patches=image.extract_patches_2d(array3d[:,:,0], (patch_size[0], patch_size[1]),max_patches=1)
        print(patches.shape)
        for i in range(1,slices):
            #patch = image.extract_patches_2d(array3d[:,:,i], (patch_size, patch_size),max_patches=max_patch_per_layer)
            patch = image.extract_patches_2d(array3d[:,:,i], (patch_size[0], patch_size[1]),max_patches=1)
            #print(patch.shape)
            #print(type(patch))
            patches=np.concatenate([patches,patch])
            #print(type(patches))
    else:
        patches=image.extract_patches_2d(array3d[:,:,0], (patch_size, patch_size),max_patches=max_patch_per_layer)
        print(patches.shape)
        for i in range(1,slices):
            #patch = image.extract_patches_2d(array3d[:,:,i], (patch_size, patch_size),max_patches=max_patch_per_layer)
            patch = image.extract_patches_2d(array3d[:,:,i], (patch_size, patch_size),max_patches=max_patch_per_layer)
            #print(patch.shape)
            #print(type(patch))
            patches=np.concatenate([patches,patch])
            #print(type(patches))
    return patches

def mri_3d_extract(array3d,patch_size=40,channels=1):
    x = torch.from_numpy(array3d)
    #x=x.reshape(1,x.shape[0],x.shape[1],x.shape[2])
    #patches = x.unfold(2, patch_size,patch_size).unfold(1, patch_size, patch_size).unfold(0, channels, channels)
    patches = x.unfold(2, channels,channels).unfold(1, patch_size, patch_size).unfold(0, patch_size, patch_size)
    patches = patches.contiguous().view(-1, patch_size, patch_size, channels)
    #patches = patches.contiguous().view(-1, channels,patch_size, patch_size )
    patches=patches.numpy()
    return patches

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
