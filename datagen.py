import os
import numpy as np
import h5py
import random as rn

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import librosa
import cv2
import albumentations as A
from torchvision import transforms
from scipy.spatial import procrustes
from scipy import signal
import scipy.ndimage.filters as fi

def gkern2(means, nsig=9):
    """Returns a 2D Gaussian kernel array."""
    
    inp = np.zeros((128, 128))
    
    if int(means[1]) > 127 or int(means[0]) > 127:
        inp[92, 92] = 1
    else:
        inp[int(means[1]), int(means[0])] = 1
    return fi.gaussian_filter(inp, nsig)

emotion_dict = {'ANG':0, 'DIS':1, 'FEA':2, 'HAP':3, 'NEU':4, 'SAD':5}
intensity_dict = {'XX':0, 'LO':1, 'MD':2, 'HI':3}

class DatasetContainer():
    def __init__(self, args, val=False):
        self.args = args
        self.filelist = []

        if not val:
            path = self.args.in_path
        else:
            path = self.args.val_path

        for root, dirnames, filenames in os.walk(path):
            for filename in filenames:
                if os.path.splitext(filename)[1] == '.hdf5':
                    labels = os.path.splitext(filename)[0].split('_')
                    emotion = emotion_dict[labels[2]]
                    
                    emotion_intensity = intensity_dict[labels[3]]
                    if val:
                        if emotion_intensity != 3:
                            continue
                    
                    self.filelist.append((root, filename, emotion, emotion_intensity))

        self.filelist = np.array(self.filelist)
        print('Num files: ', len(self.filelist))

    def getDset(self):
        return FaceDset(self.filelist, self.args)

class FaceDset(Dataset):

    def __init__(self, filelist, args):
        self.filelist = filelist
        self.args = args
        
        self.transform = transforms.Compose([transforms.ToTensor()])

        target = {}
        for i in range(1, self.args.num_frames):
            target['image' + str(i)] = 'image'

        self.augments = A.Compose([
                        A.RandomBrightnessContrast(p=0.2),    
                        A.RandomGamma(p=0.2),    
                        A.CLAHE(p=0.2),
                        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=0.2),  
                        A.ChannelShuffle(p=0.2), 
                        A.RGBShift(p=0.2),
                        A.RandomBrightness(p=0.2),
                        A.RandomContrast(p=0.2),
                        # A.HorizontalFlip(p=0.5),
                        A.GaussNoise(var_limit=(10.0, 50.0), p=0.25)
                    ], additional_targets=target, p=0.8)

        self.c_augments = A.Compose([A.GaussNoise(p=1),
            ], p=0.5)

        self.normTransform = A.Compose([ 
                        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True)
                    ], additional_targets=target, p=1)
        
    def __len__(self):
        return len(self.filelist)

    # def normFrame(self, frame):
    #     normTransform = self.normTransform(image=frame)
    #     frame = self.transform(normTransform['image'])
    #     return frame

    def normFrame(self, frame):
        normTransform = self.normTransform(image=frame)
        frame = normTransform['image']
        frame = np.moveaxis(frame, 2, 0)
        return torch.from_numpy(frame)

    def augmentVideo(self, video):
        args = {}
        args['image'] = video[0, :, :, :]
        for i in range(1, self.args.num_frames):
            args['image' + str(i)] = video[i, :, :, :]
        result = self.augments(**args)
        video[0, :, :, :] = result['image']
        for i in range(1, self.args.num_frames):
            video[i, :, :, :] = result['image' + str(i)]
        return video

    def __getitem__(self, idx):

        filename = self.filelist[idx]
        emotion = int(filename[2])
        emotion = to_categorical(emotion, num_classes=6)
        emotion_intensity = int(filename[3]) # We don't use this info
            
        filename = filename[:2]
       
        dset = h5py.File(os.path.join(*filename), 'r')
        try:
            idx = np.random.randint(dset['video'].shape[0]-self.args.num_frames, size=1)[0]
        except:
            return self.__getitem__(np.random.randint(len(self.filelist)-1, size=1)[0])
     
        video = dset['video'][idx:idx+self.args.num_frames, :, :, :]
        lmarks = dset['lmarks'][idx:idx+self.args.num_frames, 48:, :]
    
        lmarks = np.mean(lmarks, axis=1)
        video = self.augmentVideo(video)
        att_list = []
        video_normed = []
        for i in range(video.shape[0]):
            video_normed.append(self.normFrame(video[i, :, :, :]))
            att = gkern2(lmarks[i, :])
            att = att / np.max(att)
            att_list.append(att)
        video_normed = torch.stack(video_normed, 0)
        att_list = np.array(att_list)
        
        speech = dset['speech'][:]
        speech = speech/np.max(np.abs(speech))
        
        speech = speech[ int(idx*self.args.increment): int((idx+self.args.num_frames)*self.args.increment)]

        speech = np.reshape(speech, (1, speech.shape[0])) 
        
        if speech.shape[1] != self.args.increment*self.args.num_frames:
            return self.__getitem__(np.random.randint(len(self.filelist)-1, size=1)[0])
             

        return speech, video_normed, att_list, emotion


def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    # Example
    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical