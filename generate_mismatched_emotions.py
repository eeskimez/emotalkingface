import argparse
import math
import os
import random as rn
import shutil
import subprocess
import json

import h5py
import librosa
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import utils
from models import GENERATOR, DISCEMO
# import cv2
# import dlib
import sys
sys.path.insert(0, '../data_prep/')
from facealigner import FaceAligner
from helper import shape_to_np
from copy import deepcopy
import random as rn
from torchvision import transforms
import albumentations as A
from scipy.spatial import procrustes
from scipy.interpolate import CubicSpline
from torch.distributions import normal
from datagen import gkern2
from datagen import to_categorical
from collections import defaultdict
# os.environ["CUDA_VISIBLE_DEVICES"] = "" 

batchsize = 1

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-ih", "--h5-fold", type=str, help="input h5 folder")
parser.add_argument("-m", "--model", type=str, help="DNN model to use")
parser.add_argument("-p", "--pred-path", type=str, help="Predictor Path", default='data/shape_predictor_68_face_landmarks.dat')
parser.add_argument("-ti", "--temp-img", type=str, help="Template face image path", default='data_prep/template_face.jpg')
parser.add_argument("-o", "--out-fold", type=str, help="output folder")
parser.add_argument("--seed", type=int, default=142)
args = parser.parse_args()

#-----------------------------------------#
#           Reproducible results          #
#-----------------------------------------#
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
rn.seed(args.seed)
torch.manual_seed(args.seed)
#-----------------------------------------#

predictor_path = args.pred_path        
output_path = args.out_fold
h5_folder = args.h5_fold
model_path = args.model

with open(os.path.join(args.model, 'args.txt'), 'r') as f:
    model_args = json.load(f)
    model_args.pop('model', None)

if not os.path.exists(output_path):
    os.makedirs(output_path)
else:
    shutil.rmtree(output_path)
    os.mkdir(output_path)

args.offset = 10
args.increment = 8000/25.0
args.cuda = torch.cuda.is_available()
args.device = torch.device("cuda" if args.cuda else "cpu") 
args.kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

args.__dict__.update(model_args)
generator = GENERATOR(args, train=False, debug=True).to(args.device)

generator = nn.DataParallel(generator)

generator.load_state_dict(torch.load(os.path.join(model_path, 'generator.pt'), map_location="cuda" if args.cuda else "cpu"))

speaker_conditions = defaultdict(list)
cnt = 0
emotion_dict = {'ANG':0, 'DIS':1, 'FEA':2, 'HAP':3, 'NEU':4, 'SAD':5}
h5_file_list = []
for root, dirs, files in os.walk(h5_folder):
    for filename in files:
        if filename.endswith(('.h5', '.hdf5')):
            labels = os.path.splitext(filename)[0].split('_')
            emotion = emotion_dict[labels[2]]
            speaker = labels[0]
            if emotion == 4:
                dset = h5py.File(os.path.join(root, filename), 'r')
                speaker_conditions[speaker].append(dset['video'][0, :, :, :])
            h5_file_list.append((os.path.join(root, filename), emotion, speaker))
        
rn.shuffle(h5_file_list)

def deNormImg(img):
    img = np.moveaxis(255.0*img, 0, 2).astype(np.uint8)#np.moveaxis(255.0*(0.5 + (img/2)), 0, 2).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def toNumpy(img):
    img = (255*(0.5+((img)/2.0))).astype('uint8')
    img = np.moveaxis(img, 0, 2)
    return img#cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def toNumpyGray(img):
    # img = (255*(0.5+((img)/2.0))).astype('uint8')
    img = np.moveaxis(255*img, 1, 3)
    return img.astype(np.uint8)

normTransform = A.Compose([ 
                        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True)
                    ], p=1)                    
toTensorTrans = transforms.Compose([transforms.ToTensor()])

emotion_dict = {0:'ANG', 1:'DIS', 2:'FEA', 3:'HAP', 4:'NEU', 5:'SAD'}

noise_sampler = normal.Normal(0, 1)

generator.eval()
cnt = 0
with torch.no_grad():
    for s_idx, (h5_file, emotion, speaker) in tqdm(enumerate(h5_file_list)):
        if os.path.splitext(h5_file)[0].split('/')[-1].split('_')[-1] == 'HI':
            continue
        rnd_emo = np.random.randint(6, size=1)[0]
        while rnd_emo == emotion:
            rnd_emo = np.random.randint(6, size=1)[0]

        speaker = os.path.splitext(h5_file)[0].split('/')[-2]
                
        emo_label = torch.from_numpy(to_categorical(rnd_emo, num_classes=6)).float().unsqueeze(0)

        dset = h5py.File(os.path.join(h5_file), 'r')

        lmarks = dset['lmarks'][:, 48:, :]
        video = dset['video'][:, :, :, :]
        speech = dset['speech'][:]

        speech = speech / np.max(np.abs(speech))

        speech = np.reshape(speech, (1, 1, speech.shape[0]))
        speech_t = torch.from_numpy(speech).float()

        rnd_condition = np.random.randint(len(speaker_conditions[speaker]), size=1)[0]
        
        condition = speaker_conditions[speaker][rnd_condition]#video[0, :, :, :]

        normed_c = normTransform(image=condition)
        image_c = normed_c['image']
        image_c = toTensorTrans(image_c)
        image_c = np.reshape(image_c, (1, 3, 128, 128))
    
        out, _ = generator(image_c.to(args.device), speech_t.to(args.device), emo_label.to(args.device))
       

        frame_list = []
        for f_idx in range(out.shape[1]):
            frame_list.append(toNumpy(out[0, f_idx, :, :, :].data.cpu().numpy()))
        frame_list = np.array(frame_list)       
        filename = os.path.split(os.path.splitext(h5_file)[0])[-1]
        print(frame_list.shape)

        utils.write_video_cv(frame_list[:-args.offset, ...], speech[0, 0, :int(-args.offset*args.increment)], args.fs, output_path, '{}_{}_{}.mp4'.format(s_idx, emotion_dict[emotion], emotion_dict[rnd_emo]), 25.0)

      