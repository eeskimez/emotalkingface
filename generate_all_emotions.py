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
from models import GENERATOR
import cv2
import dlib
import sys
sys.path.insert(0, '../data_prep/')
from facealigner import FaceAligner
from helper import shape_to_np
from copy import deepcopy
import random as rn
from torchvision import transforms
import albumentations as A
from scipy.spatial import procrustes
from torch.distributions import normal
from datagen import gkern2
from datagen import to_categorical
from collections import defaultdict
from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim
# import cpbd
from PIL import Image
# os.environ["CUDA_VISIBLE_DEVICES"] = "" 
#-----------------------------------------#
#           Reproducible results          #
#-----------------------------------------#
os.environ['PYTHONHASHSEED'] = '999'
np.random.seed(999)
rn.seed(999)
torch.manual_seed(999)
#-----------------------------------------#

batchsize = 1

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-ih", "--h5-fold", type=str, help="input h5 folder")
parser.add_argument("-m", "--model", type=str, help="DNN model to use")
parser.add_argument("-p", "--pred-path", type=str, help="Predictor Path", default='data/shape_predictor_68_face_landmarks.dat')
parser.add_argument("-ti", "--temp-img", type=str, help="Template face image path", default='data_prep/template_face.jpg')
parser.add_argument("-o", "--out-fold", type=str, help="output folder")
args = parser.parse_args()

predictor_path = args.pred_path        
output_path = args.out_fold
h5_folder = args.h5_fold

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

with open(os.path.join(args.model, 'args.txt'), 'r') as f:
    model_args = json.load(f)
    model_args.pop('model', None)

args.__dict__.update(model_args)

generator = GENERATOR(args, debug=True).to(args.device)
generator = nn.DataParallel(generator)
generator.load_state_dict(torch.load(os.path.join(args.model, 'generator.pt'), map_location="cuda" if args.cuda else "cpu"))

cnt = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

emotion_dict = {'ANG':0, 'DIS':1, 'FEA':2, 'HAP':3, 'NEU':4, 'SAD':5}
intensity_dict = {'XX':0, 'LO':1, 'MD':2, 'HI':3}
h5_file_list = []
for root, dirs, files in os.walk(h5_folder):
    for filename in files:
        if filename.endswith(('.h5', '.hdf5')):
            labels = os.path.splitext(filename)[0].split('_')
            emotion = emotion_dict[labels[2]]
            emotion_intensity = intensity_dict[labels[3]]
            h5_file_list.append((os.path.join(root, filename), emotion, emotion_intensity, labels[0]))
rn.shuffle(h5_file_list)

condition_dict = defaultdict(list)
for s_idx, (h5_file, emotion, emotion_intensity, s_id) in tqdm(enumerate(h5_file_list)):
    if emotion == 4:
        dset = h5py.File(os.path.join(h5_file), 'r')
        condition = dset['video'][0, :, :, :]
        condition_dict[s_id].append(condition)

def deNormImg(img):
    img = np.moveaxis(255.0*img, 0, 2).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def toNumpy(img):
    img = (255*(0.5+((img)/2.0))).astype('uint8')
    img = np.moveaxis(img, 0, 2)
    return img

def toNumpyGray(img):
    img = np.moveaxis(255*img, 1, 3)
    return img.astype(np.uint8)


normTransform = A.Compose([ 
                        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True)
                    ], p=1)                    
toTensorTrans = transforms.Compose([transforms.ToTensor()])

noise_sampler = normal.Normal(0, 1)

metrics = defaultdict(list)

emotion_dict = {0:'ANG', 1:'DIS', 2:'FEA', 3:'HAP', 4:'NEU', 5:'SAD'}

generator.eval()
cnt = 0
with torch.no_grad():
    for s_idx, (h5_file, emotion, emotion_intensity, s_id) in tqdm(enumerate(h5_file_list)):

        if emotion_intensity < 2:
            continue

        if not os.path.exists(os.path.join(output_path, s_id)):
            os.makedirs(os.path.join(output_path, s_id))

        print(h5_file)
        dset = h5py.File(os.path.join(h5_file), 'r')

        video = dset['video'][:, :, :, :]
        speech = dset['speech'][:]

        speech = speech / np.max(np.abs(speech))
  
        speech = np.reshape(speech, (1, 1, speech.shape[0]))
        speech_t = torch.from_numpy( speech ).float()

        condition = condition_dict[s_id][0]

        normed_c = normTransform(image=condition)
        image_c = normed_c['image']
        image_c = toTensorTrans(image_c)
        image_c = np.reshape(image_c, (1, 3, 128, 128))

        for emotion in range(len(emotion_dict)):
            emo_label = torch.from_numpy(to_categorical(emotion, num_classes=6)).float().unsqueeze(0)
        
            out, _ = generator(image_c.to(args.device), speech_t.to(args.device), emo_label.to(args.device))

            frame_list = []

            for f_idx in range(out.shape[1]):
                frame_list.append(toNumpy(out[0, f_idx, :, :, :].data.cpu().numpy()))

            pd_video = np.array(frame_list)
            pd_video = pd_video[:video.shape[0], ...]

            

            utils.write_video_cv(pd_video, speech[0, 0, :], 8000, os.path.join(output_path, s_id), str(s_idx)+'_pd_'+emotion_dict[emotion]+'.mp4', 25.0)
      

        
              


