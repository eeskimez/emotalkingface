import argparse
import math
import os
import random as rn
import shutil
import subprocess

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
from datagen import to_categorical
import json
# os.environ["CUDA_VISIBLE_DEVICES"] = "" 
seed = 7423
#-----------------------------------------#
#           Reproducible results          #
#-----------------------------------------#
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
rn.seed(seed)
torch.manual_seed(seed)
#-----------------------------------------#


class TemplateProcessor():
    def __init__(self, path):
        template_I = cv2.imread(path)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        fa = FaceAligner(predictor, desiredLeftEye=(0.25, 0.25), desiredFaceWidth=128)

        gray = cv2.cvtColor(template_I, cv2.COLOR_BGR2GRAY)

        dets = detector(template_I, 1)
        for k, d in enumerate(dets):
            shape = predictor(gray, d)
    
        template_I, scale = fa.align(template_I, gray, d, shape, None)

        gray = cv2.cvtColor(template_I, cv2.COLOR_BGR2GRAY)

        dets = detector(template_I, 1)
        for k, d in enumerate(dets):
            shape = predictor(gray, d)

        template_I = cv2.cvtColor(template_I, cv2.COLOR_BGR2RGB)
        utils.easy_show(template_I, 'template_face.png')

        scl = 0.6

        shape = shape_to_np(shape)
        shape -= np.tile(np.min(shape, axis=0), [68, 1])
        shape = ((128*(shape/np.tile(np.max(shape, axis=0), [68, 1])))*scl) + 128*((1-scl)/2)

        self.points = shape

    def getShape(self):
        return self.points

batchsize = 1

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-is", "--speech-file", type=str, help="input speech file", default=None)
parser.add_argument("-im", "--img-file", type=str, help="input image file", default=None)
parser.add_argument("-ih", "--hdf5-file", type=str, help="input hdf5 file", default=None)
parser.add_argument("-m", "--model", type=str, help="DNN model to use")
parser.add_argument("-p", "--pred-path", type=str, help="Predictor Path", default='data/dlib_data/shape_predictor_68_face_landmarks.dat')
parser.add_argument("-ti", "--temp-img", type=str, help="Template face image path", default='data_prep/template_face.jpg')
parser.add_argument("-o", "--out-fold", type=str, help="output folder")
parser.add_argument("--gpu", type=str, help="GPU index", default="1")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

predictor_path = args.pred_path
output_path = args.out_fold

TI_process = TemplateProcessor(args.temp_img)
mean_shape = TI_process.getShape()

if not os.path.exists(output_path):
    os.makedirs(output_path)
else:
    shutil.rmtree(output_path)
    os.mkdir(output_path)

with open(os.path.join(args.model, 'args.txt'), 'r') as f:
    model_args = json.load(f)
    model_args.pop('model', None)

args.__dict__.update(model_args)

args.fs = 8000
args.fps = 25.0
args.offset = 10
args.increment = args.fs/args.fps
args.cuda = torch.cuda.is_available()
args.device = torch.device("cuda" if args.cuda else "cpu") 
args.kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

generator = GENERATOR(args, debug=True).to(args.device)
generator = nn.DataParallel(generator)
generator.load_state_dict(torch.load(os.path.join(args.model, 'generator.pt'), map_location="cuda" if args.cuda else "cpu"))

cnt = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
fa = FaceAligner(predictor, desiredLeftEye=(0.25, 0.25), desiredFaceWidth=128)

offset = 100
cnt = 0
if args.speech_file:
    speech, sr = librosa.load(args.speech_file, sr=args.fs)
    speech = speech / np.max(np.abs(speech))


condition = None # lrpAdd to prevent error and keep previous one if error
if args.hdf5_file:
    dset = h5py.File(os.path.join(args.hdf5_file), 'r')
    condition = dset['video'][0, :, :, :]
    if not args.speech_file:
        speech = dset['speech'][:]
    sr = 8000
else:
    I = cv2.imread(args.img_file)
    #print(I)
    #exit() 
    if I.shape[0] != I.shape[1] or I.shape[1] != 128 or 9 == 9:
        I_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

        dets = detector(I, 1)       
        for k, d in enumerate(dets):
            shape = predictor(I_gray, d)
        I, scale = fa.align_three_points(I, shape_to_np(shape), mean_shape, None)

        condition = cv2.resize(I, (128, 128))
        condition_memory = condition # lrpAdd
        # I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    else:
        print('Image file is not valid...')
        condition = condition_memory #lrpAdd
        #exit()#lrpDisabled this line

def deNormImg(img):
    img = np.moveaxis(255.0*img, 0, 2).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def toNumpy(img):
    img = (255*(0.5+((img)/2.0))).astype('uint8')
    img = np.moveaxis(img, 0, 2)
    return img

def toNumpyGray(img):
    img = np.moveaxis(img, 0, 2)
    return img

normTransform = A.Compose([ 
                        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True)
                    ], p=1)                    
# toTensorTrans = transforms.Compose([transforms.ToTensor()])

noise_sampler = normal.Normal(0, 1)

emotion_dict = {0:'ANG', 1:'DIS', 2:'FEA', 3:'HAP', 4:'NEU', 5:'SAD'}

cv2.imwrite(os.path.join(args.out_fold, 'condition.png'), condition)

generator.eval()
cnt = 0
with torch.no_grad():
    speech = np.reshape(speech, (1, 1, speech.shape[0]))
    speech_t = torch.from_numpy(speech).float()

    normed_c = normTransform(image=condition)
    image_c = normed_c['image']

    # image_c = toTensorTrans(image_c)

    image_c = np.moveaxis(image_c, 2, 0)
    image_c = torch.from_numpy(image_c)
    image_c = np.reshape(image_c, (1, 3, 128, 128))

    for emo_i in range(6):
        emo_label = torch.from_numpy(to_categorical(emo_i, num_classes=6)).float().unsqueeze(0)
        print(emo_label)
        out, _ = generator(image_c.to(args.device), speech_t.to(args.device), emo_label.to(args.device))

        frame_list = []
        for f_idx in range(out.shape[1]):
            frame_list.append(toNumpy(out[0, f_idx, :, :, :].data.cpu().numpy()))
        frame_list = np.array(frame_list)

        utils.write_video_cv(frame_list[:-args.offset, ...], speech[0, 0, :int(-args.offset*args.increment)], sr, output_path, emotion_dict[emo_i]+'_generated.mp4', args.fps)
