import argparse
import json
import math
import os
import random as rn
import shutil
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torch.distributions import normal
from torchvision import transforms

import models
import utils
import datagen
import trainer

def initParams():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--in-path", type=str, help="Input folder containing train data", default=None, required=True)
    parser.add_argument("-v", "--val-path", type=str, help="Input folder containing validation data", default=None, required=True)
    parser.add_argument("-o", "--out-path", type=str, help="output folder", default='../models/def', required=True)

    parser.add_argument("-m", "--model", type=str, help="Pre-trained model path", default=None)
    parser.add_argument("-mde", "--model_disc_emo", type=str, help="Pre-trained model path", default=None)
    parser.add_argument("-mdf", "--model_disc_frame", type=str, help="Pre-trained model path", default=None)
    
    parser.add_argument('--fs', type=int, default=8000)
    parser.add_argument('--fps', type=float, default=25.0)
    parser.add_argument('--num_frames', type=int, default=25)
    parser.add_argument('--context', type=int, default=17)

    parser.add_argument('--env_name', type=str, default='tface_emo')

    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)

    parser.add_argument('--lr_g', type=float, default=1e-05)
    parser.add_argument('--lr_pair', type=float, default=1e-05)
    parser.add_argument('--lr_frame', type=float, default=1e-06)
    parser.add_argument('--lr_emo', type=float, default=1e-06)
    parser.add_argument('--lr_video', type=float, default=1e-05)

    parser.add_argument("--gpu-no", type=str, help="select gpu", default='0')
    parser.add_argument('--seed', type=int, default=9)

    parser.add_argument('--disc_frame', type=float, default=None)
    parser.add_argument('--disc_pair', type=float, default=None)
    parser.add_argument('--disc_emo', type=float, default=None)
    parser.add_argument('--disc_video', type=float, default=None)

    parser.add_argument('--disc_frame_gp', type=float, help="Weight for gradient penalty of the frame discriminator.", default=10.0)
    parser.add_argument('--disc_emo_gp', type=float, help="Weight for gradient penalty of the emotion discriminator.", default=10.0)

    parser.add_argument('--disc_emo_weight', type=float, help="Weight for emotion losses in the emotion discriminator.", default=1000.0)
    
    parser.add_argument('--emo_weight', type=float, default=10)

    parser.add_argument('--plot_interval', type=int, default=10)

    parser.add_argument('--pre_train', type=bool, default=False)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no

    args.batch_size = args.batch_size * max(int(torch.cuda.device_count()), 1)
    args.increment = args.fs/args.fps
    args.img_dim = 512
    args.speech_dim = 512
    args.emo_dim = 512
    args.noise_dim = 128
    args.steplr = 200

    args.filters = [64, 128, 256, 512, 512]
    #-----------------------------------------#
    #           Reproducible results          #
    #-----------------------------------------#
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    rn.seed(args.seed)
    torch.manual_seed(args.seed)
    #-----------------------------------------#
   
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    else:
        shutil.rmtree(args.out_path)
        os.mkdir(args.out_path)

    if not os.path.exists(os.path.join(args.out_path, 'inter')):
        os.makedirs(os.path.join(args.out_path, 'inter'))
    else:
        shutil.rmtree(os.path.join(args.out_path, 'inter'))
        os.mkdir(os.path.join(args.out_path, 'inter'))

    with open(os.path.join(args.out_path, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    args.cuda = torch.cuda.is_available() 
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu") 
    args.kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}


    return args

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)

def enableGrad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def train():
    args = initParams()
    
    dsetContainer = datagen.DatasetContainer(args)
    # trainDset = nc.SafeDataset(dsetContainer.getTrainSet())
    trainDset = dsetContainer.getDset() 

    train_loader = torch.utils.data.DataLoader(trainDset,
                                               batch_size=args.batch_size, 
                                               shuffle=True,
                                               drop_last=True,
                                               **args.kwargs)
    
    dsetContainer_val = datagen.DatasetContainer(args, val=True)
    valDset = dsetContainer_val.getDset()
    val_loader = torch.utils.data.DataLoader(valDset,
                                               batch_size=4, 
                                               shuffle=True,
                                               drop_last=True,
                                               **args.kwargs)
    
    device_ids = list(range(torch.cuda.device_count()))

    generator = models.GENERATOR(args).to(args.device)
    generator.apply(init_weights)
    generator = nn.DataParallel(generator, device_ids)
    
    # Pair discriminator
    if args.disc_pair:
        disc_pair = models.DISCPAIRED(args).to(args.device)
        disc_pair.apply(init_weights)
        disc_pair = nn.DataParallel(disc_pair, device_ids)
    else:
        disc_pair = None

    if args.disc_frame:
        disc_frame = models.DISCFRAME(args).to(args.device)
        disc_frame.apply(init_weights)
        disc_frame = nn.DataParallel(disc_frame, device_ids)
    else:
        disc_frame = None

    if args.disc_video:
        disc_video = models.DISCVIDEO(args).to(args.device)
        disc_video.apply(init_weights)
        disc_video = nn.DataParallel(disc_video, device_ids)
    else:
        disc_video = None

    if args.disc_emo:
        disc_emo = models.DISCEMO(args).to(args.device)
        disc_emo.apply(init_weights)
        disc_emo = nn.DataParallel(disc_emo, device_ids)
    else:
        disc_emo = None

    if args.model:
        generator.load_state_dict(torch.load(os.path.join(args.model, 'generator.pt'), map_location="cuda" if args.cuda else "cpu"), strict=True)
        print('Generator loaded...')
    if args.model_disc_emo:
        disc_emo.load_state_dict(torch.load(os.path.join(args.model_disc_emo, 'disc_emo.pt'), map_location="cuda" if args.cuda else "cpu"), strict=True)
        print('Disc emo loaded...')
    if args.model_disc_frame:
        disc_frame.load_state_dict(torch.load(os.path.join(args.model_disc_frame, 'disc_frame.pt'), map_location="cuda" if args.cuda else "cpu"), strict=True)
        print('Disc frame loaded...')
    
    tface_trainer = trainer.tfaceTrainer(args, 
                         generator=generator,
                         disc_frame=disc_frame,
                         disc_pair=disc_pair,
                         disc_emo=disc_emo,
                         disc_video=disc_video,
                         train_loader=train_loader,
                         val_loader=val_loader)
    
    if args.pre_train:
        tface_trainer.pre_train()
    else:
        tface_trainer.train()

if __name__ == "__main__":
    train()
