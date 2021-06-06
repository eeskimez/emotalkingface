import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import imageio
import numpy as np
import argparse, os, fnmatch, shutil
from tqdm import tqdm
import cv2

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-i", "--in-file", type=str, help="input video file")
parser.add_argument("-o", "--out-fold", type=str, help="output folder")
args = parser.parse_args()

in_file = args.in_file
out_fold = args.out_fold

if not os.path.exists(out_fold):
    os.makedirs(out_fold)
else:
    shutil.rmtree(out_fold)
    os.mkdir(out_fold)

cap = cv2.VideoCapture(in_file)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for frm_cnt in tqdm(range(length)):
    ret, frame = cap.read()
    if frame is None:
        break
    cv2.imwrite(os.path.join(out_fold, 'frm_'+str(frm_cnt)+'.png'), frame)
