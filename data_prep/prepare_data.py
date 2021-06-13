import argparse
import os
import shutil

import h5py
# import librosa
import numpy as np
from tqdm import tqdm
from scipy.signal import decimate
from scipy import interpolate
from scipy import signal
from scipy.signal import butter, lfilter, freqz, wiener
from scipy.io import wavfile
import multiprocessing as mp
import utils
from facealigner import FaceAligner, crop_image
from helper import shape_to_np
import cv2
import dlib
import math
from multiprocessing import Process, Queue, Manager
import subprocess
import random as rn
from random import randint, shuffle
import time
from collections import deque
from scipy.interpolate import CubicSpline
#-----------------------------------------#
#           Reproducible results          #
#-----------------------------------------#
os.environ['PYTHONHASHSEED'] = '123'
np.random.seed(123)
rn.seed(123)
# torch.manual_seed(999)
#-----------------------------------------#
#----------------PARSER------------------#
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-i", "--input-folder", type=str, help='Path to folder that contains video files')
parser.add_argument("-p", "--pred-path", type=str, help="Predictor Path", default='data/dlib_data/shape_predictor_68_face_landmarks.dat')
parser.add_argument("-ti", "--temp-img", type=str, help="Template face image path", default='data_prep/template_face.jpg')
parser.add_argument("-fs", "--fs", type=int, help='sampling rate', default=8000)
parser.add_argument("-nw", "--nw", type=int, help='num workers', default=1)
parser.add_argument("-m", "--mode", type=int, help='Mode 0: Aligns all frames, Mode 1: Uses first frame to align all frames', default=1)
parser.add_argument("-d", "--debug", type=bool, help='Writes videos for debug purposes', default=False)
parser.add_argument("-o", "--out-path", type=str, help='Output path')
args = parser.parse_args()

in_path = args.input_folder
fs = args.fs
out_path = args.out_path
TI_path = args.temp_img
num_processes = args.nw
print("num_processes: ", num_processes)
predictor_path = args.pred_path

if not os.path.exists(out_path):
    os.makedirs(out_path)
else:
    shutil.rmtree(out_path)
    os.mkdir(out_path)

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


class Extractor():
    def __init__(self, in_path, mean_shape):
        self.dataQueue = Manager().Queue()
        self.fileList = []
        self.mean_shape = mean_shape
        for root, dirnames, filenames in os.walk(in_path):
            for filename in filenames:
                if os.path.splitext(filename)[1] == '.mp4' or os.path.splitext(filename)[1] == '.mpg' or os.path.splitext(filename)[1] == '.mov' or os.path.splitext(filename)[1] == '.flv':
                    self.fileList.append((root, filename))
        
        self.fileList = self.fileList[:]
        print(self.fileList)
        # exit()
        
    def processSample(self, process_id):
            import librosa
            n = len(self.fileList)
            increment = n // num_processes
            if process_id == num_processes-1:
                sampleList = list(range(process_id*increment, n))
            else:
                sampleList = list(range(process_id*increment, (process_id+1)*increment))
            print(process_id, len(sampleList), n,  process_id*increment, (process_id+1)*increment)

            # exit()

            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(predictor_path)
            desired_dim = 128
            fa = FaceAligner(predictor, desiredLeftEye=(0.25, 0.25), desiredFaceWidth=desired_dim)

            shuffle(sampleList)

            for j in tqdm(sampleList): 
                print(j)
                root, filename = self.fileList[j]
                
                y, sr = librosa.load(os.path.join(root, filename), sr=fs)

                try:
                    y = y-np.mean(y)
                    speech = y/np.max(np.abs(y))
                    if np.isnan(speech).any():
                        print('NaN encountered! Skipping file...')
                        continue
                except:
                    print('Exception! Skipping file...')
                    continue

                frame_list = []
                lmarks_list = []
                try:
                    cap = cv2.VideoCapture(os.path.join(root, filename))
                    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    print("Length: ", length)
                    dset = h5py.File(os.path.join(out_path, os.path.splitext(filename)[0]) + '.hdf5', 'w')
                    frame_cnt = 0
                    scale = None
                    video = []
                    tmp_lmarks = []
                    for frame_cnt in range(length):
                        ret, frame = cap.read()
                        if frame is None:
                            break
                        video.append(frame)
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        dets = detector(frame, 1)
                        for k, d in enumerate(dets):
                            tmp_lmarks.append(shape_to_np(predictor(gray, d)))

                    lmarks = np.array(tmp_lmarks)
                    
                    if args.mode == 1:
                        fa.get_tform(video[0], lmarks[0, ...], self.mean_shape, scale)
                    
                    for frame_cnt in range(length):
                        # frame, scale = fa.align_box(video[frame_cnt], lmarks[frame_cnt, ...], self.mean_shape, scale)
                        # frame, scale = fa.align_three_points(frame, np.average(np.array(buffer), axis=0, weights=[x/sum(list(range(buffersize))) for x in range(buffersize)]), self.mean_shape, scale)
                        if args.mode == 1:
                            frame, scale = fa.apply_tform(video[frame_cnt])
                        else:
                            frame, scale = fa.align_three_points(video[frame_cnt], lmarks[frame_cnt, ...], self.mean_shape, scale)

                       
                        frame_list.append(frame)
                        
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        dets = detector(frame, 1)
                        for k, d in enumerate(dets):
                            shape = shape_to_np(predictor(gray, d))
                        lmarks_list.append(shape)                       

                        frame_cnt+=1
                    
                    dset.create_dataset('speech', data=speech)
                    dset.create_dataset('video', data=np.array(frame_list))
                    dset.create_dataset('lmarks', data=np.array(lmarks_list))
                    self.dataQueue.put(('dummy', None))
                except:
                    os.remove(os.path.join(out_path, os.path.splitext(filename)[0]) + '.hdf5')            
                    print('Exception! Deleting file...', os.path.join(out_path, os.path.splitext(filename)[0]) + '.hdf5')
                    continue
                # utils.write_video_cv(np.array(frame_list), speech, 8000, '', '{}_test.mp5'.format(j), 25.0)
                # if j > 9:
                #     exit()
                if args.debug:
                    utils.write_video_cv(np.array(frame_list), speech, 8000, out_path, os.path.join(os.path.splitext(filename)[0] + '.mp4'), 25.0)

            self.dataQueue.put(('end', process_id))
            print('Thread Ended #', process_id)

    def writeToFile(self):
        cnt = 0
        threadStatus = [0] * num_processes
        pbar = tqdm(total = len(self.fileList))
        while True:
            if all(threadStatus):
                break
            data = self.dataQueue.get()
            if isinstance(data[0], str) and data[0] == 'end':
                print('End ', data[1])
                threadStatus[data[1]] = 1
                continue
            # speech_dset.create_dataset(data[0]+'_'+data[1], data=data[3])
            # image_seq_dset.create_dataset(data[0]+'_'+data[1], data=data[2])
            cnt += 1
            pbar.update(1)
            # utils.write_video(data[2], data[3], 8000, out_path, data[0]+'_'+data[1], 29.97)
        print('Main Ended.')
        # dset.close()
        pbar.close()

if __name__ == '__main__':
    TI_process = TemplateProcessor(TI_path)
    mean_shape = TI_process.getShape()
    del TI_process

    cnt = 0
    ext = Extractor(in_path, mean_shape)

    if num_processes < 2:
        ext.processSample(0)
    else:
        processes = []
        for i in range(num_processes):
            processes.append(Process(target=ext.processSample, args=(i, )))
        
        for i, p in enumerate(processes):
            # p.daemon = True
            p.start()
            print('Process #', i)
        
        p = Process(target=ext.writeToFile, args=())
        p.start()
        p.join()
        print('Main joined.')

        for i, p in enumerate(processes):
            p.join()
            print('Joined #', i)
    

