# Written by Sefik Emre Eskimez, May 29 2018 - Aug 17 2018 #

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as manimation
import matplotlib.lines as mlines
from matplotlib import transforms
import numpy as np
import os
from tqdm import tqdm
import subprocess
import librosa
import cv2 
import scipy

font = {'size'   : 18}
mpl.rc('font', **font)

Mouth = [[48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57], \
         [57, 58], [58, 59], [59, 48], [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], \
         [66, 67], [67, 60]]

Nose = [[27, 28], [28, 29], [29, 30], [30, 31], [30, 35], [31, 32], [32, 33], \
        [33, 34], [34, 35], [27, 31], [27, 35]]

leftBrow = [[17, 18], [18, 19], [19, 20], [20, 21]]
rightBrow = [[22, 23], [23, 24], [24, 25], [25, 26]]

leftEye = [[36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [36, 41]]
rightEye = [[42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [42, 47]]

other = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], \
         [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], \
         [12, 13], [13, 14], [14, 15], [15, 16]]

faceLmarkLookup = Mouth + Nose + leftBrow + rightBrow + leftEye + rightEye + other

def write_video_cv(frames, speech, fs, path, fname, fps):
    # fname = os.path.splitext(fname)[0]
    print(os.path.join(path, fname))
    # exit()
    out = cv2.VideoWriter(os.path.join(path, fname), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (frames.shape[1], frames.shape[2]), (True if len(frames.shape) == 4 else False))
    # exit()
    if out.isOpened():
        for i in range(frames.shape[0]):
            out.write(frames[i, ...])
    out.release()
    # exit()
    # print(speech.shape)
    # scipy.io.wavfile.write(os.path.join(path, fname+'.wav'), fs, speech)
    librosa.output.write_wav(os.path.join(path, fname+'.wav'), speech, fs)

    cmd = 'ffmpeg -i '+os.path.join(path, fname)+' -i '+os.path.join(path, fname)+'.wav -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0  '+os.path.join(path, fname)+'_.mp4'
    subprocess.call(cmd, shell=True) 
    print('Muxing Done')

    os.remove(os.path.join(path, fname))
    os.remove(os.path.join(path, fname+'.wav'))

def easy_show(data, lab, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(data)
    plt.savefig(lab, dpi = 300, bbox_inches='tight')
    plt.clf()
    plt.close()

def easy_show_FLM(data, lmarks, lab, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(data)
    plt.plot(lmarks[:, 0], lmarks[:, 1], 'r*')
    plt.savefig(lab, dpi = 300, bbox_inches='tight')
    plt.clf()
    plt.close()

def STFT(speech, sr, winsize, hopsize):
    cnst = 1+(int(int(sr*winsize))/2)
    res_stft =librosa.stft(speech,
                            win_length = int(sr*winsize),
                            hop_length = int(sr*hopsize),
                            n_fft = int(sr*winsize))
    
    stft_mag = np.abs(res_stft)/cnst
    stft_phase = np.angle(res_stft)

    return stft_mag, stft_phase

def plot(data, label, min_val=None, max_val=None):
    if not min_val:
        min_val = np.min(data)
        max_val = np.max(data)
    fig = plt.figure(figsize=(10, 10))
    im = plt.imshow(data, cmap=plt.get_cmap('jet'), origin='lower', vmin=min_val, vmax=max_val)
    # fig.colorbar(im)
    plt.axis('off')
    plt.savefig(label, bbox_inches='tight')
    plt.clf()
    plt.close()

def subplot(data1, data2, label):
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(211)
    im = plt.imshow(data1, cmap=plt.get_cmap('jet'), origin='lower', vmin=np.min(data1), vmax=np.max(data1))
    plt.axis('off')

    plt.subplot(212)
    im = plt.imshow(data2, cmap=plt.get_cmap('jet'), origin='lower', vmin=np.min(data1), vmax=np.max(data1))
    plt.axis('off')

    plt.savefig(label, bbox_inches='tight')
    plt.clf()
    plt.close()

def write_video(frames, sound, fs, path, fname, fps, cmap='jet'):
    try:
        os.remove(os.path.join(path, fname+'.mp4'))
        os.remove(os.path.join(path, fname+'.wav'))
        os.remove(os.path.join(path, fname+'_ws.mp4'))
    except:
        print ('Exp')

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    fig = plt.figure(figsize=(10, 10))
    l = plt.imshow(frames[0, :, :], cmap=cmap)

    librosa.output.write_wav(os.path.join(path, fname+'.wav'), sound, fs)

    with writer.saving(fig, os.path.join(path, fname+'.mp4'), 150):
        # plt.gca().invert_yaxis()
        plt.axis('off')
        for i in tqdm(range(frames.shape[0])):
            l.set_data(frames[i, :, :])
            cnt = 0
            writer.grab_frame()

    cmd = 'ffmpeg -i '+os.path.join(path, fname)+'.mp4 -i '+os.path.join(path, fname)+'.wav -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0  '+os.path.join(path, fname)+'_.mp4'
    subprocess.call(cmd, shell=True) 
    print('Muxing Done')

    os.remove(os.path.join(path, fname+'.mp4'))
    os.remove(os.path.join(path, fname+'.wav'))

def write_video_FLM(frames, sound, fs, path, fname, xLim, yLim, fps=29.97):
    try:
        os.remove(os.path.join(path, fname+'.mp4'))
        os.remove(os.path.join(path, fname+'.wav'))
        os.remove(os.path.join(path, fname+'_ws.mp4'))
    except:
        print ('Exp')

    if len(frames.shape) < 3:
        frames = np.reshape(frames, (frames.shape[0], frames.shape[1]/2, 2))
    # print frames.shape

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    fig = plt.figure(figsize=(10, 10))
    l, = plt.plot([], [], 'ko', ms=4)


    plt.xlim(xLim)
    plt.ylim(yLim)

    librosa.output.write_wav(os.path.join(path, fname+'.wav'), sound, fs)

    if frames.shape[1] == 20:
        lookup = [[x[0] - 48, x[1] - 48] for x in Mouth]
        # print lookup
    else:
        lookup = faceLmarkLookup

    lines = [plt.plot([], [], 'k')[0] for _ in range(3*len(lookup))]

    with writer.saving(fig, os.path.join(path, fname+'.mp4'), 150):
        plt.gca().invert_yaxis()
        for i in tqdm(range(frames.shape[0])):
            l.set_data(frames[i,:,0], frames[i,:,1])
            cnt = 0
            for refpts in lookup:
                lines[cnt].set_data([frames[i,refpts[1], 0], frames[i,refpts[0], 0]], [frames[i, refpts[1], 1], frames[i,refpts[0], 1]])
                cnt+=1
            writer.grab_frame()

    cmd = 'ffmpeg -i '+os.path.join(path, fname)+'.mp4 -i '+os.path.join(path, fname)+'.wav -c:v copy -c:a aac -strict experimental '+os.path.join(path, fname)+'_.mp4'
    subprocess.call(cmd, shell=True) 
    print('Muxing Done')

    os.remove(os.path.join(path, fname+'.mp4'))
    os.remove(os.path.join(path, fname+'.wav'))

def main():
    return

if __name__ == "__main__":
    main()