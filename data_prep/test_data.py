import h5py
import utils
import argparse
import os
#----------------PARSER------------------#
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-i", "--input-file", type=str, help='Path to file')
parser.add_argument("-o", "--out-path", type=str, help='Output path')
args = parser.parse_args()

dset = h5py.File(args.input_file, 'r')

video = dset['video'][...]
speech = dset['speech'][...]

print(video.shape, speech.shape)

utils.write_video_cv(video, speech, 8000, args.out_path, 'test.mp4', 25.0)