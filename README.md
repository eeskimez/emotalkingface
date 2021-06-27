# Speech Driven Talking Face Generation from a Single Image and an Emotion Condition

[[Paper link](https://arxiv.org/pdf/2008.03592.pdf)]

Each row shows 6 videos created using the same image and speech but with different emotion input. Columns represent  `anger`, `disgust`, `fear`, `happiness`, `neutral`, and `sadness`, respectively.

![screen-gif](./assets/example.gif.gif)

## Installation

```
pip install -r requirements.txt
```

#### It depends on the following packages:
* ffmpeg
* OpenCV

The code is tested on Ubuntu 18.04 and OS X 10.15.2. 

If you have troubles with D-Lib (especially for Windows), please use conda install instead of pip:

```
conda install -c conda-forge dlib
```

## Download Data
You can download the data from this [repo](https://github.com/CheyneyComputerScience/CREMA-D).


## Convert Videos to 25 FPS
Run the following code:
```
python .\data_prep\convertFPS.py -i \raw_video_folder -o \output_folder
```

## Prepare Data
```
python .\data_prep\prepare_data.py -i \25_fps_video_folder\ -o \output_folder --mode 1 --nw 1
```

## Train
First pre-train the emotion discriminator:

```
python train.py -i /train_hdf5_folder/ -v /val_hdf5_folder/ -o ../models/mde/ --pre_train 1 --disc_emo 1 --lr_emo 1e-4
```

Then pre-train the generator:

```
python train.py -i /train_hdf5_folder/ -v /val_hdf5_folder/ -o ../models/pre_gen/ --lr_g 1e-4
```

Finally, train all together:

```
python train.py -i /train_hdf5_folder/ -v /val_hdf5_folder/ -o ../models/tface_emo/ -m ../models/pre_gen/ -mde ../models/mde/ --disc_frame 0.01 --disc_emo 0.001
```

By default, the Tensorboard log file is written to the output path. You can check the intermediate video results and loss values using Tensorboard.

## Inference

Download our pretrained model (optional): Please put pre-trained [model](https://drive.google.com/file/d/1evtS1N828JsAAzIS05NoJ2k-lKQFZtsX/view?usp=sharing) in the `model` folder.

Inference from an image and speech file:

```
python generate.py -im ./data/image_samples/img01.png -is ./data/speech_samples/speech01.wav -m ./model/ -o ./results/
```

Inference from processed dataset (h5py files):

```
python generate_all_emotions.py -ih /path/to/h5py/folder/ -m ./model/ -o ./results/
```

Inference from processed dataset (h5py files) - Mismatched emotions:

```
python generate_mismatched_emotions.py -ih /path/to/h5py/folder/ -m ./model/ -o ./results/
```

## Acknowledgment
We thank the authors of the following [repo](https://github.com/kamo-naoyuki/pytorch_convolutional_rnn).

## Citation
```
@ARTICLE{seeskimezemotface,
    title={Speech Driven Talking Face Generation from a Single Image and an Emotion Condition},
    author={Eskimez, Sefik Emre and Zhang, You and Duan, Zhiyao},
    journal={arXiv preprint arXiv:2008.03592},
    year={2020}
}
```