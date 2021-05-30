# Speech Driven Talking Face Generation from a Single Image and an Emotion Condition

[[Paper link](https://arxiv.org/pdf/2008.03592.pdf)]

## Installation

```
pip install -r requirements
```

#### It also depends on the following packages:
* ffmpeg
* OpenCV

The code is tested on Ubuntu 18.04 and OS X 10.15.2.

## Train

Start visdom server:

```
visdom
```

First pre-train the emotion discriminator:

```
python train.py -i /train_hdf5_folder/ -v /val_hdf5_folder/ -o ../models/mde/ --pre_train 1 --disc_emo 1
```

Then pre-train the generator:

```
python train.py -i /train_hdf5_folder/ -v /val_hdf5_folder/ -o ../models/pre_gen/
```

Finally, train all together:

```
python train.py -i /train_hdf5_folder/ -v /val_hdf5_folder/ -o ../models/tface_emo/ -m ../models/pre_gen/ -mde ../models/mde/ --disc_frame 0.01 --disc_emo 0.001 
```

## Inference
Download our pretrained model (optional)

Please download our model to the `model` folder.
[download here](https://drive.google.com/file/d/1evtS1N828JsAAzIS05NoJ2k-lKQFZtsX/view?usp=sharing)

Inference from an image and speech file:

```
python generate.py -im ../data/image_samples/img01.png -is ../data/speech_samples/speech01.wav -m ../model/ -o ../results/
```

Inference from processed dataset (h5py files):

```
python generate_all_emotions.py -ih /path/to/h5py/folder/ -m ../model/ -o ../results/
```

Inference from processed dataset (h5py files) - Mismatched emotions:

```
python generate_mismatched_emotions.py -ih /path/to/h5py/folder/ -m ../model/ -o ../results/
```
