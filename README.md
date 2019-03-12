# Norbert
![norbert-wiener](https://user-images.githubusercontent.com/72940/45908695-15ce8900-bdfe-11e8-8420-78ad9bb32f84.jpg) 
[![Build Status](https://travis-ci.org/sigsep/norbert.svg?branch=master)](https://travis-ci.org/sigsep/norbert)
[![Coverage Status](https://coveralls.io/repos/github/sigsep/norbert/badge.svg?branch=master)](https://coveralls.io/github/sigsep/norbert?branch=master)

is an audio I/O toolbox for effiently transform, store and filter audio spectrograms, especially suited for machine learning tasks that rely on non-negative data such as in source separation. In turn, _Norbert_ does use an optimized pipeline to transform and scale audio signals and then apply lossy compression to save them efficiently as source images. This makes it an ideal fit to process music data with machine learning libraries such as [PyTorch](pytorch.org) and [Tensorflow](tensorflow.org) that have fast, builtin support to load and process images. Last but not least, Norbert provides convenient functions to easily apply multichannel Wiener filtering to the sepearated sources. 

## Features

* Multichannel Time-Frequency Transform (STFT)
* Generalized Multi-channel __Wiener__ Filter
* Log Magnitude compression
* Bandwidth reduction
* Quantization
* Image export

## Applications

* Source Separtion
* Data Prepreprocessing for audio tasks
* Create beautiful Spectrograms

## Installation

`pip install norbert`

## Usage

### Transform

Norbert includes a multichannel short-time Fourier transform by wrapping the builtin scipy implementation. For convenience the transform parameters are stored in the `norbert.TF` object. That way, the inverse transform can easily be called later.

```
tf = norbert.TF(n_fft=2048, n_hop=1024)
X = tf.transform(audio)
inverse = tf.inverse_transform(X)
```

### Filtering

### Compression

### Bandwidth Reduction

## Quantization

### Image

Many researchers save their magnitude dataset as numpy pickles or hdf5 files. While this is fast to load and write it uses a significant amount of disk space to store the files (even when zipped). Also, since jpg routines are highly optimized these days, reading jpgs is significantly faster than decoding AAC or MP3 files.
Here is a bitrate comparison:

* __npy 64bit:__ ~750 kb/s
* __npy 64bit: zipped:__ ~680 kb/s
* __MP3 good quality:__ 256 kb/s
* __AAC good quality:__ 128 kb/s
* __norbert quantization as 8bit npy:__ 89 kb/s
* __norbert quantization as 8bit jpg (`q=80`):__ 15 kb/s

![mix](https://user-images.githubusercontent.com/72940/45908846-ef5d1d80-bdfe-11e8-8531-3d30b1c98db9.jpg)

We built _Norbert_ in the context of source separation models where filtering is applied using the original mixture phase, thus reducing the influence of minor imperfections of the magnitude. We used the [PEAQ objective audio quality evaluation](example/jpg_quality_experiment.py) to assess the quality difference in a setting where we compress the magnitude of and audio signal and synthesize using the decoded (but compressed) magnitude, together with the original uncompressed mixture phase. The results on 50 music tracks from the [MUSDB18](sigsep.github.io/musdb18) dataset shows, that with the right JPG quality parameter (we pick `80` as our default), difference between the compressed magnitude and the original magnitude are almost imperceptable.

![stereo](https://user-images.githubusercontent.com/72940/41040263-2f0a08ba-699c-11e8-9d73-c52e7d04aa25.png)

Using the `Norbert.Coder` module, we built the [MUSMAG](https://sigsep.github.io/datasets/musdb.html#musmag) dataset, a dataset of precomputed audio multitrack spectrograms for source separation tasks. 
