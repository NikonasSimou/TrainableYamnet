# Trainable YAMNet
YAMNet is a pretrained deep net that predicts 521 audio event classes based on
the [AudioSet-YouTube corpus](http://g.co/audioset), and employing the
[Mobilenet_v1](https://arxiv.org/pdf/1704.04861.pdf) depthwise-separable
convolution architecture.

This directory was based on the [official YAMnet repository](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) which is maintained by  [Manoj Plakal](https://github.com/plakal) and [Dan Ellis](https://github.com/dpwe). The goal is to provide an easy-to-use function that, given a number of classes `num_classes` and an input length `input_length` in seconds, returns a trainable YAMNet model using the pretrained weights of `yamnet.h5`.

The original `yamnet.py` was slightly modified in order to accept a batch of wav files of `input_length` each instead of a singular wav file. Note that the model should be fed with segments of audio chunks rather than entire recordings. For each such segment a class label must be given (in contrast to the original implementation where one recording is segmented into chunks internally yielding scores for each chunk seperately).

## Training on your data
In order to feed data to the model one should first load them accordingly :
```python
wav_data, sr = sf.read(file_name, dtype=np.int16)
assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
waveform = waveform.astype('float32')
```
as indicated by the `inference.py` file in the original repository.
Audio should be resampled to 16kHz mono


### Tested using

* Tensorflow == 2.2.0

### Acknowledgements

* This work was implemented as part of a paper submited in ICASSP 2021 entitled **IMPROVEMENT OF DNN-BASED COUGH DETECTION USING TRANSFER LEARNING**.
Authors of that paper are : [Nikonas Simou](https://github.com/NikonasSimou), Konstantinos Psaroulakis and Nikolaos Stefanakis.