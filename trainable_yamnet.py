
import tensorflow as tf
from tensorflow.keras import Model, layers
import features as features_lib
from params import Params as Params_class
from yamnet import yamnet


def yamnet_model(feature_params):
    """Creates the yamnet model, without seperating into frames. 
    Takes as input one .wav file, performs feature extraction and
    produces a spectrogram which is the input to yamnet  """
    
    waveform = layers.Input((None,))
    _,patches = features_lib.waveform_to_log_mel_spectrogram_patches(
    tf.squeeze(waveform, axis=0), feature_params)
    predictions,_ = yamnet(patches,feature_params) #yamnet returns predictions and embedings
    
    single_spec_model = Model(name='yamnet_frames', 
                       inputs=waveform, outputs=[predictions])
    return single_spec_model

def get_batch_model(feature_params):
  waveform_batch = layers.Input((None,))
  # calculate log mel spectrograms for batch of waveforms (frontend of YAMNet)
  spec_with_params = lambda w: features_lib.waveform_to_log_mel_spectrogram_patches(w, feature_params)[1] 
  spectrogram_batch = tf.map_fn(spec_with_params,waveform_batch)
  # pass spectrogram batch to core YAMNet
  out_batch,_ = yamnet(spectrogram_batch,feature_params) 
  prepr_model = Model(name='yamnet_trainable', 
                       inputs=waveform_batch, outputs=[out_batch])

  return prepr_model

def get_idx_first_conv(model):
  """Returns the index of the first convolutional
  layer found """
  first_conv_idx = None
  for layer_idx,layer in enumerate(model.layers):
    if 'conv' in layer.name:
      first_conv_idx = layer_idx
      break
  return first_conv_idx


def get_trainable_yamnet(load_weights = True,input_duration = None,num_classes = 2):
  params = Params_class() #initialize parameters with the ones hardcoded in params.py
  yamnet_pretrained = yamnet_model(params) #model defined
  if load_weights:
    yamnet_pretrained.load_weights('./yamnet.h5')
  params.num_classes = num_classes
  if input_duration != None:
      params.patch_window_seconds = input_duration
      params.patch_hop_seconds = input_duration
  params.classifier_activation = 'softmax' 
  yamnet_new =  get_batch_model(params)
  
  # set CNN and BatchNorm layers with the values determined by yamnet.h5
  pretrained_first_conv_idx = get_idx_first_conv(yamnet_pretrained)
  new_first_conv_idx = get_idx_first_conv(yamnet_new)
  
  if load_weights:
      idx_cnt = 0
      for layer_idx,layer in enumerate(yamnet_pretrained.layers[pretrained_first_conv_idx:-2]):
        #print(layer.name)
        yamnet_new.layers[new_first_conv_idx + idx_cnt].set_weights(
                layer.get_weights())
        idx_cnt += 1
  return yamnet_new

