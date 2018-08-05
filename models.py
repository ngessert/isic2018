import torch
import numbers
import numpy as np
import functools
import h5py
import math
from torchvision import models
import pretrainedmodels


model_map = {'Dense121' : models.densenet121(pretrained=True),
             'Dense121Nopre' : models.densenet121(pretrained=False),
             'Dense169' : models.densenet169(pretrained=True),
             'Dense161' : models.densenet161(pretrained=True),
             'Dense201' : models.densenet201(pretrained=True),
             'Resnet50' : pretrainedmodels.__dict__['resnet50'](num_classes=1000, pretrained='imagenet'),
             'Resnet101' : models.resnet101(pretrained=True),   
             'InceptionV3': pretrainedmodels.__dict__['inceptionv3'](num_classes=1000, pretrained='imagenet'),# models.inception_v3(pretrained=True),
             'se_resnext50': pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained='imagenet'),
             'se_resnext101': pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained='imagenet'),
             'se_resnet50': pretrainedmodels.__dict__['se_resnet50'](num_classes=1000, pretrained='imagenet'),
             'se_resnet101': pretrainedmodels.__dict__['se_resnet101'](num_classes=1000, pretrained='imagenet'),
             'se_resnet152': pretrainedmodels.__dict__['se_resnet152'](num_classes=1000, pretrained='imagenet'),
             'resnext101': pretrainedmodels.__dict__['resnext101_32x4d'](num_classes=1000, pretrained='imagenet'),
             'resnext101_64': pretrainedmodels.__dict__['resnext101_64x4d'](num_classes=1000, pretrained='imagenet'),
             'senet154': pretrainedmodels.__dict__['senet154'](num_classes=1000, pretrained='imagenet'),
             'polynet': pretrainedmodels.__dict__['polynet'](num_classes=1000, pretrained='imagenet'),
             'dpn92': pretrainedmodels.__dict__['dpn92'](num_classes=1000, pretrained='imagenet+5k'),
             'dpn68b': pretrainedmodels.__dict__['dpn68b'](num_classes=1000, pretrained='imagenet+5k'),
             'nasnetamobile': pretrainedmodels.__dict__['nasnetamobile'](num_classes=1000, pretrained='imagenet')
               }

def getModel(model_name):
  """Returns a function for a model
  Args:
    mdlParams: dictionary, contains configuration
    is_training: bool, indicates whether training is active
  Returns:
    model: A function that builds the desired model
  Raises:
    ValueError: If model name is not recognized.
  """
  if model_name not in model_map:
    raise ValueError('Name of model unknown %s' % model_name)
  func = model_map[model_name]
  @functools.wraps(func)
  def model():
      return func
  return model