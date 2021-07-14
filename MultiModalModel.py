# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 11:27:07 2021

@author: Marco Tulio Pérez Ortega
"""

import torch
from torch import nn
import numpy as np

#convolution layer 2d and convolution to embedding
class Conv_2d(nn.Module):
  def __init__(self, input_channels, output_channels, shape=3, pooling=2):
    super(Conv_2d, self).__init__()
    self.conv = nn.Conv2d(input_channels, output_channels, shape, padding=shape//2)
    self.bn = nn.BatchNorm2d(output_channels)
    self.relu = nn.ReLU()
    self.mp = nn.MaxPool2d(pooling)

  def forward(self, x):
    out = self.mp(self.relu(self.bn(self.conv(x))))
    return out
# conv to emb
class Conv_emb(nn.Module):
  def __init__(self, input_channels, output_channels):
    super(Conv_emb, self).__init__()
    self.conv = nn.Conv2d(input_channels, output_channels, 1)
    self.bn = nn.BatchNorm2d(output_channels)
    self.relu = nn.ReLU()

  def forward(self, x):
    out = self.relu(self.bn(self.conv(x)))
    return out
###################### Model ###############################
###################### Global Modelo ###############################

class GlobalModel(nn.Module):
  def __init__(self,ind=[True,True,True,True]):
    super(GlobalModel, self).__init__()
    self.indicador=ind
    #### You indicate the modes to consider
    if self.indicador[0]:
      # CNN module for spectrograms
      self.spec_bn = nn.BatchNorm2d(1)
      self.layer1 = Conv_2d(1, 128, pooling=2)
      self.layer2 = Conv_2d(128, 128, pooling=2)
      self.layer3 = Conv_2d(128, 256, pooling=2)
      self.layer4 = Conv_2d(256, 256, pooling=2)
      self.layer5 = Conv_2d(256, 256, pooling=2)
      self.layer6 = Conv_2d(256, 256, pooling=2)
      self.layer7 = Conv_2d(256, 512, pooling=2)
      self.layer8 = Conv_emb(512, 200)
    if self.indicador[1]:
      # FC module for collaborative filtering embedding
      self.cf_fc1 = nn.Linear(200, 512)
      self.cf_bn1 = nn.BatchNorm1d(512)
      self.cf_fc2 = nn.Linear(512, 200)
    if self.indicador[2]:
      # edit module for cover embedding
      self.edit_fc1 = nn.Linear(2048, 1024)
      self.edit_bn1 = nn.BatchNorm1d(1024)
      self.edit_fc2 = nn.Linear(1024, 512)
      self.edit_bn2 = nn.BatchNorm1d(512)
      self.edit_fc3 = nn.Linear(512, 200)
    if self.indicador[3]:
      # acuoustic numeric module
      self.numeric_fc1=nn.Linear(52,100)
      self.numeric_bn1=nn.BatchNorm1d(100)
      self.numeric_fc2=nn.Linear(100,200)
    
    # FC module for concatenated embedding
    input_shape=sum(ind)*200
    self.cat_fc1 = nn.Linear(input_shape, 512)
    self.cat_bn1 = nn.BatchNorm1d(512)
    self.cat_fc2 = nn.Linear(512, 256)

    # FC module for word embedding
    self.fc1 = nn.Linear(300, 512)
    self.bn1 = nn.BatchNorm1d(512)
    self.fc2 = nn.Linear(512, 256)

    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.5)

  def spec_to_embedding(self, spec):
    out = spec.unsqueeze(1)
    out = self.spec_bn(out)

    # CNN
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.layer5(out)
    out = self.layer6(out)
    out = self.layer7(out)
    out = self.layer8(out)
    out = out.squeeze(2)
    out = nn.MaxPool1d(out.size(-1))(out)
    out = out.view(out.size(0), -1)
    return out

  def cf_to_embedding(self, cf):
    out = self.cf_fc1(cf)
    out = self.cf_bn1(out)
    out = self.relu(out)
    out = self.dropout(out)
    out = self.cf_fc2(out)
    return out
  def cover_to_embedding(self, cover):
    out = self.edit_fc1(cover)
    out = self.edit_bn1(out)
    out = self.relu(out)
    out = self.dropout(out)
    out = self.edit_fc2(out)
    out = self.edit_bn2(out)
    out = self.relu(out)
    out = self.dropout(out)
    out = self.edit_fc3(out)
    return out
  def numeric_to_embedding(self, ind_numeric):
    out = self.numeric_fc1(ind_numeric)
    out = self.numeric_bn1(out)
    out = self.relu(out)
    out = self.dropout(out)
    out = self.numeric_fc2(out)
    return out

  def cat_to_embedding(self, cat):
   
    out = self.cat_fc1(cat)
    out = self.cat_bn1(out)
    out = self.relu(out)
    out = self.dropout(out)
    out = self.cat_fc2(out)
    return out

  def song_to_embedding(self, spec, cf,cover,ind_numeric):
    # spec to embedding
    out_cat=torch.tensor(np.array([],dtype=np.float32)).cuda()
    if self.indicador[0]:
      out_spec = self.spec_to_embedding(spec)
      out_cat=torch.cat([out_cat,out_spec],-1)
    # cf to embedding
    if self.indicador[1]:
      out_cf = self.cf_to_embedding(cf)
      out_cat=torch.cat([out_cat,out_cf],-1)
    if self.indicador[2]:
      out_cover = self.cover_to_embedding(cover)
      out_cat=torch.cat([out_cat,out_cover],-1)
    if self.indicador[3]:
      out_numeric = self.numeric_to_embedding(ind_numeric)
      out_cat=torch.cat([out_cat,out_numeric],-1)
    

    # fully connected
    out = self.cat_to_embedding(out_cat)
    return out

  def word_to_embedding(self, emb):
    out = self.fc1(emb)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.dropout(out)
    out = self.fc2(out)
    return out

  def forward(self, tag, spec, cf, cover, ind_numeric):
    
    tag_emb = self.word_to_embedding(tag)
    song_emb = self.song_to_embedding(spec, cf,cover,ind_numeric)
    return tag_emb, song_emb
