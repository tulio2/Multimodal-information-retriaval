# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 21:26:10 2021

@author: Marco Tulio Pérez Ortega
"""
import pandas as pd
from MultiModalModel import GlobalModel
import torch
import pickle
import tensorflow.keras.applications as transfer
import numpy as np
import librosa,librosa.display
from m_functions import lastfm_get,preprocess_tags,cf_predict,get_numeric
from keras import layers
import cv2
import matplotlib.pyplot as plt
from torch import nn

class recuperador():
  def __init__(self,ruta):
    self.ruta=ruta
    
    # load data that we will need
    self.raw_tracks=pd.read_csv(ruta+'/raw_tracks.csv',index_col=0)
    
    self.model=GlobalModel()
    S = torch.load(ruta+'/Modelos/last_1111.ckpt')['state_dict']
    SS = {key[6:]: S[key] for key in S.keys()}
    self.model.load_state_dict(SS)
    self.model.eval()
    self.cat_embs=pickle.load(open(ruta+'embeddings/1111_embeddings.pkl','rb'))
    self.spec_embs=pickle.load(open(self.ruta+'embeddings/1111_spec_embeddings.pkl','rb'))
    self.cf_embs=pickle.load(open(ruta+'embeddings/1111_cf_embeddings.pkl','rb'))
    self.cover_embs=pickle.load(open(ruta+'embeddings/1111_cover_embeddings.pkl','rb'))
    self.numeric_embs=pickle.load(open(ruta+'embeddings/1111_numeric_embeddings.pkl','rb'))


  def fit(self,file=None,track=None,artist=None,k=10,mode='cultural'):
    self.mode=mode
    #k is the number of items that we want 
    #cultural consult by track and artist's name
    if self.mode=='cultural':
      self.cultural_tags=pd.read_csv(self.ruta+'mpt_reducida.csv',index_col=0).columns
      self.conf_tags=pd.read_csv(self.ruta+'conf_tc.csv',index_col=0)
      return self.get_most_similar(track=track,artist=artist,k=k)
      #editorial consult by cover of album's path
    elif self.mode=='editorial':
      self.transfer_model=transfer.ResNet101(weights='imagenet',include_top=False,input_shape=(300, 300,3))
      return self.get_most_similar(file,k=k)
    else:
    # acoustic consult by audio file's path
      self.numeric_sd=np.load(open(self.ruta+'n_var.npy','rb')) 
      self.numeric_mean=np.load(open(self.ruta+'n_mean.npy','rb'))
      self.var_names=pd.read_csv(self.ruta+'splits/indicadores.csv',index_col=0).columns
      return self.get_most_similar(file,k=k)

    
  ################ acoustic information to spectrogram to embeddings##############################################
  def get_spec_emb(self,file):
    y, _ = librosa.load(file)
    spec=librosa.feature.melspectrogram(y=y,sr=22050,n_fft=1024,hop_length=512,n_mels=128)
    self.consult_spec=spec
    if spec.shape[1] < 173:
      nspec = np.zeros((128, self.input_length))
      nspec[:, :spec.shape[1]] = spec
      spec = nspec
    num_chunk=8
    hop = (spec.shape[1] - 173) // num_chunk
    spec = np.array([spec[:, i*hop:i*hop+173] for i in range(num_chunk)])
    spec=torch.tensor(spec)
    out = self.model.spec_to_embedding(spec)
    spec_emb = out.mean(dim=0)
    return spec_emb

  ################ cultural information to   embeddings##############################################  
  def get_cf_emb(self,track,artist):
    metodo='track.getTopTags'
    r=lastfm_get({
            'method': metodo,
            'artist':artist,
            'track':track
          })
    tags={}
    if len(r.json())==1:
            n_tags=len(r.json()['toptags']['tag'])
            #Si hay al menos un tag
            if n_tags>0:
                for j in range(n_tags):
                    tags[r.json()['toptags']['tag'][j]['name']]=r.json()['toptags']['tag'][j]['count']
                df=preprocess_tags(tags,self.cultural_tags)
                Y=cf_predict(df,self.conf_tags)
                Y=torch.tensor(np.reshape(Y.astype('float32'),(1,len(Y))))
                cf_emb=self.model.cf_to_embedding(Y)
                return cf_emb
            else:
              print('Did not find tags')
    else:
      print('Did not find track')     
  ################  editorial information    to embeddings##############################################  
 
  def get_cover_emb(self,file):
    im = cv2.imread(file)
    im_resized = cv2.resize(im, (300, 300), interpolation=cv2.INTER_LINEAR)
    pre_emb=self.transfer_model(im_resized.reshape((1,300,300,3)))
    a=layers.MaxPooling2D((10, 10))(pre_emb)
    a=layers.Flatten()(a)
    a=torch.tensor(np.array(a))
    cover_emb=self.model.cover_to_embedding(a)
    return cover_emb
    ################ acoustic information to numeric to embeddings##############################################
 
  def get_numeric_emb(self,file):
    numeric=get_numeric(file,self.var_names)
    numeric=(numeric-self.numeric_mean)/self.numeric_sd
    numeric=torch.tensor(np.array(numeric).astype('float32'))
    numeric_emb=self.model.numeric_to_embedding(numeric)
    return numeric_emb
    ######## Get most similar embedding to the consult################################
  def get_most_similar(self,file=None,track=None,artist=None,k=10):
    if self.mode=='cultural':
      cult_emb=self.get_cf_emb(track,artist)
      id_fma=self.get_id_fma(cult_emb,self.cf_embs)
      return self.get_neighbors(id_fma,k)
    elif self.mode=='editorial':
      cover_emb=self.get_cover_emb(file)
      id_fma=self.get_id_fma(cover_emb,self.cover_embs)
      self.file=file
      return self.get_neighbors(id_fma,k)
    else:
      if self.mode=='acustic_spec':
        spec_emb=self.get_spec_emb(file)
        id_fma=self.get_id_fma(spec_emb,self.spec_embs)
        self.file=file
      else:
        numeric_emb=self.get_numeric_emb(file)
        id_fma=self.get_id_fma(numeric_emb,self.numeric_embs)
        self.file=file
      return self.get_neighbors(id_fma,k)
## Get id_fma of the first embedding found
  def get_id_fma(self,embedding,set_embeddings):
    maximo=-1
    arg_maximo=0
    for id_fma in set_embeddings.keys():
      sim=nn.CosineSimilarity(dim=-1)(set_embeddings[id_fma], embedding)
      if sim>maximo:
        maximo=sim
        arg_maximo=id_fma
    return arg_maximo
## Get k nearest neighbors in shared space
  def get_neighbors(self,id_fma,k):
    embedding=self.cat_embs[id_fma]
    fma_ids=np.array(list(self.cat_embs.keys()))
    embeddings=[np.array(self.cat_embs[key]) for key in self.cat_embs.keys()]
    sim=nn.CosineSimilarity(dim=-1)(embedding, torch.tensor(np.array(embeddings)))
    neighbors=list(np.argsort(np.array(-sim))[:k])
    self.k_ids=fma_ids[neighbors]
    return self.raw_tracks.loc[self.k_ids,['track_title','album_title','artist_name','tags', 'track_date_created', 'track_date_recorded','track_genres']]  
 # To show covers 
  def show_covers(self):
    if self.mode=='editorial':
      im = cv2.imread(self.file)
      im_resized = cv2.resize(im, (300, 300), interpolation=cv2.INTER_LINEAR)
      plt.imshow(cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB))
      plt.axis('off')
      plt.title('Imagen Consulta')
    k=len(self.k_ids)
    fig=plt.figure(figsize=(15,12))
    for i,fma_id in enumerate(self.k_ids):
      ax = fig.add_subplot((k//5)+1, 5, i+1)
      ax.set_title(str(fma_id))
      plt.axis('off')
      im = cv2.imread(self.ruta+'300x300/'+str(fma_id)+'.jpg')
      im_resized = cv2.resize(im, (300, 300), interpolation=cv2.INTER_LINEAR)
      plt.imshow(cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB))
    plt.show()
    #To show spectrograms
  def show_spec(self):
     if self.mode=='acustic_spec':
        print('Consulta')
        spec=librosa.power_to_db(self.consult_spec,ref=np.max)
        librosa.display.specshow(spec, x_axis='time',
                          y_axis='mel', sr=22050,
                          fmax=8000)
     k=len(self.k_ids)
     print('Más cercanos')
     fig=plt.figure(figsize=(15,12))
     for i,fma_id in enumerate(self.k_ids):
       ax = fig.add_subplot((k//5)+1, 5, i+1)
       ax.set_title(str(fma_id))
       plt.axis('off')
       spec=np.load(open(self.ruta+'Espectrogramas/'+str(fma_id)+'.npy','rb'))
       spec=librosa.power_to_db(spec, ref=np.max)
       librosa.display.specshow(spec, x_axis='time',
                          y_axis='mel', sr=22050,
                          fmax=8000, ax=ax)

   
 