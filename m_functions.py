# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 17:28:00 2021

@author: under
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 15:53:56 2021
@author: Marco Tulio Pérez Ortega
"""

import pandas as pd 
import numpy as np
import requests
import re
import essentia
from essentia.standard import *

def lastfm_get(payload,api_key,api_secret_key):
    #You have to use your own lastfm's api_key
    
    api_secret_key=api_secret_key
    user_agent='Dataquest'
    # define headers and URL
    headers = {'user-agent': user_agent}
    url = 'http://ws.audioscrobbler.com/2.0/'
    # Add API key and format to the payload
    payload['api_key'] = api_key
    payload['format'] = 'json'

    response = requests.get(url, headers=headers, params=payload)
    return response
# Preprocesing of cultural tags
def preprocess_tags(tags,cultural_tags):
    aux=np.zeros(len(cultural_tags))
    df=pd.DataFrame(aux,index= cultural_tags)
    df=df.T
#Remove some characters
    for tag_0 in tags.keys():
        tag=str(tag_0).lower()
        tag= re.sub('[.,&]','',tag)
        tag= re.sub('musica','',tag)
        tag= re.sub('music','',tag)
        tag= re.sub('-',' ',tag)
        tag= re.sub('   ',' ',tag)
        tag= re.sub('  ',' ',tag)
        tag= re.sub('songs','',tag)
        tag= re.sub('song','',tag)
        tag= re.sub('tag','',tag)
        tag= re.sub('\'s','',tag)
        tag= re.sub('0s','0d',tag)
        tag= re.sub('000','0',tag)
        tag= re.sub('111','1',tag) 
        tag= re.sub('222','2',tag)
        tag= re.sub('333','3',tag)
        tag= re.sub('444','4',tag)
        tag= re.sub('555','5',tag)
        tag= re.sub('666','6',tag)
        tag= re.sub('777','7',tag)
        tag= re.sub('888','8',tag)
        tag= re.sub('999','9',tag)
        tag= re.sub('sss','s',tag)
        #We  look for matches
        if  tag in cultural_tags:
            df.loc[0,tag]=tags[tag_0]
        else: 
            for tag_1 in cultural_tags:
                if tag in tag_1:
                    df.loc[0,tag_1]=tags[tag_0]
                else:
                    if tag_1 in tag:
                          df.loc[0,tag_1]=tags[tag_0]
    return df
# Vector tag-song to cf configuration
def cf_predict(conf_tags,df):
      A=np.array(df)
      P=(A>0)
      B=A+1
      m=A.shape[1]
      X=np.array(conf_tags)
      XX=X.T@X
      aux1=np.zeros((200,m))
      p=np.reshape(P,(m,1))
      C=np.diag(B[0][list(P)])
      aux1[:,list(P[0])]=X.T[:,list(P[0])]@C
      term1=XX+aux1@X+np.eye(200)
      term1=np.linalg.inv(term1)
      Y=np.reshape(term1@aux1@p,200)
      return Y

# Get numeric acustic features
def get_numeric(file,var_names):
  v_numeric=pd.DataFrame(np.zeros((1,len(var_names))),columns=var_names)
  get_dance=Danceability()
  get_energy=Energy()
  FD=FadeDetection()
  DC=DynamicComplexity()
  audio = essentia.standard.MonoLoader(filename=file)()
  _,_,bpm,_,_,fp_bpm,fp_spread,fp_w,sp_bpm,sp_spread,sp_w,_=RhythmDescriptors()(audio)
  v_numeric.loc[0,'bpm']=bpm
  v_numeric.loc[0,'fp_bpm']=fp_bpm
  v_numeric.loc[0,'fp_spread']=fp_spread
  v_numeric.loc[0,'fp_w']=fp_w
  v_numeric.loc[0,'sp_bpm']=sp_bpm
  v_numeric.loc[0,'sp_spread']=sp_spread
  v_numeric.loc[0,'sp_w']=sp_w
  
  v_numeric.loc[0,'danceability'],_=get_dance(audio)
  v_numeric.loc[0,'energy']=get_energy(audio)
  chor_changes_rate,_,chords_key,chords_number_rate,_,chords_scale,chords_strength,_,_,key_key,key_scale,key_strength=TonalExtractor()(audio)
  v_numeric.loc[0,'chor_changes_rate']=chor_changes_rate
  v_numeric.loc[0,'chor_number_rate']=chords_number_rate
  v_numeric.loc[0,'key_strength']=key_strength
  v_numeric.loc[0,'chords_strength_n']=len(chords_strength)
  if len(chords_strength)>0:
    chords_strength_mean=pd.DataFrame(chords_strength).mean()[0]
    chords_strength_sd=pd.DataFrame(chords_strength).std()[0]
    chords_strength_skew=pd.DataFrame(chords_strength).skew()[0]
    chords_strength_curt=pd.DataFrame(chords_strength).kurtosis()[0]
  else:
    chords_strength_mean=0
    chords_strength_sd=0
    chords_strength_skew=0
    chords_strength_curt=0
  v_numeric.loc[0,'chords_strength_mean']=chords_strength_mean
  v_numeric.loc[0,'chords_strength_sd']=chords_strength_sd
  v_numeric.loc[0,'chords_strength_skew']=chords_strength_skew
  v_numeric.loc[0,'chords_strength_kurtosis']=chords_strength_curt
  fade_in,fade_out=FD(audio)
  if fade_in.shape[0]>0:
    fade_time1=pd.DataFrame(fade_in[:,1]-fade_in[:,0])
    fade_mean1=fade_time1.mean()[0]
    fade_sd1=fade_time1.std()[0]
    fade_skew1=fade_time1.skew()[0]
    fade_kurt1=fade_time1.kurtosis()[0]
  else:
    fade_mean1=0
    fade_sd1=0
    fade_skew1=0
    fade_kurt1=0
  if fade_out.shape[0]>0:
    fade_time2=pd.DataFrame(fade_out[:,1]-fade_out[:,0])
    fade_mean2=fade_time2.mean()[0]
    fade_sd2=fade_time2.std()[0]
    fade_skew2=fade_time2.skew()[0]
    fade_kurt2=fade_time2.kurtosis()[0]
  else:
    fade_mean2=0
    fade_sd2=0
    fade_skew2=0
    fade_kurt2=0
  v_numeric.loc[0,'fade_mean1']=fade_mean1
  v_numeric.loc[0,'fade_mean2']=fade_mean2
  v_numeric.loc[0,'fade_sd1']=fade_sd1
  v_numeric.loc[0,'fade_sd2']=fade_sd2
  v_numeric.loc[0,'fade_skew1']=fade_skew1
  v_numeric.loc[0,'fade_skew2']=fade_skew2
  v_numeric.loc[0,'fade_kurt1']=fade_kurt1
  v_numeric.loc[0,'fade_kurt2']=fade_kurt2
  
  v_numeric.loc[0,'dynamicComplexity'],v_numeric.loc[0,'loudness']=DC(audio)
  v_numeric.loc[0,'intensity']=Intensity()(audio)
  if chords_scale=='minor':
    v_numeric.loc[0,'chord_minor']=1
  if key_scale=='minor':
    v_numeric.loc[0,'key_minor']=1
  if chords_key!='A':
    x='chord_'+chords_key
    v_numeric.loc[0,x]=1
  if key_key!='A':
    x='key_'+key_key
    v_numeric.loc[0,x]=1
  return v_numeric                      
