# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:29:07 2020

@author: Pieter

#####INFORMATION######
#I have already explored MNE. I was able to read a single BDF file, to rename the channels,
get some info on the raw dataset, excluded channels I am not interested in.

#Mere visualization: I selected a single electrode (Oz, one of the electrodes I plan to include
for my cluster). I created an epoch (-200 - 400ms, based on relevant literature concerning the
research question), based on one of my triggers.

#My thesis deadline is 28th of april, so my promotors can give some feedback. After that, my
main and single focus will go to this project which I am actually quite enthusiastic about!
I am thrilled to learn some new techniques on EEG data analyses!

Pieter.
"""
#%% IMPORT MODULES 
import numpy as np
import mne
import os
import matplotlib.pyplot as plt
#%%CWD + FNAMES + RENAME CHANNELS
cwd= os.getcwd()
data_path = cwd
fname = data_path + '/01.bdf'
raw=mne.io.read_raw_bdf(fname)
info=raw.info
mne.rename_channels(info=info, mapping={'A1':'FP1', 'A2':'AF7','A3':'AF3','A4':'F1',
                                        'A5':'F3', 'A6':'F5', 'A7':'F7', 'A8':'FT7',
                                        'A9':'FC5','A10':'FC3', 'A11':'FC1', 'A12':'C1', 
                                        'A13':'C3', 'A14':'C5', 'A15':'T7', 'A16':'TP7',
                                        'A17':'CP5','A18':'CP3', 'A19':'CP1', 'A20':'P1', 
                                        'A21':'P3', 'A22':'P5', 'A23':'P7', 'A24':'P9', 
                                        'A25':'PO7', 'A26':'PO3', 'A27':'O1', 'A28':'Iz',
                                        'A29':'Oz', 'A30':'POz', 'A31':'Pz','A32':'CPz',
                                        'B1':'FPz', 'B2':'FP2','B3':'AF8','B4':'AF4', 'B5':'AFz',
                                        'B6':'Fz', 'B7':'F2', 'B8':'F4', 'B9':'F6', 'B10':'F8',
                                        'B11':'FT8','B12':'FC6', 'B13':'FC4', 'B14':'FC2',
                                        'B15':'FCz', 'B16':'Cz', 'B17':'C2', 'B18':'C4', 'B19':'C6',
                                        'B20':'T8', 'B21':'TP8', 'B22':'CP6','B23':'CP4',
                                        'B24':'CP2', 'B25':'P2', 'B26':'P4', 'B27':'P6', 'B28':'P8',
                                        'B29':'P10','B30':'PO8', 'B31':'PO4', 'B32':'O2',
                                        'EXG1':'EXVR', 'EXG2':'EXHR','EXG3':'EXHL'})
                                        
#%%PICKS
want_eeg=True
want_meg=False
want_stim=False
excl= ['EXVR', 'EXHR', 'EXHL', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1', 'Status']
picks = mne.pick_types(raw.info, meg=want_meg, eeg=want_eeg, stim=want_stim, exclude=excl, selection=('Oz'))

#%%EVENTS + SOME VISUALIZATIONS ON TRIGGERS
events=mne.find_events(raw)
events[:, 2] &= (2**16 - 1) 
cms_bit = 20 
cms_high = (events[:, 2] & (1 << cms_bit)) != 0   

#%% EPOCH + AV
event_id=25
tmin=-0.2
tmax=0.4

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=(None, 0), preload=True)
evoked = epochs.average()
evoked.plot(time_unit='s')

#%%

