# -*- coding: utf-8 -*-
"""
Created on Sat May  2 17:55:34 2020

@author: Pieter
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 22:55:27 2020

@author: Pieter
"""
#%% IMPORT MODULES 
import numpy as np
import mne
from mne.parallel import parallel_func
import os
import os.path as op
from mne.preprocessing import create_ecg_epochs, create_eog_epochs, read_ica, ICA
from autoreject import get_rejection_threshold
import matplotlib.pyplot as plt
#%% CWD
os.getcwd()
os.chdir('C:\\Users\\Pieter\\Documents\\CAED\\')
cwd= os.getcwd()
data_path = cwd
print(cwd)
#%% PREPROCESS STEPS BEFORE EPOCHING

def run_events(subject_id):
    subject = "sub_%03d" % subject_id
    print("processing subject: %s" % subject)
    in_path = op.join(data_path, "Subjects") #make map yourself in cwd called 'Subjects'
    process_path = op.join(data_path, "EEG_Process") #make map yourself in cwd called 'EEG_Process'
    for run in range(1, 2):
        fname = op.join(in_path, 'sub_%03d.bdf' % (subject_id,))
        raw=mne.io.read_raw_bdf(fname)
        print("  S %s - R %s" % (subject, run))
        raw.plot()
        #info dataset
        info=raw.info #You can always print whatever info you'd like to acquire (events, bads, channel names...)
        #bad electrodes
        bads=raw.info['bads']
        #raw.load_data()   Uncomment this if you have bad channels
        #raw.interpolate_bads() #Uncomment this if you have bad channels
        
        events = mne.find_events(raw)
        fname_events = op.join(process_path, 'events_%03d-eve.fif' % (subject_id,))
        mne.write_events(fname_events, events)
        
        #####CHANNELS#####
        #BDF Files have other channel names. Changing them to the wide-known names
        mne.rename_channels(info=info, mapping={'A1':'Fp1', 'A2':'AF7','A3':'AF3','A4':'F1',
                                        'A5':'F3', 'A6':'F5', 'A7':'F7', 'A8':'FT7',
                                        'A9':'FC5','A10':'FC3', 'A11':'FC1', 'A12':'C1', 
                                        'A13':'C3', 'A14':'C5', 'A15':'T7', 'A16':'TP7',
                                        'A17':'CP5','A18':'CP3', 'A19':'CP1', 'A20':'P1', 
                                        'A21':'P3', 'A22':'P5', 'A23':'P7', 'A24':'P9', 
                                        'A25':'PO7', 'A26':'PO3', 'A27':'O1', 'A28':'Iz',
                                        'A29':'Oz', 'A30':'POz', 'A31':'Pz','A32':'CPz',
                                        'B1':'Fpz', 'B2':'Fp2','B3':'AF8','B4':'AF4', 'B5':'AFz',
                                        'B6':'Fz', 'B7':'F2', 'B8':'F4', 'B9':'F6', 'B10':'F8',
                                        'B11':'FT8','B12':'FC6', 'B13':'FC4', 'B14':'FC2',
                                        'B15':'FCz', 'B16':'Cz', 'B17':'C2', 'B18':'C4', 'B19':'C6',
                                        'B20':'T8', 'B21':'TP8', 'B22':'CP6','B23':'CP4',
                                        'B24':'CP2', 'B25':'P2', 'B26':'P4', 'B27':'P6', 'B28':'P8',
                                        'B29':'P10','B30':'PO8', 'B31':'PO4', 'B32':'O2','EXG1':'EXVR',
                                        'EXG2':'EXHR','EXG3':'EXHL', 'EXG5':'M1', 'EXG6':'M2'})#M1&2=mastoids
        #Changing the channel types
        raw.set_channel_types({'EXVR': 'eog','EXHR': 'eog','EXHL': 'eog','EXG4': 'eog',
                               'M1':'misc','M2':'misc','GSR1':'bio'})
    
        #topographical maps later on; set montage!
        montage = mne.channels.read_montage(kind='biosemi64')
        print(montage)
        raw.set_montage(montage, set_dig=True)
        raw.plot_sensors(ch_type='eeg')
        
        #Drop 2 otiose channels. They were saved by accidence. Need to preload data for it
        raw.load_data()
        raw.drop_channels(['EXG7','EXG8'])
        
        #PSD: quite interesting to identify noisy channels
        raw.plot_psd(dB=True,estimate='power')
        
        #re-reference
        raw.set_eeg_reference('average', projection=True)
        
        #Band-pass. Based on literature. 
        raw.filter(l_freq=0.01, h_freq=40, picks='all', method='fir', filter_length='auto', phase='zero', 
                   fir_window='hamming',fir_design='firwin')   
        
        #safe data files   
        fname_preprocess_non_epoch=op.join(process_path, "sub_%03d_raw.fif"% (subject_id,))
        raw.save(fname_preprocess_non_epoch, overwrite=True)
        raw.plot()
        

parallel, run_func, _ = parallel_func(run_events, n_jobs=1)
parallel(run_func(subject_id) for subject_id in range(1, 2)) #Change second number to N=participants+1
#%% PREPROCESS ICA + EPOCH + REJECTING BAD TRIALS

#Hierarchical events
events_id = {'LSF/C/REF': 20,
    'LSF/C/TEST': 30,
    'LSF/BT/REF': 40,
    'LSF/BT/TEST': 50,
    'LSF/PST/REF': 60,
    'LSF/PST/TEST': 70,
    'HSF/C/REF': 25,
    'HSF/C/TEST': 35,
    'HSF/BT/REF': 45,
    'HSF/BT/TEST': 55,
    'HSF/PST/REF': 65,
    'HSF/PST/TEST': 75}

def run_events(subject_id):
    subject = "sub_%03d" % subject_id
    print("processing subject: %s" % subject)
    in_path = op.join(data_path, "EEG_Process") #make map yourself in cwd called 'Subjects'
    process_path = op.join(data_path, "EEG_Process") #make map yourself in cwd called 'EEG_Process'
    raw_list = list()
    events_list = list()
    
    for run in range(1, 2):
        fname = op.join(in_path, 'sub_%03d_raw.fif' % (subject_id,))
        raw=mne.io.read_raw_fif(fname, preload=True)
        print("  S %s - R %s" % (subject, run))
        
        #import events and reorganize
        delay = int(round(0.0345 * raw.info['sfreq']))
        events = mne.read_events(op.join(in_path, 'events_%03d-eve.fif' % (subject_id,)))
        events[:, 0] = events[:, 0] + delay
        events_list.append(events)
        raw_list.append(raw)
        raw, events = mne.concatenate_raws(raw_list, events_list=events_list)
        
        ###some visualizations on the blinks in the raw data file###
        eog_events = mne.preprocessing.find_eog_events(raw)
        onsets = eog_events[:, 0] / raw.info['sfreq'] - 0.25
        durations = [0.5] * len(eog_events)
        descriptions = ['bad blink'] * len(eog_events)
        blink_annot = mne.Annotations(onsets, durations, descriptions,
                              orig_time=raw.info['meas_date'])
        raw.set_annotations(blink_annot)
        eeg_picks = mne.pick_types(raw.info, eeg=True)
        raw.plot(events=eog_events, order=eeg_picks)
        ###CONCLUSION: NOT THE BEST ALGORITHM
        
        
        #####ICA#####
        ica = ICA(random_state=97,
              n_components=15)
        picks = mne.pick_types(raw.info, eeg=True, eog=True, 
                           stim=False, exclude='bads')
        ica.fit(raw, picks=picks)
        raw.load_data()
        ica.plot_sources(raw)
        ica.plot_components()
        ica.plot_overlay(raw, exclude=[6], picks='eeg')
        #visualize the difference
        raw2=raw.copy()
        ica.exclude=[6]
        ica.apply(raw2)
        raw2.plot()
        ica.plot_properties(raw, picks=[6])
        
        
        


parallel, run_func, _ = parallel_func(run_events, n_jobs=1)
parallel(run_func(subject_id) for subject_id in range(1, 2)) #Change second number to N=participants+1



