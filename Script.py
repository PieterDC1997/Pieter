# -*- coding: utf-8 -*-
"""
Created on Sun May  3 01:29:06 2020

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
os.chdir('C:\\Users\\Pieter\\Documents\\Analysis_Python\\')
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
        
        
        #Drop 2 otiose channels. They were saved by accidence. Need to preload data for it
        raw.load_data()
        raw.drop_channels(['EXG7','EXG8'])
        #re-reference
        raw.set_eeg_reference('average', projection=True)
        
        #Band-pass. Based on literature. 
        raw.filter(l_freq=0.01, h_freq=40, picks='all', method='fir', filter_length='auto', phase='zero', 
                   fir_window='hamming',fir_design='firwin')   
        
        #safe data files   
        fname_preprocess_non_epoch=op.join(process_path, "sub_%03d_raw.fif"% (subject_id,))
        raw.save(fname_preprocess_non_epoch, overwrite=True)
        
        
        
        
parallel, run_func, _ = parallel_func(run_events, n_jobs=1)
parallel(run_func(subject_id) for subject_id in range(1, 4)) #Change second number to N=participants+1
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
              
        #####ICA#####
        ica = ICA(random_state=97,
              n_components=15)
        picks = mne.pick_types(raw.info, eeg=True, eog=True, 
                           stim=False, exclude='bads')
        ica.fit(raw, picks=picks)
        raw.load_data()
        
        #make epochs around stimuli events
        fname_events = op.join(process_path, 'events_%03d-eve.fif' % (subject_id,))
        delay = int(round(0.0345 * raw.info['sfreq']))
        events = mne.read_events(fname_events)
        events[:, 0] = events[:, 0] + delay
        events_list.append(events)
        epochs = mne.Epochs(raw, events, events_id, tmin=-0.2, tmax=0.5, proj=True,
                        picks=picks, baseline=(None, 0), preload=False, reject=None)
        
        #get EOG epochs
        eog_epochs = create_eog_epochs(raw, tmin=-.5, tmax=.5, preload=False)
        n_max_eog = 3  # use max 2 components
        eog_epochs.load_data()
        eog_epochs.apply_baseline((None, None))
        eog_inds, scores_eog = ica.find_bads_eog(eog_epochs)
        print('    Found %d EOG indices' % (len(eog_inds),))
        ica.exclude.extend(eog_inds[:n_max_eog])
        eog_epochs.average()
        del eog_epochs
        
        #apply ICA on epochs
        epochs.load_data()
        ica.apply(epochs)
        reject = get_rejection_threshold(epochs, random_state=97)
        epochs.drop_bad(reject=reject)
        print('  Dropped %0.1f%% of epochs' % (epochs.drop_log_stats(),))
        print(epochs)
        epochs.plot(picks=('Oz'), title='epochs, electrode Oz')
        #save epochs        
        epochs.save(op.join(process_path, "sub_%03d_raw-epo.fif"% (subject_id,)))


parallel, run_func, _ = parallel_func(run_events, n_jobs=1)
parallel(run_func(subject_id) for subject_id in range(1, 4)) #Change second number to N=participants+1

#%%EVOKED DATA


def run_events(subject_id):
    subject = "sub%03d" % subject_id
    print("processing subject: %s" % subject)
    in_path = op.join(data_path, "EEG_Process")
    evo_path = op.join(data_path, "EEG_Evoked")
    for run in range(1, 2):
        fname = op.join(in_path, 'sub_%03d_raw-epo.fif' % (subject_id,))
        epochs = mne.read_epochs(fname, preload=True)
        
        ###make the evoked data###
        evoked_LSF_C_REF = epochs['LSF/C/REF'].average()
        evoked_LSF_C_TEST = epochs['LSF/C/TEST'].average()
        evoked_LSF_BT_REF = epochs['LSF/BT/REF'].average()
        evoked_LSF_BT_TEST = epochs['LSF/BT/TEST'].average()
        evoked_LSF_PST_REF = epochs['LSF/PST/REF'].average()
        evoked_LSF_PST_TEST = epochs['LSF/PST/TEST'].average()
        evoked_HSF_C_REF = epochs['HSF/C/REF'].average()
        evoked_HSF_C_TEST = epochs['HSF/C/TEST'].average()
        evoked_HSF_BT_REF = epochs['HSF/BT/REF'].average()
        evoked_HSF_BT_TEST = epochs['HSF/BT/TEST'].average()
        evoked_HSF_PST_REF = epochs['HSF/PST/REF'].average()
        evoked_HSF_PST_TEST = epochs['HSF/PST/TEST'].average()
        #general: HSF vs LSF
        evoked_LSF=epochs['LSF'].average()
        evoked_HSF=epochs['HSF'].average()
        
        #plot
        evoked_LSF.plot(picks=['Oz'], window_title='evoked, condition LSF, electrode Oz')
        evoked_HSF.plot(picks=['Oz'], window_title='evoked, condition HSF, electrode Oz')
        
        fname_evo=op.join(evo_path, "sub_%03d_raw-ave.fif"% (subject_id,))
        mne.evoked.write_evokeds(fname_evo, [evoked_LSF_C_REF,evoked_LSF_C_TEST,evoked_LSF_BT_REF,
                                             evoked_LSF_BT_TEST,evoked_LSF_PST_REF,evoked_LSF_PST_TEST,
                                             evoked_HSF_C_REF,evoked_HSF_C_TEST,evoked_HSF_BT_REF,
                                             evoked_HSF_BT_TEST,evoked_HSF_PST_REF,evoked_HSF_PST_TEST,
                                             evoked_LSF, evoked_HSF])
        


parallel, run_func, _ = parallel_func(run_events, n_jobs=1)
parallel(run_func(subject_id) for subject_id in range(1, 4)) #Change second number to N=participants+1

#%%GRAND AVERAGE
all_evokeds = [list() for _ in range(14)]  # Container for all the categories

for run in range(1, 4):
    subject = "sub%03d" % run
    print("processing subject: %s" % subject)
    in_path = op.join(data_path, "EEG_Evoked")
    evo_path = op.join(data_path, "EEG_Evoked")
    evokeds = mne.read_evokeds(op.join(in_path, 'sub_%03d_raw-ave.fif' % (run,)))
    print(evokeds)
    assert len(evokeds) == len(all_evokeds)
    for idx, evoked in enumerate(evokeds):
        all_evokeds[idx].append(evoked)  # Insert to the container

for idx, evokeds in enumerate(all_evokeds):
    all_evokeds[idx] = mne.grand_average(evokeds)  # Combine subjects
    

mne.evoked.write_evokeds(op.join(evo_path, "GA_ave.fif"), all_evokeds)
mne.evoked.write_evokeds(op.join(evo_path, "GA_ave.txt"), all_evokeds)

def main():
    """Plot evokeds."""
    for evoked in all_evokeds:
        evoked.plot(picks=('Oz'))

    # ts_args = dict(gfp=True, time_unit='s')
    # topomap_args = dict(time_unit='s')

    # for idx, evokeds in enumerate(all_evokeds):
    #     all_evokeds[idx].plot_joint(title=config.conditions[idx],
    #                                 ts_args=ts_args, topomap_args=topomap_args)  # noqa: E501


if __name__ == '__main__':
    main()
