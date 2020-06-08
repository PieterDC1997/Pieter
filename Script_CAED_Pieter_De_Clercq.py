# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:04:26 2020

@author: Pieter
"""

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
from numpy.random import randn
import mne
from mne.parallel import parallel_func
import os
import os.path as op
from mne.preprocessing import create_ecg_epochs, create_eog_epochs, read_ica, ICA
from autoreject import get_rejection_threshold
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_test
from sklearn.model_selection import KFold
from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              write_inverse_operator, read_inverse_operator)
from mne.stats import (spatio_temporal_cluster_1samp_test,
                       summarize_clusters_stc)
#%% CWD + AVERAGE BRAIN
os.getcwd()
os.chdir('C:\\Users\\Pieter\\Documents\\CAED\\')
cwd= os.getcwd()
data_path = cwd
print(cwd)

from mne.datasets import fetch_fsaverage

# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
# The files live in:
subject = 'fsaverage'
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
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
        montage = mne.channels.read_montage('standard_1005',
                                    transform=True)
        print(montage)
        raw.set_montage(montage)
        
        
        #Drop 2 otiose channels. They were saved by accidence. Need to preload data for it
        raw.load_data()
        raw.drop_channels(['EXG7','EXG8'])
        #re-reference
        raw.set_eeg_reference('average', projection=True)
        
        #PSD: quite interesting to identify noisy channels
        #raw.plot_psd(dB=True,estimate='power')
        
        #Band-pass. Based on literature. 
        raw.filter(l_freq=0.01, h_freq=40, picks='all', method='fir', filter_length='auto', phase='zero', 
                   fir_window='hamming',fir_design='firwin')   
        
        #safe data files   
        fname_preprocess_non_epoch=op.join(process_path, "sub_%03d_raw.fif"% (subject_id,))
        raw.save(fname_preprocess_non_epoch, overwrite=True)
        
        #Check if template brain matches electrodes!
        #mne.viz.plot_alignment(raw.info, src=src, eeg=['projected'], trans=trans,
        #                        show_axes=True, mri_fiducials=True, dig='fiducials')
        
            
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

def run_epoch(subject_id):
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
        #epochs.plot(picks=('Oz'), title='epochs, electrode Oz')
        #save epochs        
        epochs.save(op.join(process_path, "sub_%03d_raw-epo.fif"% (subject_id,)))


parallel, run_func, _ = parallel_func(run_epoch, n_jobs=1)
parallel(run_func(subject_id) for subject_id in range(1, 4)) #Change second number to N=participants+1

#%%EVOKED DATA

def run_evoked(subject_id):
    subject = "sub%03d" % subject_id
    print("processing subject: %s" % subject)
    in_path = op.join(data_path, "EEG_Process")
    evo_path = op.join(data_path, "EEG_Evoked")
    for run in range(1, 2):
        fname = op.join(in_path, 'sub_%03d_raw-epo.fif' % (subject_id,))
        epochs = mne.read_epochs(fname, preload=True)
        
        #compute covariance for later on (inverse solution)
        fname_cov=op.join(evo_path, "sub_%03d_LSF_HSF-cov.fif"% (subject_id,))
        cv = KFold(3, random_state=97)  # make sure cv is deterministic
        cov = mne.compute_covariance(epochs, tmax=-0.01, method='shrunk', cv=cv)
        cov.save(fname_cov)
        mne.viz.plot_cov(cov, epochs.info)
        
        #general: HSF vs LSF
        evoked_LSF=epochs['LSF'].average()
        evoked_HSF=epochs['HSF'].average()
        contrast= mne.combine_evoked([evoked_HSF, evoked_LSF],
                                     weights=[1, -1])
        #name the conditions
        # Simplify comment
        evoked_LSF.comment = 'evoked_LSF'
        evoked_HSF.comment = 'evoked_HSF'
        contrast.comment = 'contrast'
        
        #contrast.plot(picks=('Oz'), window_title='CONTRAST')
        
        #plot
        #evoked_LSF.plot(picks=['Oz'], window_title='evoked, condition LSF, electrode Oz')
        #evoked_HSF.plot(picks=['Oz'], window_title='evoked, condition HSF, electrode Oz')
        
        fname_evo=op.join(evo_path, "sub_%03d_LSF_HSF-ave.fif"% (subject_id,))
        mne.evoked.write_evokeds(fname_evo, [evoked_LSF, evoked_HSF, contrast])
        
        #compute forward solution for later on (inverse solution)
        fname_fwd=op.join(evo_path, "sub_%03d_LSF_HSF-fwd.fif"% (subject_id,))
        info = mne.io.read_info(fname_evo)
        fwd = mne.make_forward_solution(info=info, trans=trans, src=src,
                                bem=bem, eeg=True, mindist=5.0, n_jobs=1)
        print(fwd)
        leadfield = fwd['sol']['data']
        print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)
        mne.write_forward_solution(fname_fwd, fwd, overwrite=True)
        
        # for illustration purposes use fwd to compute the sensitivity map
        eeg_map = mne.sensitivity_map(fwd, ch_type='eeg', mode='fixed')
        eeg_map.plot(time_label='EEG sensitivity LSF', clim=dict(lims=[5, 50, 100]))
        

parallel, run_func, _ = parallel_func(run_evoked, n_jobs=1)
parallel(run_func(subject_id) for subject_id in range(1, 4)) #Change second number to N=participants+1

#%%GRAND AVERAGE
all_evokeds = [list() for _ in range(3)]  # Container for all the categories

for run in range(1, 4):
    subject = "sub%03d" % run
    print("processing subject: %s" % subject)
    in_path = op.join(data_path, "EEG_Evoked")
    evo_path = op.join(data_path, "EEG_Evoked")
    evokeds = mne.read_evokeds(op.join(in_path, 'sub_%03d_LSF_HSF-ave.fif' % (run,)))
    print(evokeds)
    assert len(evokeds) == len(all_evokeds)
    for idx, evoked in enumerate(evokeds):
        all_evokeds[idx].append(evoked)  # Insert to the container
        

for idx, evokeds in enumerate(all_evokeds):
    all_evokeds[idx] = mne.grand_average(evokeds)  # Combine subjects
    if idx==0:
        GA_LSF_all=all_evokeds[idx]
        #YOU CAN GET THE PEAKS FOR TOPOGRAPHICAL MAPS LATER ON BY ADJUSTING TMIN AND TMAX       
        #print(GA_LSF_all.get_peak(ch_type='eeg',tmin=0.135, tmax=0.145, return_amplitude=True))
        LSF_df_cluster1=GA_LSF_all.to_data_frame(picks=('Oz', 'O1','O2'))
        LSF_df_cluster2=GA_LSF_all.to_data_frame(picks=('PO7', 'PO8'))
        LSF_cluster1=LSF_df_cluster1.mean(1)
        LSF_cluster2=LSF_df_cluster2.mean(1)
        #LSF_cluster1.plot()
        #LSF_cluster2.plot()
    if idx==1:
        GA_HSF_all=all_evokeds[idx]
        #YOU CAN GET THE PEAKS FOR TOPOGRAPHICAL MAPS LATER ON BY ADJUSTING TMIN AND TMAX       
        #print(GA_HSF_all.get_peak(ch_type='eeg',tmin=0.145, tmax=0.155, return_amplitude=True))
        HSF_df_cluster1=GA_HSF_all.to_data_frame(picks=('Oz', 'O1','O2'))
        HSF_df_cluster2=GA_HSF_all.to_data_frame(picks=('PO7', 'PO8'))
        HSF_cluster1=HSF_df_cluster1.mean(1)
        HSF_cluster2=HSF_df_cluster2.mean(1)
        #HSF_cluster1.plot()
        #HSF_cluster2.plot()
    if idx==2:
        GA_contrast_all=all_evokeds[idx]
        contrast_df_cluster1=GA_contrast_all.to_data_frame(picks=('Oz', 'O1','O2'))
        contrast_df_cluster2=GA_contrast_all.to_data_frame(picks=('PO7', 'PO8'))
        contrast_cluster1=contrast_df_cluster1.mean(1)
        contrast_cluster2=contrast_df_cluster2.mean(1)
        contrast_cluster1.plot()
        contrast_cluster2.plot()


mne.evoked.write_evokeds(op.join(evo_path, "GA_LSF_HSF-ave.fif"), all_evokeds)
mne.evoked.write_evokeds(op.join(evo_path, "GA_LSF_HSF-ave.txt"), all_evokeds)


###SOME CONVENIENT CODE F0R VISUALIZATION###
def main():
    """Plot evokeds."""
    for evoked in all_evokeds:
        #evoked.plot()
        df_cluster1=evoked.to_data_frame(picks=('Oz', 'O1','O2'))
        df_cluster2=evoked.to_data_frame(picks=('PO7', 'PO8'))
        cluster1=df_cluster1.mean(1)
        cluster2=df_cluster2.mean(1)
        #cluster1.plot()
        #cluster2.plot()
        #times = np.arange(0.080, 0.110, 0.005)
        #evoked.plot_topomap(times, ch_type='eeg', time_unit='s')

if __name__ == '__main__':
    main()
#%% COMPUTE STATISTICS
T_obs, clusters, cluster_p_values, H0 = \
    permutation_cluster_test([HSF_cluster1, LSF_cluster1],threshold=None, tail=0)


#%% sLORETA
    #inverse is made and applied here. Forward solution and cov matrix were made in section "EVOKED DATA" above
def run_inverse(subject_id):
    subject = "sub%03d" % subject_id
    print("processing subject: %s" % subject)
    evo_path = op.join(data_path, "EEG_Evoked")
    inv_path= op.join(data_path, "EEG_Source")
    for run in range(1, 2):
        fname_ave = op.join(evo_path, 'sub_%03d_LSF_HSF-ave.fif' % (subject_id,))
        fname_cov = op.join(evo_path, 'sub_%03d_LSF_HSF-cov.fif' % (subject_id,))
        fname_fwd = op.join(evo_path, 'sub_%03d_LSF_HSF-fwd.fif' % (subject_id,))
        evokeds = mne.read_evokeds(fname_ave, condition=['evoked_LSF', 'evoked_HSF', 'contrast'])
        cov = mne.read_cov(fname_cov)
        forward = mne.read_forward_solution(fname_fwd)
        info = evokeds[0].info
        #make inverse
        inverse_operator = make_inverse_operator(info, forward, cov, loose=0.2, depth=0.8)
        # Apply inverse
        snr = 3.0
        lambda2 = 1.0 / snr ** 2
        for evoked in evokeds:
            #add 'pick_ori='vector'' to apply_inverse for vector source estimate. Change name to save as well.
            stc = apply_inverse(evoked, inverse_operator, lambda2, "sLORETA")
            stc.save(op.join(inv_path, 'mne_sLORETA_inverse-%s-%03d'
                         % (evoked.comment, subject_id)))
            
parallel, run_func, _ = parallel_func(run_inverse, n_jobs=1)
parallel(run_func(subject_id) for subject_id in range(1, 4)) #Change second number to N=participants+1 

#%%GROUP AVERAGE SLORETA
inv_path = op.join(data_path, "EEG_Source")

def ave_stc(subject_id):
    subject = "sub_%03d" % subject_id
    print("processing subject: %s" % subject)
    inv_path = op.join(data_path, "EEG_Source")
    for condition in ('evoked_LSF', 'evoked_HSF', 'contrast'):
        #change file name to: 'mne_sLORETA_inverse-vector-%s-%03d' for vector source estimate
        source = mne.read_source_estimate(op.join(inv_path, 'mne_sLORETA_inverse-%s-%03d'
                    % (condition, subject_id)))
        print(source.data)
        
        if condition=='contrast':
            out=source
        
    return out
parallel, run_func, _ = parallel_func(ave_stc, n_jobs=1)
stcs=parallel(run_func(subject_id) for subject_id in range(1, 4))
data = np.average([s.data for s in stcs], axis=0)
#change to VectorSourceEstimate for vectors
stc = mne.SourceEstimate(data, stcs[0].vertices,
                               stcs[0].tmin, stcs[0].tstep, stcs[0].subject)
#delete vertno_max and time_max for vector estimate. Comment lines out when necessary
vertno_max, time_max = stc.get_peak(hemi=None, tmin=0.0, tmax=0.1)
print(time_max)
brain = stc.plot(views='lat', hemi='split', size=(800,400), subject='fsaverage', initial_time=time_max, time_viewer=False)
brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue',
               scale_factor=0.6, alpha=0.5)
plt.figure()
plt.plot(1e3 * stc.times, stc.data[::100, :].T)
plt.xlabel('time (ms)')
plt.ylabel('sLORETA value')
plt.show()


#%% SPATIOTEMPORAL CLUSTERING. !!!!!!!!!!!!!!!!!!!!TOOK TOO MUCH MEMORY. WON'T RUN IT ANYMORE!!!!!!!!!!!!
print(stc.data.shape) #run def above with 'evoked_LSF'
print(stc2.data.shape)  #run def above with 'evoked_HSF' and saved it for once in stc2
n_vertices_sample, n_times = stc2.data.shape
n_subjects = 3

np.random.seed(0)
X = randn(n_vertices_sample, n_times, n_subjects, 2) * 10
X[:, :, :, 0] += stc2.data[:, :, np.newaxis]
X[:, :, :, 1] += stc.data[:, :, np.newaxis]

X = np.abs(X)  # only magnitude
X = X[:, :, :, 0] - X[:, :, :, 1]  # make paired contrast
X = np.transpose(X, [2, 1, 0])
print(X)
print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu = \
    spatio_temporal_cluster_1samp_test(X, n_jobs=1, threshold=0.05)

