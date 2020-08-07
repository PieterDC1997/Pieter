I made this repository 2 months ago for a course. All info and code is in here. Also, the (small) paper I wrote for the course is uploaded here as well. That way, you really know what preprocessing steps I used, why, and some more MNE-background info. 

The 'pipeline' is based on 3 raw BDF files with EEG data. I made the script parallel for all 3 participants in my pilot study. It contains preprocess steps, grand averages, ERP plots and source localisation.

The user should ofc change the name of the cwd, line 39 in the main script, and line 25 in the single subject script, 
to own preferences.

You better make some directories

•	A directory called ‘Subjects’, were the you store your raw BDF files.

•	An empty ‘EEG_Process’

•	An empty ‘EEG_Evoked’

•	An empty ‘EEG_Source’

But you can change the organization of the directories, ofc.

The single subject script just adds some more visualization, for a single subject. But the idea behind is basically the same as the main
script, so if you'd like, you can skip this script and fully focus on the main script.


###UPDATE###

I started working on a script to implement TRF's and compute encoding accuracies in a dataset of Holdgraf, based on this repo: https://github.com/choldgraf/paper-encoding_decoding_electrophysiology . I adapted the code and made some extra speech features. It's all in the directory paper-encoding_decoding_electrophysiology-master . Make sure you add file 'modelingtools' to site-packages in python.  


Cheers,
Pieter
