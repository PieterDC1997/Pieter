I made this repository for a course in Ghent. All info and code is in here. Also, the (small) paper I wrote for the course is uploaded here as well. That way, you really know what preprocessing steps I used, why, and some more MNE-background info. 

The code contains preprocess steps, grand averages, ERP plots and source localisation.

The user should change the path to the raw BDF files, line 39 in the main script, and line 25 in the single subject script.

The user should make some directories:

•	A directory called ‘Subjects’, were the you store your raw BDF files.

•	An empty ‘EEG_Process’

•	An empty ‘EEG_Evoked’

•	An empty ‘EEG_Source’

But you can change the organization of the directories.

The single subject script just adds some more visualization, for a single subject. But the idea behind is basically the same as the main
script, so if you'd like, you can skip this script and fully focus on the main script.


Cheers,
Pieter
