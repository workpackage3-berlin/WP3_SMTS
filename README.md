# WP3_SMTS

Square matching to sample (SMTS) task, behavioral task for investigating checking behavior. 

Getting Started

These instructions will get you a copy of the project up and running on your local machine.

Repository

GUI: use a git-manager of preference, and clone: https://github.com/workpackage3-berlin/WP3_SMTS.git
Command line:
Set working directory to desired folder and run: git clone https://github.com/workpackage3-berlin/WP3_SMTS.git
To check initiated remote-repo link, and current branch: cd WP3_SMTS, git init, git remote -v, git branch (switch to branch main e.g. with git checkout main)
Environment Task environment:

Navigate to repo directory, e.g.: cd Users/USERNAME/Research/WP3_SMTS
conda create --name tmsi python==3.9.18 pyqtgraph==0.12.3 pandas==2.1.4 (Confirm Proceed? with y)
conda activate tmsi
pip install edflib-Python==1.0.8 mne==1.6.0 pylsl==1.16.2 pyside2==5.15.2.1 pyxdf==1.16.5
