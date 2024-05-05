#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on Sun May  5 16:41:57 2024
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# Run 'Before Experiment' code from lsl_initiation
# Initiate LSL
import pylsl

# Make stream outlets & info for each "marker" we want to push, and a corresponding outlet
print('Creating a new streaminfo...')
screen_info = pylsl.StreamInfo('screen_markers', 'screen_pres', 1, 0, 'string')
behav_info = pylsl.StreamInfo('button_press', 'beh', 1, 0, 'string')

print('Opening an outlet...')
screen_outlet = pylsl.StreamOutlet(screen_info)
behav_outlet = pylsl.StreamOutlet(behav_info)

print("now sending markers...")
screen_markers = [['Fixation'], ['Target'], ['ITI']]
behav_markers = [['Correct'], ['Incorrect'], ['Repeat']]
condition_markers = [['Identical'], ['Color_Change'], ['Locat_Change']]
# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'smts'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/Merle/Desktop/Merle/Medizin/Promotion/WP3_SMTS/scripts/smts_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.DATA)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.DATA)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1440, 900], fullscr=True, screen=1,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='ptb')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='PsychToolbox')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "pre_setup" ---
    test_condition = keyboard.Keyboard()
    
    # --- Initialize components for Routine "instruction" ---
    text = visual.TextStim(win=win, name='text',
        text='Willkommen beim SMTS. \n\nWenn Sie bereit sind, drücken Sie die rechte oder die linke Taste, um den Trainingsdurchlauf zu starten',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard()
    # Run 'Begin Experiment' code from setup
    #Setup some of the variables that are used for experiment
    import time
    practice_counter = 0 #keep track of practice trials
    trial_counter = 0 #keep track of test trial we are in
    repeat_trial = False #default is set to continuing with next trial
    
    # actual buttons we observe
    same_button = 'left'
    different_button = 'right'
    
    #Color palette
    potential_colors = ['white', 'black', 'red', 'green', 'blue', 'orange', 'pink', 'purple', 'cyan', 'magenta', 'lightsteelblue', 'yellow', 'lightgreen'] 
    
    #Practice
    response_accuracy_practice = []
    repeat_trial_practice = False
    
    #Test trials
    response_accuracy_trial = []
    
    #Opacity for iti
    opacity_cross = None
    opacity_text = None
    
    #Feedback accuracy for iti
    correct_text = None
    
    #Trial numbers we want to have for practice and trials
    #can be changed as needed
    nReps_trial = 80
    nReps_practice = 15
    
    #Accuracy
    correct_response = None
    
    #Create unique locations so no locations of squares are overlapping
    def unique_locations(upper_limit):
        potential_location = []
        while len(potential_location) < upper_limit:
            potential_coord = 0.1*randint(-3,3)
            if potential_coord != 0 and not(potential_coord in potential_location):
                potential_location.append(potential_coord) 
        return potential_location
    
    #Function that creates four squares in which color/position are set
    def square_manipulation(squares_list, x_coord, y_coord, colors, decider_randomisation, color_or_position, square_to_change, practice_switch):
        if decider_randomisation == 1: # if change
            if color_or_position == 1: #color changes
                change_square = squares_list[square_to_change] 
                change_square.color = colors[4]
                if practice_switch == 0: #only save for trial, not for practice
                    thisExp.addData('label_square', "Color_Change")
            elif color_or_position == 2: #position changes
                change_square = squares_list[square_to_change] 
                change_square.pos = (x_coord[4], y_coord[4])
                if practice_switch == 0:
                    thisExp.addData('label_square', "Locat_Change")
        else:
            if practice_switch == 0: 
                thisExp.addData('label_square', "No_change")
    
    
    
    
    # --- Initialize components for Routine "ITI_preparation_trial" ---
    practice_iti_500 = visual.ShapeStim(
        win=win, name='practice_iti_500', vertices='cross',
        size=(0.01, 0.01),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "sqr_practice" ---
    fix_cross = visual.ShapeStim(
        win=win, name='fix_cross', vertices='cross',
        size=(0.01, 0.01),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    sqr1 = visual.Rect(
        win=win, name='sqr1',
        width=(0.02, 0.02)[0], height=(0.02, 0.02)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    sqr2 = visual.Rect(
        win=win, name='sqr2',
        width=(0.02, 0.02)[0], height=(0.02, 0.02)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='yellow',
        opacity=None, depth=-2.0, interpolate=True)
    sqr3 = visual.Rect(
        win=win, name='sqr3',
        width=(0.02, 0.02)[0], height=(0.02, 0.02)[1],
        ori=0.0, pos=(0.3, 0.3), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=-3.0, interpolate=True)
    sqr4 = visual.Rect(
        win=win, name='sqr4',
        width=(0.02, 0.02)[0], height=(0.02, 0.02)[1],
        ori=0.0, pos=(0.4, 0.4), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-4.0, interpolate=True)
    
    # --- Initialize components for Routine "fixation" ---
    fix = visual.ShapeStim(
        win=win, name='fix', vertices='cross',
        size=(0.01, 0.01),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "square_ident" ---
    fix_cross2 = visual.ShapeStim(
        win=win, name='fix_cross2', vertices='cross',
        size=(0.01, 0.01),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    sqr_ident_1 = visual.Rect(
        win=win, name='sqr_ident_1',
        width=(0.02, 0.02)[0], height=(0.02, 0.02)[1],
        ori=0.0, pos=(-0.1, 0.1), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    sqr_ident_2 = visual.Rect(
        win=win, name='sqr_ident_2',
        width=(0.02, 0.02)[0], height=(0.02, 0.02)[1],
        ori=0.0, pos=(0.2, 0.2), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='yellow',
        opacity=None, depth=-2.0, interpolate=True)
    sqr_ident_3 = visual.Rect(
        win=win, name='sqr_ident_3',
        width=(0.02, 0.02)[0], height=(0.02, 0.02)[1],
        ori=0.0, pos=(0.3, 0.3), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-3.0, interpolate=True)
    sqr_ident_4 = visual.Rect(
        win=win, name='sqr_ident_4',
        width=(0.02, 0.02)[0], height=(0.02, 0.02)[1],
        ori=0.0, pos=(0.4, 0.4), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=-4.0, interpolate=True)
    
    # --- Initialize components for Routine "pause_practice" ---
    next_round_text = visual.TextStim(win=win, name='next_round_text',
        text='Nächste Runde beginnt',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=0.0);
    repeat_cross = visual.ShapeStim(
        win=win, name='repeat_cross', vertices='cross',
        size=(0.01, 0.01),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=1.0, depth=-1.0, interpolate=True)
    accuracy_text = visual.TextStim(win=win, name='accuracy_text',
        text='',
        font='Open Sans',
        pos=(0, -0.2), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "start_test" ---
    starting_test = visual.TextStim(win=win, name='starting_test',
        text='Herzlichen Glückwunsch, Sie haben die Trainingsphase erfolgreich beendet! \n\nBitte geben Sie der Versuchsleiterin Bescheid, um fortzufahren.\n\n',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    beginn_test = keyboard.Keyboard()
    
    # --- Initialize components for Routine "Participation_start_main" ---
    Start_main_trial = visual.TextStim(win=win, name='Start_main_trial',
        text='Wenn Sie bereit sind, drücken Sie die rechte oder die linke Taste, um mit der Testphase zu starten.\n\nViel Erfolg!',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Response_left_right = keyboard.Keyboard()
    
    # --- Initialize components for Routine "ITI_preparation_trial" ---
    practice_iti_500 = visual.ShapeStim(
        win=win, name='practice_iti_500', vertices='cross',
        size=(0.01, 0.01),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "sqr_trial" ---
    fix_cross_trial = visual.ShapeStim(
        win=win, name='fix_cross_trial', vertices='cross',
        size=(0.01, 0.01),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    sqr1_trial = visual.Rect(
        win=win, name='sqr1_trial',
        width=(0.02, 0.02)[0], height=(0.02, 0.02)[1],
        ori=0.0, pos=(0.1, 0.1), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    sqr2_trial = visual.Rect(
        win=win, name='sqr2_trial',
        width=(0.02, 0.02)[0], height=(0.02, 0.02)[1],
        ori=0.0, pos=(0.2, 0.2), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='green', fillColor='green',
        opacity=None, depth=-2.0, interpolate=True)
    sqr3_trial = visual.Rect(
        win=win, name='sqr3_trial',
        width=(0.02, 0.02)[0], height=(0.02, 0.02)[1],
        ori=0.0, pos=(0.3, 0.3), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='red', fillColor='red',
        opacity=None, depth=-3.0, interpolate=True)
    sqr4_trial = visual.Rect(
        win=win, name='sqr4_trial',
        width=(0.02, 0.02)[0], height=(0.02, 0.02)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='blue', fillColor='blue',
        opacity=None, depth=-4.0, interpolate=True)
    
    # --- Initialize components for Routine "fixation_trial" ---
    fix_cross5 = visual.ShapeStim(
        win=win, name='fix_cross5', vertices='cross',
        size=(0.01, 0.01),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "sqr_ident_trial" ---
    fix_cross_4 = visual.ShapeStim(
        win=win, name='fix_cross_4', vertices='cross',
        size=(0.01, 0.01),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    sqr_ident_1_trial = visual.Rect(
        win=win, name='sqr_ident_1_trial',
        width=(0.02, 0.02)[0], height=(0.02, 0.02)[1],
        ori=0.0, pos=(0.1, 0.1), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    sqr_ident_2_trial = visual.Rect(
        win=win, name='sqr_ident_2_trial',
        width=(0.02, 0.02)[0], height=(0.02, 0.02)[1],
        ori=0.0, pos=(0.2, 0.2), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='green', fillColor='green',
        opacity=None, depth=-2.0, interpolate=True)
    sqr_ident_3_trial = visual.Rect(
        win=win, name='sqr_ident_3_trial',
        width=(0.02, 0.02)[0], height=(0.02, 0.02)[1],
        ori=0.0, pos=(0.3, 0.3), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='red', fillColor='red',
        opacity=None, depth=-3.0, interpolate=True)
    sqr_ident_4_trial = visual.Rect(
        win=win, name='sqr_ident_4_trial',
        width=(0.02, 0.02)[0], height=(0.02, 0.02)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='blue', fillColor='blue',
        opacity=None, depth=-4.0, interpolate=True)
    
    # --- Initialize components for Routine "pause_trial" ---
    next_round_text_trial = visual.TextStim(win=win, name='next_round_text_trial',
        text='Nächste Runde beginnt',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-1.0);
    repeat_trial_cross = visual.ShapeStim(
        win=win, name='repeat_trial_cross', vertices='cross',
        size=(0.01, 0.01),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=1.0, depth=-2.0, interpolate=True)
    
    # --- Initialize components for Routine "goodbye" ---
    goodbye_text = visual.TextStim(win=win, name='goodbye_text',
        text='Vielen Dank für die Teilnahme!',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    end_experiment = keyboard.Keyboard()
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "pre_setup" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('pre_setup.started', globalClock.getTime())
    test_condition.keys = []
    test_condition.rt = []
    _test_condition_allKeys = []
    # keep track of which components have finished
    pre_setupComponents = [test_condition]
    for thisComponent in pre_setupComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "pre_setup" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *test_condition* updates
        waitOnFlip = False
        
        # if test_condition is starting this frame...
        if test_condition.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            test_condition.frameNStart = frameN  # exact frame index
            test_condition.tStart = t  # local t and not account for scr refresh
            test_condition.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(test_condition, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'test_condition.started')
            # update status
            test_condition.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(test_condition.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(test_condition.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if test_condition.status == STARTED and not waitOnFlip:
            theseKeys = test_condition.getKeys(keyList=['1','2'], ignoreKeys=["escape"], waitRelease=False)
            _test_condition_allKeys.extend(theseKeys)
            if len(_test_condition_allKeys):
                test_condition.keys = _test_condition_allKeys[-1].name  # just the last key pressed
                test_condition.rt = _test_condition_allKeys[-1].rt
                test_condition.duration = _test_condition_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in pre_setupComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "pre_setup" ---
    for thisComponent in pre_setupComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('pre_setup.stopped', globalClock.getTime())
    # check responses
    if test_condition.keys in ['', [], None]:  # No response was made
        test_condition.keys = None
    thisExp.addData('test_condition.keys',test_condition.keys)
    if test_condition.keys != None:  # we had a response
        thisExp.addData('test_condition.rt', test_condition.rt)
        thisExp.addData('test_condition.duration', test_condition.duration)
    thisExp.nextEntry()
    # the Routine "pre_setup" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instruction" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instruction.started', globalClock.getTime())
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # Run 'Begin Routine' code from setup
    #Set seed -> every participants gets same randomization
    if '1' in test_condition.keys: 
        print('hello condition 1')
        np.random.seed(42)
        thisExp.addData('Condition', 'First')
    elif '2' in test_condition.keys: 
        print('hello condition 2')
        np.random.seed(69)
        thisExp.addData('Condition', 'Second')
    
    #Set up lists with possible conditions for practice and trials
    #Conditions: Squares stay identical, color change, location change, both change
    practice_list = [0]*5 + [1]*5 + [2]*5  #15 practice trials
    shuffle(practice_list)
    test_list = [0]*40 + [1]*20 + [2]*20 #80 test trials
    shuffle(test_list)
    
    # keep track of which components have finished
    instructionComponents = [text, key_resp]
    for thisComponent in instructionComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruction" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruction" ---
    for thisComponent in instructionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instruction.stopped', globalClock.getTime())
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "instruction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "ITI_preparation_trial" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('ITI_preparation_trial.started', globalClock.getTime())
    # Run 'Begin Routine' code from lsl_iti
    #sending first iti for practice
    #screen_outlet.push_sample([screen_markers[2]])
    
    
    # keep track of which components have finished
    ITI_preparation_trialComponents = [practice_iti_500]
    for thisComponent in ITI_preparation_trialComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "ITI_preparation_trial" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.5:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *practice_iti_500* updates
        
        # if practice_iti_500 is starting this frame...
        if practice_iti_500.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            practice_iti_500.frameNStart = frameN  # exact frame index
            practice_iti_500.tStart = t  # local t and not account for scr refresh
            practice_iti_500.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(practice_iti_500, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'practice_iti_500.started')
            # update status
            practice_iti_500.status = STARTED
            practice_iti_500.setAutoDraw(True)
        
        # if practice_iti_500 is active this frame...
        if practice_iti_500.status == STARTED:
            # update params
            pass
        
        # if practice_iti_500 is stopping this frame...
        if practice_iti_500.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > practice_iti_500.tStartRefresh + 1.5-frameTolerance:
                # keep track of stop time/frame for later
                practice_iti_500.tStop = t  # not accounting for scr refresh
                practice_iti_500.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_iti_500.stopped')
                # update status
                practice_iti_500.status = FINISHED
                practice_iti_500.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in ITI_preparation_trialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "ITI_preparation_trial" ---
    for thisComponent in ITI_preparation_trialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('ITI_preparation_trial.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.500000)
    
    # set up handler to look after randomisation of conditions etc
    trials_practice = data.TrialHandler(nReps=100.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='trials_practice')
    thisExp.addLoop(trials_practice)  # add the loop to the experiment
    thisTrials_practice = trials_practice.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_practice.rgb)
    if thisTrials_practice != None:
        for paramName in thisTrials_practice:
            globals()[paramName] = thisTrials_practice[paramName]
    
    for thisTrials_practice in trials_practice:
        currentLoop = trials_practice
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_practice.rgb)
        if thisTrials_practice != None:
            for paramName in thisTrials_practice:
                globals()[paramName] = thisTrials_practice[paramName]
        
        # --- Prepare to start Routine "sqr_practice" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('sqr_practice.started', globalClock.getTime())
        sqr1.setPos((0.1, 0.1))
        sqr2.setPos((0.2, 0.2))
        # Run 'Begin Routine' code from practice_location_color
        #Randomize locations of squares
        
        #Create list with four squares and one with potential colors to choose from
        squares_test = [sqr1, sqr2, sqr3, sqr4]
        
        #If task is not repeated, go on to next one
        if repeat_trial_practice == False: 
             x_coord = unique_locations(5) #Create one extra position in case location changes
             y_coord = unique_locations(5) #Create one extra position in case location changes
             color_indices = randchoice(len(potential_colors), 5, replace = False) #get 5 indices to create one extra color
             colors = [potential_colors[i] for i in color_indices] #select shuffled colours
             if practice_list[practice_counter] == 0:
                decider_randomisation = 0 #squares stay the same
                color_or_position = 0
             elif practice_list[practice_counter] != 0:
                decider_randomisation = 1 #squares change (color, position, both)
                color_or_position = practice_list[practice_counter]
             square_to_change = randint(0,4) # for which square
             practice_counter = practice_counter + 1
        
        # set first squares with respective color and location
        # the created extra 5th location/color is not used here
        for i in range(len(squares_test)):
             current_square = squares_test[i]
             current_square.pos = (x_coord[i], y_coord[i])
             current_square.color = colors[i]
        
        # keep track of which components have finished
        sqr_practiceComponents = [fix_cross, sqr1, sqr2, sqr3, sqr4]
        for thisComponent in sqr_practiceComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "sqr_practice" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.1:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fix_cross* updates
            
            # if fix_cross is starting this frame...
            if fix_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fix_cross.frameNStart = frameN  # exact frame index
                fix_cross.tStart = t  # local t and not account for scr refresh
                fix_cross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix_cross, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fix_cross.started')
                # update status
                fix_cross.status = STARTED
                fix_cross.setAutoDraw(True)
            
            # if fix_cross is active this frame...
            if fix_cross.status == STARTED:
                # update params
                pass
            
            # if fix_cross is stopping this frame...
            if fix_cross.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fix_cross.tStartRefresh + 0.1-frameTolerance:
                    # keep track of stop time/frame for later
                    fix_cross.tStop = t  # not accounting for scr refresh
                    fix_cross.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fix_cross.stopped')
                    # update status
                    fix_cross.status = FINISHED
                    fix_cross.setAutoDraw(False)
            
            # *sqr1* updates
            
            # if sqr1 is starting this frame...
            if sqr1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sqr1.frameNStart = frameN  # exact frame index
                sqr1.tStart = t  # local t and not account for scr refresh
                sqr1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sqr1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sqr1.started')
                # update status
                sqr1.status = STARTED
                sqr1.setAutoDraw(True)
            
            # if sqr1 is active this frame...
            if sqr1.status == STARTED:
                # update params
                pass
            
            # if sqr1 is stopping this frame...
            if sqr1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sqr1.tStartRefresh + 0.1-frameTolerance:
                    # keep track of stop time/frame for later
                    sqr1.tStop = t  # not accounting for scr refresh
                    sqr1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sqr1.stopped')
                    # update status
                    sqr1.status = FINISHED
                    sqr1.setAutoDraw(False)
            
            # *sqr2* updates
            
            # if sqr2 is starting this frame...
            if sqr2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sqr2.frameNStart = frameN  # exact frame index
                sqr2.tStart = t  # local t and not account for scr refresh
                sqr2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sqr2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sqr2.started')
                # update status
                sqr2.status = STARTED
                sqr2.setAutoDraw(True)
            
            # if sqr2 is active this frame...
            if sqr2.status == STARTED:
                # update params
                pass
            
            # if sqr2 is stopping this frame...
            if sqr2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sqr2.tStartRefresh + 0.1-frameTolerance:
                    # keep track of stop time/frame for later
                    sqr2.tStop = t  # not accounting for scr refresh
                    sqr2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sqr2.stopped')
                    # update status
                    sqr2.status = FINISHED
                    sqr2.setAutoDraw(False)
            
            # *sqr3* updates
            
            # if sqr3 is starting this frame...
            if sqr3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sqr3.frameNStart = frameN  # exact frame index
                sqr3.tStart = t  # local t and not account for scr refresh
                sqr3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sqr3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sqr3.started')
                # update status
                sqr3.status = STARTED
                sqr3.setAutoDraw(True)
            
            # if sqr3 is active this frame...
            if sqr3.status == STARTED:
                # update params
                pass
            
            # if sqr3 is stopping this frame...
            if sqr3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sqr3.tStartRefresh + 0.1-frameTolerance:
                    # keep track of stop time/frame for later
                    sqr3.tStop = t  # not accounting for scr refresh
                    sqr3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sqr3.stopped')
                    # update status
                    sqr3.status = FINISHED
                    sqr3.setAutoDraw(False)
            
            # *sqr4* updates
            
            # if sqr4 is starting this frame...
            if sqr4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sqr4.frameNStart = frameN  # exact frame index
                sqr4.tStart = t  # local t and not account for scr refresh
                sqr4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sqr4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sqr4.started')
                # update status
                sqr4.status = STARTED
                sqr4.setAutoDraw(True)
            
            # if sqr4 is active this frame...
            if sqr4.status == STARTED:
                # update params
                pass
            
            # if sqr4 is stopping this frame...
            if sqr4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sqr4.tStartRefresh + 0.1-frameTolerance:
                    # keep track of stop time/frame for later
                    sqr4.tStop = t  # not accounting for scr refresh
                    sqr4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sqr4.stopped')
                    # update status
                    sqr4.status = FINISHED
                    sqr4.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in sqr_practiceComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "sqr_practice" ---
        for thisComponent in sqr_practiceComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('sqr_practice.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.100000)
        
        # --- Prepare to start Routine "fixation" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('fixation.started', globalClock.getTime())
        # keep track of which components have finished
        fixationComponents = [fix]
        for thisComponent in fixationComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.9:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fix* updates
            
            # if fix is starting this frame...
            if fix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fix.frameNStart = frameN  # exact frame index
                fix.tStart = t  # local t and not account for scr refresh
                fix.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fix.started')
                # update status
                fix.status = STARTED
                fix.setAutoDraw(True)
            
            # if fix is active this frame...
            if fix.status == STARTED:
                # update params
                pass
            
            # if fix is stopping this frame...
            if fix.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fix.tStartRefresh + 0.9-frameTolerance:
                    # keep track of stop time/frame for later
                    fix.tStop = t  # not accounting for scr refresh
                    fix.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fix.stopped')
                    # update status
                    fix.status = FINISHED
                    fix.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixationComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation" ---
        for thisComponent in fixationComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('fixation.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.900000)
        
        # --- Prepare to start Routine "square_ident" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('square_ident.started', globalClock.getTime())
        # Run 'Begin Routine' code from practice_button_presses
        ###### Randomization of location and color ######
        
        
        #Create list with all four squares that are used for comparison
        squares_test = [sqr_ident_1, sqr_ident_2, sqr_ident_3, sqr_ident_4]
        
        # bools to check if buttons are pressed
        same_button_pressed = False
        different_button_pressed = False
        
        #Register possible buttons
        same_button_time = 0
        different_button_time = 0
        
        #Button counter to avoid multiple registrations of button presses
        same_button_counter = 0
        different_button_counter = 0
        
        # initialize a keyboard listener - won't work if there is an additional response key in the routine...
        kb = keyboard.Keyboard()
        
        #Create variables for position and color
        #Set these to be identical to squares that were first displayed
        for i in range(len(squares_test)):
             current_square = squares_test[i]
             current_square.pos = (x_coord[i], y_coord[i])
             current_square.color = colors[i]
        
        square_manipulation(squares_test, x_coord, y_coord, colors, decider_randomisation, color_or_position, square_to_change, 1)
        
        # keep track of which components have finished
        square_identComponents = [fix_cross2, sqr_ident_1, sqr_ident_2, sqr_ident_3, sqr_ident_4]
        for thisComponent in square_identComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "square_ident" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fix_cross2* updates
            
            # if fix_cross2 is starting this frame...
            if fix_cross2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fix_cross2.frameNStart = frameN  # exact frame index
                fix_cross2.tStart = t  # local t and not account for scr refresh
                fix_cross2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix_cross2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fix_cross2.started')
                # update status
                fix_cross2.status = STARTED
                fix_cross2.setAutoDraw(True)
            
            # if fix_cross2 is active this frame...
            if fix_cross2.status == STARTED:
                # update params
                pass
            
            # *sqr_ident_1* updates
            
            # if sqr_ident_1 is starting this frame...
            if sqr_ident_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sqr_ident_1.frameNStart = frameN  # exact frame index
                sqr_ident_1.tStart = t  # local t and not account for scr refresh
                sqr_ident_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sqr_ident_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sqr_ident_1.started')
                # update status
                sqr_ident_1.status = STARTED
                sqr_ident_1.setAutoDraw(True)
            
            # if sqr_ident_1 is active this frame...
            if sqr_ident_1.status == STARTED:
                # update params
                pass
            
            # *sqr_ident_2* updates
            
            # if sqr_ident_2 is starting this frame...
            if sqr_ident_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sqr_ident_2.frameNStart = frameN  # exact frame index
                sqr_ident_2.tStart = t  # local t and not account for scr refresh
                sqr_ident_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sqr_ident_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sqr_ident_2.started')
                # update status
                sqr_ident_2.status = STARTED
                sqr_ident_2.setAutoDraw(True)
            
            # if sqr_ident_2 is active this frame...
            if sqr_ident_2.status == STARTED:
                # update params
                pass
            
            # *sqr_ident_3* updates
            
            # if sqr_ident_3 is starting this frame...
            if sqr_ident_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sqr_ident_3.frameNStart = frameN  # exact frame index
                sqr_ident_3.tStart = t  # local t and not account for scr refresh
                sqr_ident_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sqr_ident_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sqr_ident_3.started')
                # update status
                sqr_ident_3.status = STARTED
                sqr_ident_3.setAutoDraw(True)
            
            # if sqr_ident_3 is active this frame...
            if sqr_ident_3.status == STARTED:
                # update params
                pass
            
            # *sqr_ident_4* updates
            
            # if sqr_ident_4 is starting this frame...
            if sqr_ident_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sqr_ident_4.frameNStart = frameN  # exact frame index
                sqr_ident_4.tStart = t  # local t and not account for scr refresh
                sqr_ident_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sqr_ident_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sqr_ident_4.started')
                # update status
                sqr_ident_4.status = STARTED
                sqr_ident_4.setAutoDraw(True)
            
            # if sqr_ident_4 is active this frame...
            if sqr_ident_4.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from practice_button_presses
            #Create conditions that register button presses
            #Decide whether last trial is repeated or if we go to the next trial
            
            
            keys = kb.getKeys([same_button, different_button], waitRelease = False, clear = False)
            
            for key in keys:
                if key == same_button and key.duration == None and same_button_counter == 0: # "left" is pressed and no duration assigned yet
                    same_button_pressed = True
                    same_button_counter = 1 
                if key == different_button and key.duration == None and different_button_counter == 0: # "right"is pressed and no duration assigned yet
                    different_button_pressed = True
                    different_button_counter = 1
                if same_button_pressed == True and different_button_pressed == True: #left and right are pressed at the same time
                    repeat_trial_practice = True
                    continueRoutine = False
                    different_button_counter = 1
                    same_button_counter = 1
                #check if button is released and no second button pressed during that time
                if different_button_counter == 1:
                    if key == different_button and key.duration != None: # "right" button pressed & released while duration is there
                        repeat_trial_practice = False
                        continueRoutine = False
                        different_button_time = key.tDown
                if same_button_counter == 1:
                    if key == same_button and key.duration != None: # "left" button pressed and released with duration 
                        repeat_trial_practice = False
                        continueRoutine = False
                    same_button_time = key.tDown          
            
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in square_identComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "square_ident" ---
        for thisComponent in square_identComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('square_ident.stopped', globalClock.getTime())
        # Run 'End Routine' code from practice_button_presses
        #Create variable for accuracy (correct button presses)
        #Squares are identical -> decider_randomisation == 0 -> "left" 
        #Squares are not identical (change of color/location) -> decider_randomisation != 0 -> "right" 
        kb.clearEvents()
        
        #Save trial number in .csv file 
        thisExp.addData('trial_number_practice', practice_counter)
        
        if different_button_pressed != 0 and same_button_pressed != 0 :  #"left" and "right" pressed: repeat trial
            opacity_text = 0
            opacity_cross = 1
            thisExp.addData('response_accuracy_practice', "Repeat/Checking")   
            correct_text = None
        elif same_button_time != 0 and decider_randomisation == 0: #squares identical
            correct_response = True
            response_accuracy_practice.append(1)
            thisExp.addData('response_accuracy_practice', "Correct")
            opacity_text = 1
            opacity_cross = 0
            correct_text = 'Richtig'
        elif different_button_time != 0 and decider_randomisation != 0: #squares not identical
            correct_response = True    
            response_accuracy_practice.append(1)
            thisExp.addData('response_accuracy_practice', "Correct")
            opacity_text = 1
            opacity_cross = 0
            correct_text = 'Richtig'
        else: 
            correct_response = False    
            response_accuracy_practice.append(0) #wrong button was pressed
            thisExp.addData('response_accuracy_practice', "Incorrect")
            opacity_text = 1
            opacity_cross = 0
            correct_text = 'Falsch'
        
        #If at last trial, we want to continue without showing text
        #Also continue to main part of experiment
        if practice_counter == nReps_practice:
            opacity_text = 0
            opacity_cross = 0
            trials_practice.finished = True
            correct_text = None
        
            
        
        # the Routine "square_ident" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "pause_practice" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('pause_practice.started', globalClock.getTime())
        next_round_text.setOpacity(opacity_text)
        repeat_cross.setOpacity(opacity_cross)
        # keep track of which components have finished
        pause_practiceComponents = [next_round_text, repeat_cross, accuracy_text]
        for thisComponent in pause_practiceComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "pause_practice" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *next_round_text* updates
            
            # if next_round_text is starting this frame...
            if next_round_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                next_round_text.frameNStart = frameN  # exact frame index
                next_round_text.tStart = t  # local t and not account for scr refresh
                next_round_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(next_round_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'next_round_text.started')
                # update status
                next_round_text.status = STARTED
                next_round_text.setAutoDraw(True)
            
            # if next_round_text is active this frame...
            if next_round_text.status == STARTED:
                # update params
                pass
            
            # if next_round_text is stopping this frame...
            if next_round_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > next_round_text.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    next_round_text.tStop = t  # not accounting for scr refresh
                    next_round_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'next_round_text.stopped')
                    # update status
                    next_round_text.status = FINISHED
                    next_round_text.setAutoDraw(False)
            
            # *repeat_cross* updates
            
            # if repeat_cross is starting this frame...
            if repeat_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                repeat_cross.frameNStart = frameN  # exact frame index
                repeat_cross.tStart = t  # local t and not account for scr refresh
                repeat_cross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(repeat_cross, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'repeat_cross.started')
                # update status
                repeat_cross.status = STARTED
                repeat_cross.setAutoDraw(True)
            
            # if repeat_cross is active this frame...
            if repeat_cross.status == STARTED:
                # update params
                pass
            
            # if repeat_cross is stopping this frame...
            if repeat_cross.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > repeat_cross.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    repeat_cross.tStop = t  # not accounting for scr refresh
                    repeat_cross.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'repeat_cross.stopped')
                    # update status
                    repeat_cross.status = FINISHED
                    repeat_cross.setAutoDraw(False)
            
            # *accuracy_text* updates
            
            # if accuracy_text is starting this frame...
            if accuracy_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                accuracy_text.frameNStart = frameN  # exact frame index
                accuracy_text.tStart = t  # local t and not account for scr refresh
                accuracy_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(accuracy_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'accuracy_text.started')
                # update status
                accuracy_text.status = STARTED
                accuracy_text.setAutoDraw(True)
            
            # if accuracy_text is active this frame...
            if accuracy_text.status == STARTED:
                # update params
                accuracy_text.setText(correct_text, log=False)
            
            # if accuracy_text is stopping this frame...
            if accuracy_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > accuracy_text.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    accuracy_text.tStop = t  # not accounting for scr refresh
                    accuracy_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'accuracy_text.stopped')
                    # update status
                    accuracy_text.status = FINISHED
                    accuracy_text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in pause_practiceComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "pause_practice" ---
        for thisComponent in pause_practiceComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('pause_practice.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.500000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 100.0 repeats of 'trials_practice'
    
    # get names of stimulus parameters
    if trials_practice.trialList in ([], [None], None):
        params = []
    else:
        params = trials_practice.trialList[0].keys()
    # save data for this loop
    trials_practice.saveAsExcel(filename + '.xlsx', sheetName='trials_practice',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "start_test" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('start_test.started', globalClock.getTime())
    beginn_test.keys = []
    beginn_test.rt = []
    _beginn_test_allKeys = []
    # keep track of which components have finished
    start_testComponents = [starting_test, beginn_test]
    for thisComponent in start_testComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "start_test" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *starting_test* updates
        
        # if starting_test is starting this frame...
        if starting_test.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            starting_test.frameNStart = frameN  # exact frame index
            starting_test.tStart = t  # local t and not account for scr refresh
            starting_test.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(starting_test, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'starting_test.started')
            # update status
            starting_test.status = STARTED
            starting_test.setAutoDraw(True)
        
        # if starting_test is active this frame...
        if starting_test.status == STARTED:
            # update params
            pass
        
        # *beginn_test* updates
        waitOnFlip = False
        
        # if beginn_test is starting this frame...
        if beginn_test.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            beginn_test.frameNStart = frameN  # exact frame index
            beginn_test.tStart = t  # local t and not account for scr refresh
            beginn_test.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(beginn_test, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'beginn_test.started')
            # update status
            beginn_test.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(beginn_test.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(beginn_test.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if beginn_test.status == STARTED and not waitOnFlip:
            theseKeys = beginn_test.getKeys(keyList=['s'], ignoreKeys=["escape"], waitRelease=False)
            _beginn_test_allKeys.extend(theseKeys)
            if len(_beginn_test_allKeys):
                beginn_test.keys = _beginn_test_allKeys[-1].name  # just the last key pressed
                beginn_test.rt = _beginn_test_allKeys[-1].rt
                beginn_test.duration = _beginn_test_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in start_testComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "start_test" ---
    for thisComponent in start_testComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('start_test.stopped', globalClock.getTime())
    # check responses
    if beginn_test.keys in ['', [], None]:  # No response was made
        beginn_test.keys = None
    thisExp.addData('beginn_test.keys',beginn_test.keys)
    if beginn_test.keys != None:  # we had a response
        thisExp.addData('beginn_test.rt', beginn_test.rt)
        thisExp.addData('beginn_test.duration', beginn_test.duration)
    thisExp.nextEntry()
    # the Routine "start_test" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Participation_start_main" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Participation_start_main.started', globalClock.getTime())
    Response_left_right.keys = []
    Response_left_right.rt = []
    _Response_left_right_allKeys = []
    # keep track of which components have finished
    Participation_start_mainComponents = [Start_main_trial, Response_left_right]
    for thisComponent in Participation_start_mainComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Participation_start_main" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Start_main_trial* updates
        
        # if Start_main_trial is starting this frame...
        if Start_main_trial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Start_main_trial.frameNStart = frameN  # exact frame index
            Start_main_trial.tStart = t  # local t and not account for scr refresh
            Start_main_trial.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Start_main_trial, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Start_main_trial.started')
            # update status
            Start_main_trial.status = STARTED
            Start_main_trial.setAutoDraw(True)
        
        # if Start_main_trial is active this frame...
        if Start_main_trial.status == STARTED:
            # update params
            pass
        
        # *Response_left_right* updates
        waitOnFlip = False
        
        # if Response_left_right is starting this frame...
        if Response_left_right.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Response_left_right.frameNStart = frameN  # exact frame index
            Response_left_right.tStart = t  # local t and not account for scr refresh
            Response_left_right.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Response_left_right, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Response_left_right.started')
            # update status
            Response_left_right.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Response_left_right.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Response_left_right.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if Response_left_right.status == STARTED and not waitOnFlip:
            theseKeys = Response_left_right.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
            _Response_left_right_allKeys.extend(theseKeys)
            if len(_Response_left_right_allKeys):
                Response_left_right.keys = _Response_left_right_allKeys[-1].name  # just the last key pressed
                Response_left_right.rt = _Response_left_right_allKeys[-1].rt
                Response_left_right.duration = _Response_left_right_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Participation_start_mainComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Participation_start_main" ---
    for thisComponent in Participation_start_mainComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Participation_start_main.stopped', globalClock.getTime())
    # check responses
    if Response_left_right.keys in ['', [], None]:  # No response was made
        Response_left_right.keys = None
    thisExp.addData('Response_left_right.keys',Response_left_right.keys)
    if Response_left_right.keys != None:  # we had a response
        thisExp.addData('Response_left_right.rt', Response_left_right.rt)
        thisExp.addData('Response_left_right.duration', Response_left_right.duration)
    thisExp.nextEntry()
    # the Routine "Participation_start_main" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "ITI_preparation_trial" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('ITI_preparation_trial.started', globalClock.getTime())
    # Run 'Begin Routine' code from lsl_iti
    #sending first iti for practice
    #screen_outlet.push_sample([screen_markers[2]])
    
    
    # keep track of which components have finished
    ITI_preparation_trialComponents = [practice_iti_500]
    for thisComponent in ITI_preparation_trialComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "ITI_preparation_trial" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.5:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *practice_iti_500* updates
        
        # if practice_iti_500 is starting this frame...
        if practice_iti_500.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            practice_iti_500.frameNStart = frameN  # exact frame index
            practice_iti_500.tStart = t  # local t and not account for scr refresh
            practice_iti_500.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(practice_iti_500, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'practice_iti_500.started')
            # update status
            practice_iti_500.status = STARTED
            practice_iti_500.setAutoDraw(True)
        
        # if practice_iti_500 is active this frame...
        if practice_iti_500.status == STARTED:
            # update params
            pass
        
        # if practice_iti_500 is stopping this frame...
        if practice_iti_500.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > practice_iti_500.tStartRefresh + 1.5-frameTolerance:
                # keep track of stop time/frame for later
                practice_iti_500.tStop = t  # not accounting for scr refresh
                practice_iti_500.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_iti_500.stopped')
                # update status
                practice_iti_500.status = FINISHED
                practice_iti_500.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in ITI_preparation_trialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "ITI_preparation_trial" ---
    for thisComponent in ITI_preparation_trialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('ITI_preparation_trial.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.500000)
    
    # set up handler to look after randomisation of conditions etc
    trials_trial = data.TrialHandler(nReps=300.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='trials_trial')
    thisExp.addLoop(trials_trial)  # add the loop to the experiment
    thisTrials_trial = trials_trial.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_trial.rgb)
    if thisTrials_trial != None:
        for paramName in thisTrials_trial:
            globals()[paramName] = thisTrials_trial[paramName]
    
    for thisTrials_trial in trials_trial:
        currentLoop = trials_trial
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_trial.rgb)
        if thisTrials_trial != None:
            for paramName in thisTrials_trial:
                globals()[paramName] = thisTrials_trial[paramName]
        
        # --- Prepare to start Routine "sqr_trial" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('sqr_trial.started', globalClock.getTime())
        # Run 'Begin Routine' code from trial_location_color
        #Randomize locations of squares
        
        #Create list with four squares and one with potential colors to choose from
        squares_trial = [sqr1_trial, sqr2_trial, sqr3_trial, sqr4_trial]
        
        #If task is not repeated, go on to next one
        #New location, color and decision whether second set of stimuli
        #is going to be the same or with different properties
        if repeat_trial == False:
             x_coord = unique_locations(5) #Create one extra position in case location changes
             y_coord = unique_locations(5) #Create one extra position in case location changes
             color_indices = randchoice(len(potential_colors), 5, replace = False) #get 5 indices to create one extra color
             colors = [potential_colors[i] for i in color_indices] #select shuffled colours
             if test_list[trial_counter] == 0:
                decider_randomisation = 0 #squares stay the same
                color_or_position = 0
             elif test_list[trial_counter] != 0:
                decider_randomisation = 1 #squares change (color, position, both)
                color_or_position = test_list[trial_counter]
             square_to_change = randint(0,4) # for which square
             trial_counter = trial_counter + 1
           
           
        # set first squares with respective color and location
        # the created extra 5th location/color is not used here
        for i in range(len(squares_trial)):
             current_square = squares_trial[i]
             current_square.pos = (x_coord[i], y_coord[i])
             current_square.color = colors[i]
        
        
        
        # keep track of which components have finished
        sqr_trialComponents = [fix_cross_trial, sqr1_trial, sqr2_trial, sqr3_trial, sqr4_trial]
        for thisComponent in sqr_trialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "sqr_trial" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.1:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fix_cross_trial* updates
            
            # if fix_cross_trial is starting this frame...
            if fix_cross_trial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fix_cross_trial.frameNStart = frameN  # exact frame index
                fix_cross_trial.tStart = t  # local t and not account for scr refresh
                fix_cross_trial.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix_cross_trial, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fix_cross_trial.started')
                # update status
                fix_cross_trial.status = STARTED
                fix_cross_trial.setAutoDraw(True)
            
            # if fix_cross_trial is active this frame...
            if fix_cross_trial.status == STARTED:
                # update params
                pass
            
            # if fix_cross_trial is stopping this frame...
            if fix_cross_trial.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fix_cross_trial.tStartRefresh + 0.1-frameTolerance:
                    # keep track of stop time/frame for later
                    fix_cross_trial.tStop = t  # not accounting for scr refresh
                    fix_cross_trial.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fix_cross_trial.stopped')
                    # update status
                    fix_cross_trial.status = FINISHED
                    fix_cross_trial.setAutoDraw(False)
            
            # *sqr1_trial* updates
            
            # if sqr1_trial is starting this frame...
            if sqr1_trial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sqr1_trial.frameNStart = frameN  # exact frame index
                sqr1_trial.tStart = t  # local t and not account for scr refresh
                sqr1_trial.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sqr1_trial, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sqr1_trial.started')
                # update status
                sqr1_trial.status = STARTED
                sqr1_trial.setAutoDraw(True)
            
            # if sqr1_trial is active this frame...
            if sqr1_trial.status == STARTED:
                # update params
                pass
            
            # if sqr1_trial is stopping this frame...
            if sqr1_trial.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sqr1_trial.tStartRefresh + 0.1-frameTolerance:
                    # keep track of stop time/frame for later
                    sqr1_trial.tStop = t  # not accounting for scr refresh
                    sqr1_trial.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sqr1_trial.stopped')
                    # update status
                    sqr1_trial.status = FINISHED
                    sqr1_trial.setAutoDraw(False)
            
            # *sqr2_trial* updates
            
            # if sqr2_trial is starting this frame...
            if sqr2_trial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sqr2_trial.frameNStart = frameN  # exact frame index
                sqr2_trial.tStart = t  # local t and not account for scr refresh
                sqr2_trial.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sqr2_trial, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sqr2_trial.started')
                # update status
                sqr2_trial.status = STARTED
                sqr2_trial.setAutoDraw(True)
            
            # if sqr2_trial is active this frame...
            if sqr2_trial.status == STARTED:
                # update params
                pass
            
            # if sqr2_trial is stopping this frame...
            if sqr2_trial.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sqr2_trial.tStartRefresh + 0.1-frameTolerance:
                    # keep track of stop time/frame for later
                    sqr2_trial.tStop = t  # not accounting for scr refresh
                    sqr2_trial.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sqr2_trial.stopped')
                    # update status
                    sqr2_trial.status = FINISHED
                    sqr2_trial.setAutoDraw(False)
            
            # *sqr3_trial* updates
            
            # if sqr3_trial is starting this frame...
            if sqr3_trial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sqr3_trial.frameNStart = frameN  # exact frame index
                sqr3_trial.tStart = t  # local t and not account for scr refresh
                sqr3_trial.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sqr3_trial, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sqr3_trial.started')
                # update status
                sqr3_trial.status = STARTED
                sqr3_trial.setAutoDraw(True)
            
            # if sqr3_trial is active this frame...
            if sqr3_trial.status == STARTED:
                # update params
                pass
            
            # if sqr3_trial is stopping this frame...
            if sqr3_trial.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sqr3_trial.tStartRefresh + 0.1-frameTolerance:
                    # keep track of stop time/frame for later
                    sqr3_trial.tStop = t  # not accounting for scr refresh
                    sqr3_trial.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sqr3_trial.stopped')
                    # update status
                    sqr3_trial.status = FINISHED
                    sqr3_trial.setAutoDraw(False)
            
            # *sqr4_trial* updates
            
            # if sqr4_trial is starting this frame...
            if sqr4_trial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sqr4_trial.frameNStart = frameN  # exact frame index
                sqr4_trial.tStart = t  # local t and not account for scr refresh
                sqr4_trial.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sqr4_trial, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sqr4_trial.started')
                # update status
                sqr4_trial.status = STARTED
                sqr4_trial.setAutoDraw(True)
            
            # if sqr4_trial is active this frame...
            if sqr4_trial.status == STARTED:
                # update params
                pass
            
            # if sqr4_trial is stopping this frame...
            if sqr4_trial.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sqr4_trial.tStartRefresh + 0.1-frameTolerance:
                    # keep track of stop time/frame for later
                    sqr4_trial.tStop = t  # not accounting for scr refresh
                    sqr4_trial.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sqr4_trial.stopped')
                    # update status
                    sqr4_trial.status = FINISHED
                    sqr4_trial.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in sqr_trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "sqr_trial" ---
        for thisComponent in sqr_trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('sqr_trial.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.100000)
        
        # --- Prepare to start Routine "fixation_trial" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('fixation_trial.started', globalClock.getTime())
        # Run 'Begin Routine' code from lsl_fixation_trial
        #Push fixation
        screen_outlet.push_sample(screen_markers[0])
        fixation_marker_count = 0
        # keep track of which components have finished
        fixation_trialComponents = [fix_cross5]
        for thisComponent in fixation_trialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation_trial" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.9:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fix_cross5* updates
            
            # if fix_cross5 is starting this frame...
            if fix_cross5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fix_cross5.frameNStart = frameN  # exact frame index
                fix_cross5.tStart = t  # local t and not account for scr refresh
                fix_cross5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix_cross5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fix_cross5.started')
                # update status
                fix_cross5.status = STARTED
                fix_cross5.setAutoDraw(True)
            
            # if fix_cross5 is active this frame...
            if fix_cross5.status == STARTED:
                # update params
                pass
            
            # if fix_cross5 is stopping this frame...
            if fix_cross5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fix_cross5.tStartRefresh + 0.9-frameTolerance:
                    # keep track of stop time/frame for later
                    fix_cross5.tStop = t  # not accounting for scr refresh
                    fix_cross5.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fix_cross5.stopped')
                    # update status
                    fix_cross5.status = FINISHED
                    fix_cross5.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixation_trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation_trial" ---
        for thisComponent in fixation_trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('fixation_trial.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.900000)
        
        # --- Prepare to start Routine "sqr_ident_trial" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('sqr_ident_trial.started', globalClock.getTime())
        # Run 'Begin Routine' code from trial_button_presses
        ###### Randomization of location and color ######
        
        #Placeholder for correctness of participant's answer
        correct_response = None
        
        #Bools to check if buttons are pressed
        same_button_pressed = False
        different_button_pressed = False
        
        #Variable for timing of button presses/release time
        same_button_time = 0
        different_button_time = 0
        
        #Button counter to avoid multiple registrations of button presses
        same_button_counter = 0
        different_button_counter = 0
        
        #Create list with all four squares that are used for comparison
        squares_test = [sqr_ident_1_trial, sqr_ident_2_trial, sqr_ident_3_trial, sqr_ident_4_trial]
        
        #Create variables for position and color
        #Set these to be identical to squares that were first displayed
        for i in range(len(squares_test)):
             current_square = squares_test[i]
             current_square.pos = (x_coord[i], y_coord[i])
             current_square.color = colors[i]
        
        square_manipulation(squares_test, x_coord, y_coord, colors, decider_randomisation, color_or_position, square_to_change, 0)
        
        
        # Run 'Begin Routine' code from lsl_trial_accuracy
        #Push screen target
        screen_outlet.push_sample(screen_markers[1])
        target_present_marker_count = 0
        
        
        
        # keep track of which components have finished
        sqr_ident_trialComponents = [fix_cross_4, sqr_ident_1_trial, sqr_ident_2_trial, sqr_ident_3_trial, sqr_ident_4_trial]
        for thisComponent in sqr_ident_trialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "sqr_ident_trial" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fix_cross_4* updates
            
            # if fix_cross_4 is starting this frame...
            if fix_cross_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fix_cross_4.frameNStart = frameN  # exact frame index
                fix_cross_4.tStart = t  # local t and not account for scr refresh
                fix_cross_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix_cross_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fix_cross_4.started')
                # update status
                fix_cross_4.status = STARTED
                fix_cross_4.setAutoDraw(True)
            
            # if fix_cross_4 is active this frame...
            if fix_cross_4.status == STARTED:
                # update params
                pass
            
            # *sqr_ident_1_trial* updates
            
            # if sqr_ident_1_trial is starting this frame...
            if sqr_ident_1_trial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sqr_ident_1_trial.frameNStart = frameN  # exact frame index
                sqr_ident_1_trial.tStart = t  # local t and not account for scr refresh
                sqr_ident_1_trial.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sqr_ident_1_trial, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sqr_ident_1_trial.started')
                # update status
                sqr_ident_1_trial.status = STARTED
                sqr_ident_1_trial.setAutoDraw(True)
            
            # if sqr_ident_1_trial is active this frame...
            if sqr_ident_1_trial.status == STARTED:
                # update params
                pass
            
            # *sqr_ident_2_trial* updates
            
            # if sqr_ident_2_trial is starting this frame...
            if sqr_ident_2_trial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sqr_ident_2_trial.frameNStart = frameN  # exact frame index
                sqr_ident_2_trial.tStart = t  # local t and not account for scr refresh
                sqr_ident_2_trial.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sqr_ident_2_trial, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sqr_ident_2_trial.started')
                # update status
                sqr_ident_2_trial.status = STARTED
                sqr_ident_2_trial.setAutoDraw(True)
            
            # if sqr_ident_2_trial is active this frame...
            if sqr_ident_2_trial.status == STARTED:
                # update params
                pass
            
            # *sqr_ident_3_trial* updates
            
            # if sqr_ident_3_trial is starting this frame...
            if sqr_ident_3_trial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sqr_ident_3_trial.frameNStart = frameN  # exact frame index
                sqr_ident_3_trial.tStart = t  # local t and not account for scr refresh
                sqr_ident_3_trial.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sqr_ident_3_trial, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sqr_ident_3_trial.started')
                # update status
                sqr_ident_3_trial.status = STARTED
                sqr_ident_3_trial.setAutoDraw(True)
            
            # if sqr_ident_3_trial is active this frame...
            if sqr_ident_3_trial.status == STARTED:
                # update params
                pass
            
            # *sqr_ident_4_trial* updates
            
            # if sqr_ident_4_trial is starting this frame...
            if sqr_ident_4_trial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sqr_ident_4_trial.frameNStart = frameN  # exact frame index
                sqr_ident_4_trial.tStart = t  # local t and not account for scr refresh
                sqr_ident_4_trial.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sqr_ident_4_trial, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sqr_ident_4_trial.started')
                # update status
                sqr_ident_4_trial.status = STARTED
                sqr_ident_4_trial.setAutoDraw(True)
            
            # if sqr_ident_4_trial is active this frame...
            if sqr_ident_4_trial.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from trial_button_presses
            #Create conditions that register button presses
            #Decide whether last trial is repeated or if we go to the next trial
                
            keys = kb.getKeys([same_button, different_button], waitRelease=False, clear = False)
            
            for key in keys:
                if key == same_button and key.duration == None and same_button_counter == 0: # "left" is pressed and no duration assigned yet
                    same_button_pressed = True
                    same_button_counter = 1
                if key == different_button and key.duration == None and different_button_counter == 0: # "right"is pressed and no duration assigned yet
                    different_button_pressed = True
                    different_button_counter = 1
                if same_button_pressed == True and different_button_pressed == True: #left and right are pressed at the same time
                    repeat_trial = True
                    continueRoutine = False
                    same_button_counter = 1
                    different_button_counter = 1
                #check if button is released and no second button pressed during that time
                if different_button_counter == 1:
                    if key == different_button and key.duration != None: # "right" button pressed & released while duration is there
                        repeat_trial = False
                        continueRoutine = False
                        different_button_time = key.tDown          
                if same_button_counter == 1:
                    if key == same_button and key.duration != None: # "left" button pressed and released with duration 
                        repeat_trial = False
                        continueRoutine = False
                        same_button_time = key.tDown 
            
                         
            
            
            
            
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in sqr_ident_trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "sqr_ident_trial" ---
        for thisComponent in sqr_ident_trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('sqr_ident_trial.stopped', globalClock.getTime())
        # Run 'End Routine' code from trial_button_presses
        #Create variable for accuracy (correct button presses)
        #Squares are identical -> decider_randomisation == 0 -> "left" 
        #Squares are not identical (change of color/location) -> decider_randomisation != 0 -> "right" 
        kb.clearEvents()
        
        #Save trial number in .csv file 
        thisExp.addData('trial_number_main', trial_counter)
        
        if same_button_pressed != 0 and different_button_pressed != 0: #"left" and "right" pressed: repeat trial
            opacity_text = 0
            opacity_cross = 1
            thisExp.addData('response_accuracy', "Repeat/Checking")
        elif same_button_time != 0 and decider_randomisation == 0: #squares identical
            correct_response = True
            response_accuracy_trial.append(1)
            thisExp.addData('response_accuracy', "Correct")
            opacity_text = 1
            opacity_cross = 0
        elif different_button_time != 0 and decider_randomisation != 0: #squares not identical
            correct_response = True
            response_accuracy_trial.append(1)
            thisExp.addData('response_accuracy', "Correct")
            opacity_text = 1
            opacity_cross = 0
        else: 
            correct_response = False
            response_accuracy_trial.append(0) #wrong button was pressed
            thisExp.addData('response_accuracy', "Incorrect")
            opacity_text = 1
            opacity_cross = 0
        
        #For last trial don't show cross
        #Do not repeat loop but finish experiment
        if trial_counter == nReps_trial: 
            opacity_text = 0
            opacity_cross = 0
            trials_trial.finished = True
        
        #print('-----------------')
        #print('we are in trial no:', trial_counter)
        #print('randomization:', decider_randomisation)
        #print('response:', response_randomisation)
        #if repeat_trial == True: 
        #    print('!!!!REPEAT!!!!')
        #if repeat_trial == False: 
        #    print('list of results:', response_accuracy_trial)
            
        
        # Run 'End Routine' code from lsl_trial_accuracy
        ##Push accuracy and if conditions change
        if repeat_trial == True: 
            behav_outlet.push_sample(behav_markers[2]) #repeated trial
        else:
            #Push whether correct or incorrect button presses
            if correct_response == True: 
                behav_outlet.push_sample(behav_markers[0]) #correct response
            elif correct_response == False: 
                behav_outlet.push_sample(behav_markers[1]) #incorrect response
        
            #Push whether target squares stayed identical or not
            if decider_randomisation == 0:
                screen_outlet.push_sample(condition_markers[0]) #identical squares
            elif decider_randomisation == 1: #color_or_position -> 1 = color change, 2 = location change
                screen_outlet.push_sample(condition_markers[color_or_position])
           
        
        
        
        
            
        
        # the Routine "sqr_ident_trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "pause_trial" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('pause_trial.started', globalClock.getTime())
        # Run 'Begin Routine' code from lsl_iti_trial
        #sending first iti for practice
        screen_outlet.push_sample(screen_markers[2])
        next_round_text_trial.setOpacity(opacity_text)
        repeat_trial_cross.setOpacity(opacity_cross)
        # keep track of which components have finished
        pause_trialComponents = [next_round_text_trial, repeat_trial_cross]
        for thisComponent in pause_trialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "pause_trial" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *next_round_text_trial* updates
            
            # if next_round_text_trial is starting this frame...
            if next_round_text_trial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                next_round_text_trial.frameNStart = frameN  # exact frame index
                next_round_text_trial.tStart = t  # local t and not account for scr refresh
                next_round_text_trial.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(next_round_text_trial, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'next_round_text_trial.started')
                # update status
                next_round_text_trial.status = STARTED
                next_round_text_trial.setAutoDraw(True)
            
            # if next_round_text_trial is active this frame...
            if next_round_text_trial.status == STARTED:
                # update params
                pass
            
            # if next_round_text_trial is stopping this frame...
            if next_round_text_trial.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > next_round_text_trial.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    next_round_text_trial.tStop = t  # not accounting for scr refresh
                    next_round_text_trial.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'next_round_text_trial.stopped')
                    # update status
                    next_round_text_trial.status = FINISHED
                    next_round_text_trial.setAutoDraw(False)
            
            # *repeat_trial_cross* updates
            
            # if repeat_trial_cross is starting this frame...
            if repeat_trial_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                repeat_trial_cross.frameNStart = frameN  # exact frame index
                repeat_trial_cross.tStart = t  # local t and not account for scr refresh
                repeat_trial_cross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(repeat_trial_cross, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'repeat_trial_cross.started')
                # update status
                repeat_trial_cross.status = STARTED
                repeat_trial_cross.setAutoDraw(True)
            
            # if repeat_trial_cross is active this frame...
            if repeat_trial_cross.status == STARTED:
                # update params
                pass
            
            # if repeat_trial_cross is stopping this frame...
            if repeat_trial_cross.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > repeat_trial_cross.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    repeat_trial_cross.tStop = t  # not accounting for scr refresh
                    repeat_trial_cross.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'repeat_trial_cross.stopped')
                    # update status
                    repeat_trial_cross.status = FINISHED
                    repeat_trial_cross.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in pause_trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "pause_trial" ---
        for thisComponent in pause_trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('pause_trial.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.500000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 300.0 repeats of 'trials_trial'
    
    # get names of stimulus parameters
    if trials_trial.trialList in ([], [None], None):
        params = []
    else:
        params = trials_trial.trialList[0].keys()
    # save data for this loop
    trials_trial.saveAsExcel(filename + '.xlsx', sheetName='trials_trial',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "goodbye" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('goodbye.started', globalClock.getTime())
    end_experiment.keys = []
    end_experiment.rt = []
    _end_experiment_allKeys = []
    # keep track of which components have finished
    goodbyeComponents = [goodbye_text, end_experiment]
    for thisComponent in goodbyeComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "goodbye" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *goodbye_text* updates
        
        # if goodbye_text is starting this frame...
        if goodbye_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            goodbye_text.frameNStart = frameN  # exact frame index
            goodbye_text.tStart = t  # local t and not account for scr refresh
            goodbye_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(goodbye_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'goodbye_text.started')
            # update status
            goodbye_text.status = STARTED
            goodbye_text.setAutoDraw(True)
        
        # if goodbye_text is active this frame...
        if goodbye_text.status == STARTED:
            # update params
            pass
        
        # *end_experiment* updates
        waitOnFlip = False
        
        # if end_experiment is starting this frame...
        if end_experiment.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_experiment.frameNStart = frameN  # exact frame index
            end_experiment.tStart = t  # local t and not account for scr refresh
            end_experiment.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_experiment, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_experiment.started')
            # update status
            end_experiment.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(end_experiment.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(end_experiment.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if end_experiment.status == STARTED and not waitOnFlip:
            theseKeys = end_experiment.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
            _end_experiment_allKeys.extend(theseKeys)
            if len(_end_experiment_allKeys):
                end_experiment.keys = _end_experiment_allKeys[-1].name  # just the last key pressed
                end_experiment.rt = _end_experiment_allKeys[-1].rt
                end_experiment.duration = _end_experiment_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in goodbyeComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "goodbye" ---
    for thisComponent in goodbyeComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('goodbye.stopped', globalClock.getTime())
    # check responses
    if end_experiment.keys in ['', [], None]:  # No response was made
        end_experiment.keys = None
    thisExp.addData('end_experiment.keys',end_experiment.keys)
    if end_experiment.keys != None:  # we had a response
        thisExp.addData('end_experiment.rt', end_experiment.rt)
        thisExp.addData('end_experiment.duration', end_experiment.duration)
    thisExp.nextEntry()
    # the Routine "goodbye" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
