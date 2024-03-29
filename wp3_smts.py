#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on Fri Mar 22 14:35:51 2024
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
condition_markers = [['Color_Change'], ['Locat_Change'], ['Color_Locat_Change'], ['Identical']]
# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'wp3_smts'  # from the Builder filename that created this script
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
        originPath='/Users/Merle/Desktop/Merle/Medizin/Promotion/WP3_SMTS/wp3_smts.py',
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
            size=[1440, 900], fullscr=False, screen=0,
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
    win.mouseVisible = True
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
    
    # --- Initialize components for Routine "instruction" ---
    text = visual.TextStim(win=win, name='text',
        text='Willkommen beim SMTS. Bei diesem Test werden sie erst vier Quadrate in vier verschiedenen Farben sehen. Nach einer kurzen Pause werden erneut vier Quadrate sehen, bei denen sie folgende Entscheidung treffen sollen:\n- Sind die Quadrate in Farbe und Position identisch? Dann drücken sie "p"\n- Unterscheiden sich die Quadrate in der Farbe, der Position, oder beidem? Dann drücken sie "q"\nMöchten Sie die Quadrate erneut anschauen, weil sie sich nicht sicher sind, dann drücken sie bitte "p" und "q" gleichzeitig.\n\nWenn Sie bereit sind, drücken Sie einen der Knöpfe um den Testdurchlauf zu starten',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard()
    # Run 'Begin Experiment' code from setup
    #Setup some of the variables that are used for experiment
    import time
    grace_period = 0.5 #upper time limit for button presses
    practice_counter = 0 #keep track of practice trials
    trial_counter = 0 #keep track of test trial we are in
    repeat_trial = False #default is set to continuing with next trial
    
    #Set up lists with possible conditions for practice and trials
    #Conditions: Squares stay identical, color change, location change, both change
    practice_list = [0]*4 + [1]*4 + [2]*4 + [3]*4 #16 practice trials
    shuffle(practice_list)
    test_list = [0]*20 + [1]*20 + [2]*20 + [3]*20 #80 test trials
    shuffle(test_list)
    
    #Color palette
    potential_colors = ['white', 'black', 'red', 'green', 'blue', 'orange', 'pink', 'purple', 'cyan', 'magenta', 'lightsteelblue', 'yellow', 'lightgreen'] 
    
    #Create unique locations so no locations of squares are overlapping
    def unique_locations(upper_limit):
        potential_location = []
        while len(potential_location) < upper_limit:
            potential_coord = 0.1*randint(-3,3)
            if potential_coord != 0 and not(potential_coord in potential_location):
                potential_location.append(potential_coord) 
        return potential_location
    
    def square_manipulation(squares_list, x_coord, y_coord, colors, decider_randomisation, color_or_position, square_to_change, practice_switch):
        if decider_randomisation == 1: # if change
            if color_or_position == 0: #color changes
              change_square = squares_list[square_to_change] 
              change_square.color = colors[4]
              if practice_switch == 0: #only save for trial, not for practice
                thisExp.addData('label_square', "Color_Change")
            elif color_or_position == 1: #position changes
              change_square = squares_list[square_to_change] 
              change_square.pos = (x_coord[4], y_coord[4])
              if practice_switch == 0:
                thisExp.addData('label_square', "Locat_Change")
            elif color_or_position == 2: #color and position change
              change_square = squares_list[square_to_change] 
              change_square.pos = (x_coord[4], y_coord[4])
              change_square.color = colors[4]
              if practice_switch == 0:
                thisExp.addData('label_square', "Color_Locat_Change")
    
    #Practice
    response_accuracy_practice = []
    repeat_trial_practice = False
    
    #Test trials
    response_accuracy_trial = []
    
    # --- Initialize components for Routine "Practice_ITI" ---
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
    response_key = keyboard.Keyboard()
    
    # --- Initialize components for Routine "pause_practice" ---
    pause_btw_trials = visual.TextStim(win=win, name='pause_btw_trials',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    fix_cross3 = visual.ShapeStim(
        win=win, name='fix_cross3', vertices='cross',
        size=(0.01, 0.01),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "start_test" ---
    starting_test = visual.TextStim(win=win, name='starting_test',
        text='Nun beginnt der richtige Test',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    beginn_test = keyboard.Keyboard()
    
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
    response_key_trial = keyboard.Keyboard()
    
    # --- Initialize components for Routine "pause_trial" ---
    fix_cross4 = visual.ShapeStim(
        win=win, name='fix_cross4', vertices='cross',
        size=(0.01, 0.01),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    
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
    
    # --- Prepare to start Routine "instruction" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instruction.started', globalClock.getTime())
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
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
    
    # --- Prepare to start Routine "Practice_ITI" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Practice_ITI.started', globalClock.getTime())
    # Run 'Begin Routine' code from lsl_practice_iti
    #sending first iti for practice
    #screen_outlet.push_sample([screen_markers[2]])
    
    
    # keep track of which components have finished
    Practice_ITIComponents = [practice_iti_500]
    for thisComponent in Practice_ITIComponents:
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
    
    # --- Run Routine "Practice_ITI" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 0.5:
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
            if tThisFlipGlobal > practice_iti_500.tStartRefresh + 0.5-frameTolerance:
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
        for thisComponent in Practice_ITIComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Practice_ITI" ---
    for thisComponent in Practice_ITIComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Practice_ITI.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.500000)
    
    # set up handler to look after randomisation of conditions etc
    trials_practice = data.TrialHandler(nReps=10.0, method='random', 
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
        
        # set up handler to look after randomisation of conditions etc
        repeat_last_practice = data.TrialHandler(nReps=5.0, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='repeat_last_practice')
        thisExp.addLoop(repeat_last_practice)  # add the loop to the experiment
        thisRepeat_last_practice = repeat_last_practice.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisRepeat_last_practice.rgb)
        if thisRepeat_last_practice != None:
            for paramName in thisRepeat_last_practice:
                globals()[paramName] = thisRepeat_last_practice[paramName]
        
        for thisRepeat_last_practice in repeat_last_practice:
            currentLoop = repeat_last_practice
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
            # abbreviate parameter names if possible (e.g. rgb = thisRepeat_last_practice.rgb)
            if thisRepeat_last_practice != None:
                for paramName in thisRepeat_last_practice:
                    globals()[paramName] = thisRepeat_last_practice[paramName]
            
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
                 practice_counter = practice_counter + 1
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
                 #decider_randomisation = randint(0,2) # should anything be randomized
                 #color_or_position = randint(0,3) # color or position
                 square_to_change = randint(0,4) # for which square
            
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
            # Run 'Begin Routine' code from lsl_fixation_practice
            #Push screen fixation
            #screen_outlet.push_sample(screen_markers[0])
            #fixation_marker_count = 0
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
            response_key.keys = []
            response_key.rt = []
            _response_key_allKeys = []
            # Run 'Begin Routine' code from practice_button_presses
            ###### Randomization of location and color ######
            
            #Create list with all four squares that are used for comparison
            squares_test = [sqr_ident_1, sqr_ident_2, sqr_ident_3, sqr_ident_4]
            
            #Register possible buttons
            p_pressed = 0
            q_pressed = 0
            
            #Create variables for position and color
            #Set these to be identical to squares that were first displayed
            for i in range(len(squares_test)):
                 current_square = squares_test[i]
                 current_square.pos = (x_coord[i], y_coord[i])
                 current_square.color = colors[i]
            
            square_manipulation(squares_test, x_coord, y_coord, colors, decider_randomisation, color_or_position, square_to_change, 1)
            
            # Run 'Begin Routine' code from lsl_practice_accuracy
            #Push target presentation and accuracy
            #screen_outlet.push_sample(screen_markers[1])
            #target_present_marker_count = 0
            # keep track of which components have finished
            square_identComponents = [fix_cross2, sqr_ident_1, sqr_ident_2, sqr_ident_3, sqr_ident_4, response_key]
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
                
                # *response_key* updates
                waitOnFlip = False
                
                # if response_key is starting this frame...
                if response_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    response_key.frameNStart = frameN  # exact frame index
                    response_key.tStart = t  # local t and not account for scr refresh
                    response_key.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(response_key, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'response_key.started')
                    # update status
                    response_key.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(response_key.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(response_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if response_key.status == STARTED and not waitOnFlip:
                    theseKeys = response_key.getKeys(keyList=['p','q'], ignoreKeys=["escape"], waitRelease=True)
                    _response_key_allKeys.extend(theseKeys)
                    if len(_response_key_allKeys):
                        response_key.keys = [key.name for key in _response_key_allKeys]  # storing all keys
                        response_key.rt = [key.rt for key in _response_key_allKeys]
                        response_key.duration = [key.duration for key in _response_key_allKeys]
                # Run 'Each Frame' code from practice_button_presses
                #Create conditions that register button presses
                #Decide whether last trial is repeated or if we go to the next trial
                
                #Generate variable for time and grace period during which two buttons can be pressed
                
                if q_pressed == 0 and 'q' in response_key.keys: #only q is pressed
                    q_pressed = time.time()
                    #thisExp.addData('button_press', "response_q")
                elif p_pressed == 0 and 'p' in response_key.keys: #only p is pressed
                    p_pressed = time.time()
                    #thisExp.addData('button_press', "response_p")
                elif q_pressed != 0 and p_pressed != 0: #q and p are pressed within grace period: last trial is repeated
                    repeat_last_practice.finished = False
                    repeat_trial_practice = True
                    continueRoutine = False
                    #thisExp.addData('button_press', "response_p_and_q")
                #q and p are pressed outside of grace period: last trial is not repeated
                elif (q_pressed != 0 and (time.time() - q_pressed) > grace_period) or (p_pressed != 0 and (time.time() - p_pressed) > grace_period):
                    repeat_last_practice.finished = True
                    repeat_trial_practice = False
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
            # check responses
            if response_key.keys in ['', [], None]:  # No response was made
                response_key.keys = None
            repeat_last_practice.addData('response_key.keys',response_key.keys)
            if response_key.keys != None:  # we had a response
                repeat_last_practice.addData('response_key.rt', response_key.rt)
                repeat_last_practice.addData('response_key.duration', response_key.duration)
            # Run 'End Routine' code from practice_button_presses
            #Create variable for accuracy (correct button presses)
            #Squares are identical -> decider_randomisation == 0 -> "p" 
            #Squares are not identical (change of color/location) -> decider_randomisation != 0 -> "q" 
            
            if p_pressed != 0 and q_pressed != 0:  #p and q pressed: repeat trial
                pass
            elif p_pressed != 0 and decider_randomisation == 0: #squares identical
                response_accuracy_practice.append(1)
                thisExp.addData('response_accuracy', "Correct")
            elif q_pressed != 0 and decider_randomisation != 0: #squares not identical
                response_accuracy_practice.append(1)
                thisExp.addData('response_accuracy', "Correct")
            else: 
                response_accuracy_practice.append(0) #wrong button was pressed
                thisExp.addData('response_accuracy', "Incorrect")
            
            
                
            
            # Run 'End Routine' code from lsl_practice_accuracy
            #Push accuracy: correct and incorrect button presses
            #if (p_pressed != 0 or q_pressed != 0) and (time.time() - p_pressed) < grace_period:
            #    if target_present_marker_count == 0:
            #        if accuracy_button_press_trial == 1:
            #            behav_outlet.push_sample(behav_markers[0]) #correct response
            #            target_present_marker_count += 1
            #        else:
            #            behav_outlet.push_sample(behav_markers[1]) #incorrect response
            #            target_present_marker_count += 1
            
            #Push whether target stayed identical or not
            #if decider_randomisation == 0:
            #    screen_outlet.push_sample(condition_markers[3]) #identical squares
            #elif decider_randomisation == 1: #color_or_position -> 0 = color change, 1 = location change, 2 = both change
            #     screen_outlet.push_sample(condition_markers[color_or_position])
                
            
            # the Routine "square_ident" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "pause_practice" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('pause_practice.started', globalClock.getTime())
            # Run 'Begin Routine' code from lsl_iti_practice
            #sending first iti for practice
            #screen_outlet.push_sample(screen_markers[2])
            # keep track of which components have finished
            pause_practiceComponents = [pause_btw_trials, fix_cross3]
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
                
                # *pause_btw_trials* updates
                
                # if pause_btw_trials is starting this frame...
                if pause_btw_trials.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    pause_btw_trials.frameNStart = frameN  # exact frame index
                    pause_btw_trials.tStart = t  # local t and not account for scr refresh
                    pause_btw_trials.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pause_btw_trials, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pause_btw_trials.started')
                    # update status
                    pause_btw_trials.status = STARTED
                    pause_btw_trials.setAutoDraw(True)
                
                # if pause_btw_trials is active this frame...
                if pause_btw_trials.status == STARTED:
                    # update params
                    pass
                
                # if pause_btw_trials is stopping this frame...
                if pause_btw_trials.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > pause_btw_trials.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        pause_btw_trials.tStop = t  # not accounting for scr refresh
                        pause_btw_trials.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'pause_btw_trials.stopped')
                        # update status
                        pause_btw_trials.status = FINISHED
                        pause_btw_trials.setAutoDraw(False)
                
                # *fix_cross3* updates
                
                # if fix_cross3 is starting this frame...
                if fix_cross3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    fix_cross3.frameNStart = frameN  # exact frame index
                    fix_cross3.tStart = t  # local t and not account for scr refresh
                    fix_cross3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fix_cross3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fix_cross3.started')
                    # update status
                    fix_cross3.status = STARTED
                    fix_cross3.setAutoDraw(True)
                
                # if fix_cross3 is active this frame...
                if fix_cross3.status == STARTED:
                    # update params
                    pass
                
                # if fix_cross3 is stopping this frame...
                if fix_cross3.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > fix_cross3.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        fix_cross3.tStop = t  # not accounting for scr refresh
                        fix_cross3.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'fix_cross3.stopped')
                        # update status
                        fix_cross3.status = FINISHED
                        fix_cross3.setAutoDraw(False)
                
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
        # completed 5.0 repeats of 'repeat_last_practice'
        
    # completed 10.0 repeats of 'trials_practice'
    
    
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
            theseKeys = beginn_test.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
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
    
    # set up handler to look after randomisation of conditions etc
    trials_trial = data.TrialHandler(nReps=20.0, method='random', 
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
        
        # set up handler to look after randomisation of conditions etc
        repeat_last_trial = data.TrialHandler(nReps=8.0, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='repeat_last_trial')
        thisExp.addLoop(repeat_last_trial)  # add the loop to the experiment
        thisRepeat_last_trial = repeat_last_trial.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisRepeat_last_trial.rgb)
        if thisRepeat_last_trial != None:
            for paramName in thisRepeat_last_trial:
                globals()[paramName] = thisRepeat_last_trial[paramName]
        
        for thisRepeat_last_trial in repeat_last_trial:
            currentLoop = repeat_last_trial
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
            # abbreviate parameter names if possible (e.g. rgb = thisRepeat_last_trial.rgb)
            if thisRepeat_last_trial != None:
                for paramName in thisRepeat_last_trial:
                    globals()[paramName] = thisRepeat_last_trial[paramName]
            
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
                 trial_counter = trial_counter + 1
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
                 #decider_randomisation = randint(0,2) # should anything be randomized
                 #color_or_position = randint(0,3) # color or position
                 square_to_change = randint(0,4) # for which square
               
               
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
            response_key_trial.keys = []
            response_key_trial.rt = []
            _response_key_trial_allKeys = []
            # Run 'Begin Routine' code from trial_button_presses
            ###### Randomization of location and color ######
            
            #Register possible buttons
            p_pressed = 0
            q_pressed = 0
            
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
            sqr_ident_trialComponents = [fix_cross_4, sqr_ident_1_trial, sqr_ident_2_trial, sqr_ident_3_trial, sqr_ident_4_trial, response_key_trial]
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
                
                # *response_key_trial* updates
                waitOnFlip = False
                
                # if response_key_trial is starting this frame...
                if response_key_trial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    response_key_trial.frameNStart = frameN  # exact frame index
                    response_key_trial.tStart = t  # local t and not account for scr refresh
                    response_key_trial.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(response_key_trial, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'response_key_trial.started')
                    # update status
                    response_key_trial.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(response_key_trial.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(response_key_trial.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if response_key_trial.status == STARTED and not waitOnFlip:
                    theseKeys = response_key_trial.getKeys(keyList=['p','q'], ignoreKeys=["escape"], waitRelease=True)
                    _response_key_trial_allKeys.extend(theseKeys)
                    if len(_response_key_trial_allKeys):
                        response_key_trial.keys = [key.name for key in _response_key_trial_allKeys]  # storing all keys
                        response_key_trial.rt = [key.rt for key in _response_key_trial_allKeys]
                        response_key_trial.duration = [key.duration for key in _response_key_trial_allKeys]
                # Run 'Each Frame' code from trial_button_presses
                #Create conditions that register button presses
                #Decide whether last trial is repeated or if we go to the next trial
                
                if q_pressed == 0 and 'q' in response_key_trial.keys: #only q is pressed
                    q_pressed = time.time()
                    thisExp.addData('button_press', "response_q")
                elif p_pressed == 0 and 'p' in response_key_trial.keys: #only p is pressed
                    p_pressed = time.time()
                    thisExp.addData('button_press', "response_p")
                elif q_pressed != 0 and p_pressed != 0: #q and p are pressed within grace period: last trial is repeated
                    repeat_last_trial.finished = False
                    repeat_trial = True
                    continueRoutine = False
                    thisExp.addData('button_press', "response_p_and_q")
                #q and p are pressed outside of grace period: last trial is not repeated
                elif (q_pressed != 0 and (time.time() - q_pressed) > grace_period) or (p_pressed != 0 and (time.time() - p_pressed) > grace_period):
                    repeat_last_trial.finished = True
                    repeat_trial = False
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
            # check responses
            if response_key_trial.keys in ['', [], None]:  # No response was made
                response_key_trial.keys = None
            repeat_last_trial.addData('response_key_trial.keys',response_key_trial.keys)
            if response_key_trial.keys != None:  # we had a response
                repeat_last_trial.addData('response_key_trial.rt', response_key_trial.rt)
                repeat_last_trial.addData('response_key_trial.duration', response_key_trial.duration)
            # Run 'End Routine' code from trial_button_presses
            #Create variable for accuracy (correct button presses)
            #Squares are identical -> decider_randomisation == 0 -> "p" 
            #Squares are not identical (change of color/location) -> decider_randomisation != 0 -> "q" 
            
            if p_pressed != 0 and q_pressed != 0 and ((time.time() - p_pressed) < grace_period): #p and q pressed: repeat trial
                pass
            elif p_pressed != 0 and decider_randomisation == 0: #squares identical
                response_accuracy_trial.append(1)
                thisExp.addData('response_accuracy', "Correct")
            elif q_pressed != 0 and decider_randomisation != 0: #squares not identical
                response_accuracy_trial.append(1)
                thisExp.addData('response_accuracy', "Correct")
            else: 
                response_accuracy_trial.append(0) #wrong button was pressed
                thisExp.addData('response_accuracy', "Incorrect")
            
            
            print('-----------------')
            print('we are in trial no:', trial_counter)
            print('randomization:', decider_randomisation)
            #print('response:', response_randomisation)
            if repeat_trial == True: 
                print('!!!!REPEAT!!!!')
            if repeat_trial == False: 
                print('list of results:', response_accuracy_trial)
                
            
            # Run 'End Routine' code from lsl_trial_accuracy
            #Push accuracy: correct and incorrect button presses
            if (p_pressed != None or q_pressed != None) and (time.time() - p_pressed) < grace_period:
                if target_present_marker_count == 0:
                    if response_accuracy_trial == 1:
                        behav_outlet.push_sample(behav_markers[0]) #correct response
                        target_present_marker_count += 1
                    else:
                        behav_outlet.push_sample(behav_markers[1]) #incorrect response
                        target_present_marker_count += 1
            
            #Push whether target squares stayed identical or not
            if decider_randomisation == 0:
                screen_outlet.push_sample(condition_markers[3]) #identical squares
            elif decider_randomisation == 1: #color_or_position -> 0 = color change, 1 = location change, 2 = both change
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
            # keep track of which components have finished
            pause_trialComponents = [fix_cross4]
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
                
                # *fix_cross4* updates
                
                # if fix_cross4 is starting this frame...
                if fix_cross4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    fix_cross4.frameNStart = frameN  # exact frame index
                    fix_cross4.tStart = t  # local t and not account for scr refresh
                    fix_cross4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fix_cross4, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fix_cross4.started')
                    # update status
                    fix_cross4.status = STARTED
                    fix_cross4.setAutoDraw(True)
                
                # if fix_cross4 is active this frame...
                if fix_cross4.status == STARTED:
                    # update params
                    pass
                
                # if fix_cross4 is stopping this frame...
                if fix_cross4.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > fix_cross4.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        fix_cross4.tStop = t  # not accounting for scr refresh
                        fix_cross4.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'fix_cross4.stopped')
                        # update status
                        fix_cross4.status = FINISHED
                        fix_cross4.setAutoDraw(False)
                
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
        # completed 8.0 repeats of 'repeat_last_trial'
        
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 20.0 repeats of 'trials_trial'
    
    
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
