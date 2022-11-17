
# ----------------------------------------------
#   Importations
# ----------------------------------------------

import numpy as np
import sys
import matplotlib.pyplot as plt
import os

## import the Solver
sys.path.append('src')
from PANEL import launch_solver

## import Input parameters
from Config import CFG_PANEL

## define folder for save solution 
folder = "Data/Data_Ref/"


if not os.path.exists(folder):
    os.makedirs(folder)



# ------------------------------------------
#   Launch the Solver
# ------------------------------------------
## Input parameters
cfg                 = CFG_PANEL()   


time,w_disp,w_vel = launch_solver(cfg,folder)



