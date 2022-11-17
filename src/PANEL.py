import numpy as np
import os

from  PANEL_Solver import PANEL_Solver

def get_prfx(cfg) :
    prfx = '_Int_Rx_'+str(np.round(- cfg.Rx /(np.pi)**2,10))
    return prfx

def launch_solver(cfg,folder='',overwrite = False) :
    '''
    Launch the PANEL solver ( or Load the solution if it exists in folder):
        - Input : 
                * cfg       : (Objet)  the parameters of the solver
                * folder    : (Str) Repertory for save  Solutions ( time, displacement, velocity)
                * overwrite : (Boolean) If it is true, it overwrites the previously saved solution and starts a new calculation.
        - Output :
                * time         : (array) dimension less time
                * w_disp       : (array) displacement of the panel 
                * tiw_velme    : (array) velocity of the panel 
    '''

    #File_folder
    prfx = get_prfx(cfg)
    file = folder+'time'+prfx+'.dat'
    Load = False
    if os.path.exists(file) and not overwrite :
        Load = True
    
    if Load : 
        time   =  np.loadtxt(folder+'time'+prfx+'.dat')
        w_disp =  np.loadtxt(folder+'w_disp'+prfx+'.dat')
        w_vel  =  np.loadtxt(folder+'w_vel'+prfx+'.dat')

    else :
        time, w_disp, w_vel= PANEL_Solver(cfg)
        np.savetxt(folder+'time'+prfx+'.dat',time)
        np.savetxt(folder+'w_disp'+prfx+'.dat',w_disp)
        np.savetxt(folder+'w_vel'+prfx+'.dat',w_vel)
    return time, w_disp, w_vel
