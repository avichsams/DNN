import numpy as np
import sys
import matplotlib.pyplot as plt
import os
from scipy.fft import fft, fftfreq


def find_index(Arr, elm) :
    T = [ i for i,e in enumerate(Arr) if e<=elm]
    return T[-1]



def get_prfx(cfg) :
    prfx = '_Int_Rx_'+str(np.round(- cfg.Rx /(np.pi)**2,10))
    return prfx

def Load_solution(cfg,folder='') :
    
    prfx = get_prfx(cfg)


    time   =  np.loadtxt(folder+'time'+prfx+'.dat')
    w_disp =  np.loadtxt(folder+'w_disp'+prfx+'.dat')
    w_vel  =  np.loadtxt(folder+'w_vel'+prfx+'.dat')


    return time, w_disp, w_vel


def compute_spectre(time_cut, w_cut, Dt, coef_spec_size = 2):
    # Number of sample points
    N = len(time_cut)
    # sample spacing
    T = Dt
    x = time_cut
    y = w_cut
    yf = fft(y)
    xf = fftfreq(N, T)[:N//coef_spec_size]
    y_amp =2.0/N * np.abs(yf[0:N//coef_spec_size])
    return xf, y_amp 





def plot_time_history(time,w_xp_disp,w_xp_vel, cfg, t_show_diag = 5  , t_end_show = None, label_diag= True):

    if t_end_show == None :
        endshwo = len(time)
    else :
        endshwo = find_index(time, t_end_show)

    nshow = find_index(time, t_show_diag)
    plt.suptitle('Complete temporal evolution of displacement and velocity', fontweight='bold', fontsize=18)
    plt.subplot(1, 2, 1)
    plt.plot(time, w_xp_disp,label = cfg.time_intgrt_mtd,color='k')
    plt.xlabel(r'$\tau$',fontsize=14)
    plt.ylabel(r'$w$',fontsize=14)
    if label_diag :
        plt.axvline(x=time[nshow],ls='--',lw = 2.5,color='r', label='Diagram window')
    else :
        plt.axvline(x=time[nshow],ls='--',lw = 2.5,color='r')
    
    if t_end_show != None :
        plt.axvline(x=time[endshwo], ls='--',lw = 2.5,color='r')
    plt.grid(True)
    plt.legend(loc ='upper left',fontsize=15)
    
    plt.subplot(1, 2, 2)
    plt.plot(time, w_xp_vel,color='k')
    plt.xlabel(r'$\tau$',fontsize=14)
    plt.ylabel(r'$\dot{w}$',fontsize=14)
    plt.axvline(x=time[nshow],ls='--',lw = 2.5,color='r', label='Diagram window')
    if t_end_show != None :
        plt.axvline(x=time[endshwo], ls='--',lw = 2.5,color='r')
    plt.grid(True)

def plot_time_history_cut(time,w_xp_disp,w_xp_vel, cfg, t_show_diag = None  , t_end_show = None):
    if t_end_show == None :
        endshwo = len(time)
    else :
        endshwo = find_index(time, t_end_show)
    if  t_show_diag==None :
         t_show_diag = time[0]
    nshow = find_index(time, t_show_diag)
    plt.suptitle(' Truncated time series', fontweight='bold', fontsize=18)
    plt.subplot(1, 2, 1)
    plt.plot(time[nshow:endshwo], w_xp_disp[nshow:endshwo], label = cfg.time_intgrt_mtd,color='k')
    plt.xlabel(r'$\tau$',fontsize=14)
    plt.ylabel(r'$w$',fontsize=14)
    plt.grid(True)
    plt.legend(loc ='upper left',fontsize=15)
    
    plt.subplot(1, 2, 2)
    plt.plot(time[nshow:endshwo], w_xp_vel[nshow:endshwo],color='k')
    plt.xlabel(r'$\tau$',fontsize=14)
    plt.ylabel(r'$\dot{w}$',fontsize=14)
    plt.grid(True)   

def plot_diag_fft(time,w_disp,w_vel, cfg, t_show_diag = None  , t_end_show = None, typ_diag='-', coef_spec_size = 2):

    if t_end_show == None :
        endshwo = len(time)
    else :
        endshwo = find_index(time, t_end_show)
    
    if  t_show_diag==None :
         t_show_diag = time[0]

    nshow = find_index(time, t_show_diag)
    plt.suptitle('Phase diagram and Spectre ', fontweight='bold', fontsize=18)
    plt.subplot(1, 2, 1)
    plt.plot(w_disp[nshow:endshwo],w_vel[nshow:endshwo], typ_diag,color='k')
    plt.xlabel(r'$w$',fontsize=14)
    plt.ylabel(r'$\dot{w}$',fontsize=14)
    plt.grid(True)

    # compute spectre
    X_spec, Y_spec = compute_spectre(time[nshow:endshwo], w_disp[nshow:endshwo], cfg.dt, coef_spec_size)
    plt.subplot(1, 2, 2)
    plt.plot(X_spec, Y_spec,color='k')
    plt.xlabel(r'$freq$',fontsize=14)
    plt.ylabel(r'$Amplitude$',fontsize=14)
    plt.grid(True)
    
