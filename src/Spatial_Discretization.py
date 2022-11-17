
import numpy as np
from scipy.integrate import simps

def get_D4(cfg) :
    delta = cfg.dx **4 
    c1 = 1./delta ; c2 = -4./delta; c3 = 6./delta; c4 = -4./delta; c5 = 1./delta

    D4 = c1*np.eye(cfg.Nc, k =-2 ) +  c2*np.eye(cfg.Nc, k =-1) + c3*np.eye(cfg.Nc, k =0 )  \
        + c4*np.eye(cfg.Nc, k =1)  + c5*np.eye(cfg.Nc, k =2) 
    
    return D4

def get_D2(cfg) :

    delta = (cfg.dx)**2
    c1 = 1./delta; c2 = -2./delta; c3 = 1./delta

    D2 = c1*np.eye(cfg.Nc,k=-1) + c2*np.eye(cfg.Nc,k=0) + c3*np.eye(cfg.Nc,k=1)
    return D2 

def get_D1(cfg) :
    delta = (cfg.dx)
    c1    = -(1./2)/delta; c2 = (1./2)/delta

    D1 = c1*np.eye(cfg.Nc,k=-1) + c2*np.eye(cfg.Nc,k=1)
    return D1




def compute_Non_linear(cfg, w_disp) :


    # convert to the computational domain :
    DispNp = np.zeros(cfg.Np)
    DispNp[1:cfg.Np-1] = w_disp
    #---- cumpute dw/dx 
    DispNp_dx = 0*w_disp

    if cfg.Nn_ordr_drv == 2: 
        # Internal node : Central finite difference : order 2
        for i in range(1,cfg.Nc-1):
            DispNp_dx[i] = (DispNp[i+1] - DispNp[i-1])/ ( 2*cfg.dx)
        
        # Border Left :  Forward finite difference : order 2
        i = 0
        DispNp_dx[i] = (-DispNp[i+2] + 4.* DispNp[i+1] - 3.*DispNp[i])/(2*cfg.dx)

        # Border Right :  Backward finite difference : order 2
        i = cfg.Nc-1
        DispNp_dx[i] = (DispNp[i-2] - 4.* DispNp[i-1] + 3.*DispNp[i])/(2*cfg.dx)


    if cfg.Nn_ordr_drv == 4 : 
        # Internal node : Central finite difference : order 4
        for i in range(2,cfg.Nc-2):
            DispNp_dx[i] = (-DispNp[i+2] + 8*DispNp[i+1]  - 8*DispNp[i-1] + DispNp[i-2])/ ( 12*cfg.dx)
        
        # Border Left :  Forward finite difference : order 4
        i=1 
        DispNp_dx[i] = (-(1./4.)*DispNp[i+4] +4./3* DispNp[i+3] -3* DispNp[i+2] + 4.* DispNp[i+1] - (25./12)*DispNp[i])/(cfg.dx)
           
        i= 0 
        DispNp_dx[i] = (-(1./4.)*DispNp[i+4] +4./3* DispNp[i+3] -3* DispNp[i+2] + 4.* DispNp[i+1] - (25./12)*DispNp[i])/(cfg.dx)

        # Border Right : Forward finite difference : order 4
        i = cfg.Nc-2
        DispNp_dx[i] = -(-(1./4.)*DispNp[i-4] +4./3* DispNp[i-3] -3* DispNp[i-2] + 4.* DispNp[i-1] - (25./12)*DispNp[i])/(cfg.dx)

        i= cfg.Nc-1 
        DispNp_dx[i] = -(-(1./4.)*DispNp[i-4] +4./3* DispNp[i-3] -3* DispNp[i-2] + 4.* DispNp[i-1] - (25./12)*DispNp[i])/(cfg.dx)

    pos   = np.linspace(0,1.,cfg.Np)
    psi   = pos[1:cfg.Np-1]
    func  = (DispNp_dx)**2
    # -- compute the integrale in eq. (**)
    if cfg.Nn_integrl_mtd == "trapz" :
        Nn    = np.trapz(y=func,x=psi)
    elif cfg.Nn_integrl_mtd == "simps" : 
        Nn    = simps(y=func,x=psi)
    return Nn
    