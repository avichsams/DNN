import numpy as np
from Spatial_Discretization import get_D1, get_D2, get_D4
from Time_Integration import Newmark_Methode, scipy_Methode, find_index



def PANEL_Solver(cfg) :


    #------------------------------------------
    #  cumpute other paratemers                     
    #------------------------------------------
    cfg.ntime       = int(cfg.tau_max/cfg.dt)     #  Number of time steps
    cfg.Nt          = cfg.Np+2                    #  Number of total nodes   (physical + ghost)
    cfg.Nc          = cfg.Np-2                    #  Nomber of internal nodes
    cfg.dx          =  1 / (cfg.Np-1)             # Size of the spatial discretization
    cfg.mu          = (0.1**2)*cfg.Mach           # Mass ratio 
    cfg.Nn_ordr_drv = int(cfg.Nn_ordr_drv)


    #------------------------------------------
    #  Build the Matrix M, C, K      (Finite di:erence approach )           
    #------------------------------------------
    # the matrix M
    M_mat = np.eye(cfg.Nc) 

    # the matrix C
    c_b   = np.sqrt(cfg.mu*cfg.lmbd/cfg.Mach)
    C_mat = c_b*np.eye(cfg.Nc)

    # the matrix K
    D1 = get_D1(cfg); D2 = get_D2(cfg); D4 = get_D4(cfg)
    K_mat = D4 - cfg.Rx * D2 + cfg.lmbd * D1

    # boundary condition ( ghost node)
    K_mat[0,0  ]  = K_mat[0,0]   -  K_mat[0,2]   
    K_mat[-1, -1] = K_mat[-1,-1] -  K_mat[2,0]   


    # ----- initial conditions -----  
    x_panel  = np.linspace(0,1,cfg.Np)
    disp_0   = 0.1*np.sin(np.pi*x_panel)
    # convert to the computational domain  
    w_disp_n = disp_0[1:cfg.Np-1]
    w_vel_n  = np.zeros(cfg.Nc)

    # Coordinate of a point of an X_p for stcoker solutions  (displacment + velocity)
    # if xprb = None, return the displacement ( and velocity ) of all the points of the panel
    # xprb in ]O,1[ or xprb = None
    # --- Coordinate of x_p
    if not hasattr(cfg, 'xprb'):  # if not define x_p in cfg
        cfg.xprb = None # if not define x_p in cfg 

    xprb = cfg.xprb
    if xprb != None :
        Ind_xprb = find_index(x_panel,xprb)
    else :
        Ind_xprb = None
    #------------------------------------------
    #  TIME-INTEGRATION                       
    #------------------------------------------
    if cfg.time_intgrt_mtd  == "Newmark":
        prgrss_bar = cfg.prgrss_bar
        time, Sol_disp, Sol_vel = Newmark_Methode(cfg, w_disp_n,  w_vel_n, \
            M_mat,  C_mat,  K_mat, D2, prgrss_bar,Ind_xprb)
    else : 
        prgrss_bar = cfg.prgrss_bar
        time, Sol_disp, Sol_vel = scipy_Methode(cfg, w_disp_n,  w_vel_n, \
            M_mat,  C_mat,  K_mat, D2, prgrss_bar,Ind_xprb)



    return time, Sol_disp, Sol_vel














   