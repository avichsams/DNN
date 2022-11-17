
import numpy as np
import scipy
import  scipy.linalg
from tqdm import tqdm
from Spatial_Discretization import compute_Non_linear

# --------------------------------------------------------------------
#                Utils 
# --------------------------------------------------------------------
def find_index(Arr, elm) :
    T = [ i for i,e in enumerate(Arr) if e<=elm]
    return T[-1]

def savefiles_and_print(cfg,w_disp,w_vel,time,num_itr,Ind_xprb):
    print('Iteration   Time  X_prb  Disp  Vel')


# --------------------------------------------------------------------
#
#                  NEWMARK METHODE
#
# ---------------------------------------------------------------------
def Newmark_Methode(cfg, w_disp_n,  w_vel_n, M_mat,  C_mat, K_mat, D2, prgrss_bar = True, Ind_xprb = None ) : 
    '''
    TIME-INTEGRATION: newmark 
    '''


    #------------------------------------------
    #  Compute the matrix S                      
    #------------------------------------------
    # the matrix S
    Nn_term =  compute_Non_linear(cfg, w_disp_n)
    S_mat   =  - cfg.G*Nn_term*D2



    # ----- Compute the initiale  acceleration
    M_inv   = M_mat #scipy.linalg.inv(M_mat)
    K_p_S   = K_mat + S_mat
    C_KpS   = C_mat.dot(w_vel_n) + K_p_S.dot(w_disp_n)
    w_acc_n = -M_inv.dot(C_KpS)

    ## store the solution in the array
    if Ind_xprb != None :
        Sol_disp =  np.zeros(cfg.ntime) ;   Sol_disp[0] = w_disp_n[Ind_xprb] 
        Sol_vel  =  np.zeros(cfg.ntime) ;   Sol_vel[0]  = w_vel_n[Ind_xprb] 
    
    else :
        Sol_disp = np.zeros((cfg.ntime,cfg.Nc)); Sol_disp[0,:] = w_disp_n
        Sol_vel  = np.zeros((cfg.ntime,cfg.Nc)); Sol_vel[0,:]  = w_vel_n
        
    time  = np.zeros(cfg.ntime)    
    tm = 0

    # Integration Loop
    for ind_t in tqdm(range(1,cfg.ntime), disable= not prgrss_bar) :
        tm  += cfg.dt 
        time[ind_t] = tm
        

        # ---- compute the displacements at time n+1
        #build the A and B matrix 
        A_mat       = 4./(cfg.dt**2)*M_mat + 2./(cfg.dt) * C_mat + S_mat + K_mat
        A_mat_inv   = scipy.linalg.inv(A_mat)
        B_mat       = M_mat.dot(4./(cfg.dt**2)*w_disp_n + 4./(cfg.dt)*w_vel_n + w_acc_n)\
                    + C_mat.dot(2./(cfg.dt)*w_disp_n + w_vel_n)
        # w_{n+1}
        new_w_disp_n = A_mat_inv.dot(B_mat)
        # dot_w_{n+1}
        new_w_vel_n  = 2./cfg.dt *(new_w_disp_n - w_disp_n) - w_vel_n
        # ddot_x_{n+1}
        new_w_acc_n  = 4./(cfg.dt**2) * (new_w_disp_n - w_disp_n) - 4./cfg.dt * w_vel_n - w_acc_n

        ## update the solution
        w_disp_n  = new_w_disp_n ;  w_vel_n  = new_w_vel_n ;  w_acc_n  = new_w_acc_n

   
        ## store the solution in the array
        if Ind_xprb !=None :
            Sol_disp[ind_t]  = w_disp_n[Ind_xprb]; Sol_vel[ind_t]   = w_vel_n[Ind_xprb]
        else :
            Sol_disp[ind_t,:] = w_disp_n;  Sol_vel[ind_t,:]  = w_vel_n    

        if ind_t != cfg.ntime-1 :
            # ---- Build the matrix S ------
            # Compute the matrix S (nonlinear term) 
            Nn_term    =  compute_Non_linear(cfg, w_disp_n)
            S_mat      =  - cfg.G*Nn_term*D2


    return time, Sol_disp, Sol_vel


# --------------------------------------------------------------------
#
#           Backward differentiation formula (BDF)
#
# ---------------------------------------------------------------------

def FODE(time, U, cfg, M_mat,  C_mat, K_mat, D2, pbar= None, state= None):

    """  - U est un vecteur defini  : 
           U = [w , v] : dim 1 x (2*Nc)
        
        
        return : dU = [v , v_dot] : dim 1 x (2*Nc)
    """

    dU           = np.zeros(2*cfg.Nc) 
    dU[:cfg.Nc]  = U[cfg.Nc:]
    
    w_disp_n = U[:cfg.Nc]; w_vel_n = U[cfg.Nc:]
    
    #------------------------------------------
    #  Compute the matrix S                      
    #------------------------------------------
    # the matrix S
    Nn_term =  compute_Non_linear(cfg, w_disp_n)
    S_mat   =  - cfg.G*Nn_term*D2
    
    # inverse M = Id
    M_inv = M_mat #scipy.linalg.inv(M_mat)
    # terme 2
    K_p_S = K_mat + S_mat
    term2 =  C_mat.dot(w_vel_n) + K_p_S.dot(w_disp_n)
    
    dU[cfg.Nc:] = - M_inv.dot(term2)

    
    # progression bar :

    if cfg.prgrss_bar :  
    
        last_t, dt = state
        n = int((time - last_t)/dt)
        pbar.update(n)
        
        state[0] = last_t + dt * n
    else :
        last_t, dt = state
        n = int((time - last_t)/dt)
        if n%5000== 0 :
            print('iteration :',n)
    
    return dU

def scipy_Methode(cfg, w_disp_n,  w_vel_n, M_mat,  C_mat, K_mat, D2, prgrss_bar = True, Ind_xprb = None ) :

    
    
    # generalized displacement vector
    U = np.zeros(2*cfg.Nc)
    U [:cfg.Nc] = w_disp_n
    U [cfg.Nc:] = w_vel_n

    # Solve the panel systeme
    T0 = 0
    T1 = cfg.tau_max
    t_span = (T0, T1)
    t_eval = np.linspace(T0,T1,cfg.ntime)

    if cfg.prgrss_bar :
        with tqdm(total=cfg.ntime, unit="‰") as pbar:
            Sol = scipy.integrate.solve_ivp(
                fun=FODE,t_span =t_span,y0=U,t_eval=t_eval,
                method=cfg.time_intgrt_mtd,
                rtol = cfg.rtol,atol = cfg.atol,
                args = (cfg, M_mat,  C_mat, K_mat, D2, pbar,[T0, (T1-T0)/cfg.ntime]),
        )   
    else : 
        Sol = scipy.integrate.solve_ivp(
                fun=FODE,t_span =t_span,y0=U,t_eval=t_eval,
                method=cfg.time_intgrt_mtd,
                rtol = cfg.rtol,atol = cfg.atol,
                args = (cfg, M_mat,  C_mat, K_mat, D2,None,[T0, (T1-T0)/cfg.ntime]),
        )   


    Y     = Sol.y
    Y     = Y.T
    time  = Sol.t
    
    
    ## store the solution in the array
    if Ind_xprb != None :
        Sol_disp = Y[:,Ind_xprb] 
        Sol_vel  = Y[:,cfg.Nc+Ind_xprb] 
    
    else :
        Sol_disp = Y[:,:cfg.Nc]
        Sol_vel  = Y[:,cfg.Nc:]
    
    return time, Sol_disp, Sol_vel

        
       

