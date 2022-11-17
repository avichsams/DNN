# --------------------------------------
#   Configure Panel System
# --------------------------------------
import numpy as np

class CFG_PANEL:
    def __init__(self) :

        # -------- Non-dimensional parameters -------------     
        self.nu      = 0.3                # (float) : Poisson coefficient 
        self.lmbd    = 150                # (float) : Parameters of the flow velocity
        self.G       = 5.46               # (float) : Material type parameter
        self.Rx      = -3.*(np.pi)**2     # (float) : Parameters of the tension of the structure
        self.Mach    = 2.                 # (float) : Mach number
      
        # -------- Spatial discretization ------------- 
        self.Np            = 101  # (int)   : Number of physical nodes
        self.xprb          = 1/4  # (float) : Coordinate of x_p in ]0,1[
        self.Nn_ordr_drv   = 4    # (int)   : [2,4] Order of accuracy in the derivative for the non-linear term (recomended : 4)
        
        # -------- Time Integration ------------- 
        self.tau_max              = 50            # (float)          : Non-dimensional maximal time 
        self.time_intgrt_mtd      = "BDF"         # (str)            : [Newmark, BDF, RK45,..] Integration method of the PANEL equation (recomended : BDF)
        self.dt                   = 1e-2          # (float)          : Size of the time step (recomended : 1e-2)
        self.rtol,self.atol       = 1e-11,1e-6    # (float),(float)  : Relative and absolute tolerances

        # --------  Numerical parameters ------------- 
        self.prgrss_bar       = True       # (bool) : Display of a progress bar (in time-integration step) ( False not test)
        self.Nn_integrl_mtd   = "simps"    # (str)  : [tsimps,rapz]  method of integration of the non-linear term (recomended : simps)
        self.restart          = False      # (bool) : restart a calculation ( not yet : todo) 

