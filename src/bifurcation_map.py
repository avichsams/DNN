import numpy as np
from tqdm import tqdm
from scipy.signal import  find_peaks

def find_index(Arr, elm) :
    T = [ i for i,e in enumerate(Arr) if e<=elm]
    return T[-1]

def get_LCO_amp(X):
    """
    """
    
    # ---- get index peaks
    ind_peaks, _   = find_peaks(X)
    ind_valleys, _ = find_peaks(-X)
    
    # take the   amplitude of the sequence 
    X_lco_max = X[ind_peaks]
    X_lco_min = X[ind_valleys]
    
    # arrondi les elements de X_lco_max à la seconde decimale puis elimine les doublons
    X_lco_max_red = np.unique(np.round(X_lco_max[:],2))
    X_lco_min_red = np.unique(np.round(X_lco_min[:],2))

    
    return X_lco_max_red, X_lco_min_red



def Compute_Bifurcation_Map(myfunc, cfg, name_prm, prm_min, prm_max,  Num_prm, time_retained,\
                            save_map =  True, folder  = '' , continuous = False) :
    
    Param = np.linspace(prm_min, prm_max, Num_prm)
    X_maps = []
    Y_maps = []

    # Continue a calculation that has been interrupted
    if continuous :
        last_prm = np.loadtxt(folder+'last_prm.dat')
        last_prm+=0 # corrige a bug  array with one elmnt
        X_maps = np.loadtxt(folder+'X_maps.dat')
        Y_maps = np.loadtxt(folder+'Y_maps.dat')
        Param_cont  = [ p for p in Param if p > last_prm]
        Param = Param_cont
        
    for indx,prm in enumerate(tqdm(Param)):
        # ------ update cfg.name_prm
        code = 'cfg.'+name_prm+'='+str(prm)
        exec(code)
        time, sequence = myfunc(cfg)
        
        # ---- Get the retained sequence
        index_retained = find_index(time,time_retained)
        X_sequence     = sequence[index_retained:]
        
        # ---- Get the peaks for the retained sequence
        Peaks, Valleys = get_LCO_amp(X_sequence)
        
        # Add points in Maps
        for y in Peaks :
            X_maps = np.append(X_maps, prm)
            Y_maps = np.append(Y_maps, y)
        for y in Valleys :
            X_maps = np.append(X_maps, prm)
            Y_maps = np.append(Y_maps, y)
            
        # save_data
        if save_map  :
            np.savetxt(folder+'X_maps.dat' ,X_maps)
            np.savetxt(folder+'Y_maps.dat' ,Y_maps)
            last_prm =[prm]
            np.savetxt(folder+'last_prm.dat' ,last_prm)
        
    return X_maps,Y_maps