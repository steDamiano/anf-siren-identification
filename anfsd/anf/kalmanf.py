import numpy as np 

def kalmanf (y: np.array, fs: int, rho: float, q: float, r: float, num_channels: int = 2):
    '''
    Kalman-Based Adaptive Notch Filter (KalmANF)
    

    Input arguments:
        y               - input data vector (Lx1)
        fs              - sampling frequency (Hz)
        rho             - pole raduis (value between 0 and 1, with values closer to 1 creating a narrower notch)
        q               - Covariance of process noise
        r               - Covariance of measurement noise
        num_channels    - pow_ratio is estimated if num_channels = 2
        
    Returns:
        f_kal       - Estimated frequency over time (Lx1)
        a_kal       - Estimated filter coefficient over time (Lx1)
        e_kal       - Output from notch filter (Lx1)
        pow_ratio   - Power ratio (Lx1)
        
    '''  


    s_kal = np.zeros(len(y)) # intermediate variable of ANF
    e_kal = np.zeros(len(y)) # output of ANF
    a_kal = 2*np.ones(len(y)) # coefficient to be updated (corresponds to f = 0)
    f_kal = np.zeros(len(y)) # frequency to be tracked
    pow_ratio = np.ones(len(y)) # power ratio of tracked sinusoid to input signal
    
    p_cov = fs*q # initialise covariance of the error
    K = np.zeros(len(y)) # Kalman gain

    # Power estimation averaging parameter
    tau = 0.025 # time constant 
    lambd = np.exp(-1/(tau*fs)) # forgetting factor
    
    # Initialize power estimates
    pow_y = 1e-6 
    pow_sin = 1e-6
    pow_e = 1e-6
    
    for n in np.arange(2,len(y),1): # start loop from two samples ahead because we need samples at m-1 and m-2 
        
        # Prediction
        # a(n|n-1) is simply a(n-1) since the state transition matrix is an identiy
        p_cov = p_cov + q; # update covariance of prediction error
        
        
        # Estimation
        s_kal[n] = y[n] + rho*s_kal[n-1]*a_kal[n-1] - (rho**2)*s_kal[n-2] # Define s_kal from data
        K[n] = (s_kal[n-1])/( (s_kal[n-1]**2) + r/p_cov )
        e_kal[n] = s_kal[n] - s_kal[n-1]*a_kal[n-1] + s_kal[n-2] 
        
        a_kal[n] = a_kal[n-1] + K[n]*e_kal[n]
        
        # Update covariance of error
        p_cov = (1 - K[n]*s_kal[n-1])*p_cov
        
        # Compute frequency
        if (a_kal[n] > 2):
            a_kal[n] = 2 # reset coefficient if a is out of range to compute acos
        elif  (a_kal[n] < -2):
            a_kal[n] = -2  # reset coefficient if a is out of range to compute acos
        
        omega_hat_kal = np.arccos(a_kal[n]/2)
        f_kal[n] = (omega_hat_kal*fs)/(2*np.pi) # estimated frequency


       ###### 1 CHANNEL, MODULATED ##########
        if num_channels == 1:
            pow_y = 0.99*pow_y + 0.01*y[n]**2
            pow_sin = 0.99*pow_sin + 0.01*(y[n] - e_kal[n])**2
                
            if pow_sin/(pow_y + 1e-8) < 0.1:
                f_kal[n] = 0 
            if pow_sin/(pow_y + 1e-8) < 0.1:
                f_kal[n] = 0

        ###### 2 CHANNELS ##########
        elif num_channels == 2:
            pow_y = lambd*pow_y + (1-lambd)*y[n]**2
            pow_e = lambd * pow_e + (1 - lambd) * e_kal[n] ** 2
            
            pow_sin = pow_y - pow_e
            pow_ratio[n] = pow_sin/(pow_y + 1e-8)
    
    return f_kal, a_kal, e_kal, pow_ratio

