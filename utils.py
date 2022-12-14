from scipy.linalg import sqrtm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def moving_average(X, n) :
    ret = np.cumsum(X, axis=0)
    a = (ret[n:] - ret[:-n]) / n
    ma_part = (ret[n:] - ret[:-n]) / n
    return np.vstack([ret[:n]/n, ma_part])


def get_hi(X, sigmas):
    return np.maximum((X - sigmas), 0)

def get_lo(X, sigmas):
    return np.maximum(-(X - sigmas), 0)

def get_pos(X):
    return np.maximum(X, 0)

def get_neg(X):
    return np.maximum(-X, 0)


def get_corr(X1, X2):
    """
    Computes correlation as \Expect a^T b /(\Expect a^Ta \Expect b^Tb)^{1/2}
    where realizations of a and b are rows of X1 and X2, respectively.
    """
    assert X1.shape == X2.shape
    at_b = []
    at_a = []
    bt_b = []

    for t in range(X1.shape[0]):
        a = X1[t]
        b = X2[t]
        at_b.append(np.dot(a,b))
        at_a.append(np.dot(a,a))
        bt_b.append(np.dot(b,b))
    
    return np.mean(at_b) / np.sqrt(np.mean(at_a) * np.mean(bt_b))



def get_yearly_corrs(X1, X2):
    X1 = X1.flatten().copy()
    X2 = X2.flatten().copy()

    corrs = []
    for i in range(len(X1) - 252):
        corrs.append(np.corrcoef(X1[i:i+252], X2[i:i+252])[0,1])
    
    return corrs, np.mean(corrs), np.corrcoef(X1, X2)[0,1]

def get_monthly_corrs(X1, X2):
    X1 = X1.flatten().copy()
    X2 = X2.flatten().copy()

    corrs = []
    for i in range(len(X1) - 25):
        corrs.append(np.corrcoef(X1[i:i+25], X2[i:i+25])[0,1])
    
    return corrs, np.mean(corrs), np.corrcoef(X1, X2)[0,1]


def get_R(sqrt_Sigma_hats, r_tildes, sqrt_V_hats=None):
    rs = []
    if sqrt_V_hats is None:
        for t in range(len(sqrt_Sigma_hats)):
            sqrt_Sigma_hat = sqrt_Sigma_hats[t] 
            r_tilde = r_tildes[t].reshape(-1,1)
            r = (sqrt_Sigma_hat @ r_tilde).reshape(-1,)
            rs.append(r)
    else: # Double whitening
        for t in range(len(sqrt_Sigma_hats)):
            sqrt_Sigma_hat = sqrt_Sigma_hats[t] 
            sqrt_V_hat = sqrt_V_hats[t]
            r_tilde = r_tildes[t].reshape(-1,1)
            r_tilde_var = (sqrt_Sigma_hat @ r_tilde)
            r = (sqrt_V_hat @ r_tilde_var).reshape(-1,)
            rs.append(r)
    return np.array(rs)

def get_daily_covs(R):
    """
    param R: numpy array where rows are vector of asset returns for t=0,1,...

    returns: list of r_t*r_t' (matrix multiplication) for all days, i.e,\
        "daily covariances"
    """
    covs = []
    for t in range(R.shape[0]):
        r_t = R[t, :].reshape(-1,1)
        covs.append(r_t @ r_t.T)
    return np.array(covs)

def get_daily_sizes(R):
    sizes = []
    for t in range(R.shape[0]):
        r_t = R[t, :].reshape(-1,1)
        sizes.append(r_t.T @ r_t)
    return np.array(sizes)


def get_daily_vars(R):
    """
    param R: numpy array where rows are vector of asset returns for t=0,1,...

    returns: list of diag(r_t^2) for all days, i.e,\
        "daily variances"
    """
    variances = []
    for t in range(R.shape[0]):
        r_t = R[t, :].reshape(-1,)
        variances.append(np.diag(r_t**2))
    return np.array(variances)

def get_next_EWMA(EWMA, y_last, t, beta):
    """
    param EWMA: EWMA at time t-1
    param y_last: observation at time t-1
    param t: current time step
    param beta: EWMA exponential forgetting parameter

    returns: EWMA at time t
    """

    old_weight = (beta-beta**t)/(1-beta**t)
    new_weight = (1-beta) / (1-beta**t)

    return old_weight*EWMA + new_weight*y_last

def get_EWMAs(y, beta):
    """
    y: array with measurements for times t=0,1,...,len(t)-1
    beta: EWMA exponential forgetting parameter

    returns: list of EWMAs for times t=1,2,...,len(t)

    Note: We define EWMA_t as a function of the 
    observations up to time t-1. This means that
    y = [y_0,y_1,...,y_T] (for some T), while
    EWMA = [EWMA_1, EWMA_2, ..., EWMA_{T+1}]
    """

    EWMA_t = 0
    EWMAs = []
    for t in range(1,y.shape[0]+1): # First EWMA is for t=2 
        y_last = y[t-1] # Note zero-indexing
        EWMA_t = get_next_EWMA(EWMA_t, y_last, t, beta)
        EWMAs.append(EWMA_t)
    return np.array(EWMAs)


def get_r_tildes(sqrt_Sigma_hats, R):
    """
    param sqrt_Sigma_hats: array of sqrt(Sigma_hat_t) for t=1,2,...
    param R: array of returns for t=1,2,...

    returns numpy array with tilde_hats as rows
    """
    r_tildes = []
    for t in range(sqrt_Sigma_hats.shape[0]):
        sqrt_Sigma_hat = sqrt_Sigma_hats[t] 
        r_t = R[t].reshape(-1,1)
        r_tilde = np.linalg.solve(sqrt_Sigma_hat, r_t).reshape(-1,)
        r_tildes.append(r_tilde)
    return np.array(r_tildes)

def get_standerdized_data(R, train_len):
    """
    Non-dynamic standardization
    """
    Sigma_hat = np.cov(R[:train_len], rowvar=False)
    sqrt_Sigma_hat = np.linalg.cholesky(Sigma_hat) 

    r_tildes = []
    for t in range(len(R)):
        r_t = R[t].reshape(-1,1)
        r_tilde = np.linalg.solve(sqrt_Sigma_hat, r_t).reshape(-1,)
        r_tildes.append(r_tilde)
    
    return np.array(r_tildes), sqrt_Sigma_hat

def get_whitened_data(R, whitener='Var', beta=0.9, EWMA_reg=0):
    if whitener=='double':
        assert len(beta) == 2
        beta1 = beta[0]
        beta2 = beta[1]
        start1 = 100; start2 = 100
    else:
        start = 100 
    if whitener=='Var':
        variances = get_daily_vars(R)
        EWMAs = get_EWMAs(variances, beta=beta)
    elif whitener=='Cov':
        covs = get_daily_covs(R)
        EWMAs = get_EWMAs(covs, beta=beta)
    elif whitener=='double':
        variances = get_daily_vars(R)
        V_hats = get_EWMAs(variances, beta=beta1)
        V_hats = V_hats[start1:-1] # Start after EWMA initialization period is over

        for t, V_hat in enumerate(V_hats):
            V_hats[t] = sqrtm(V_hat) 
        sqrt_V_hats = V_hats

        R_new = R[1+start1:] # The first return is used in the first V_hat (discard)
        r_tildes_var = get_r_tildes(sqrt_V_hats, R_new)

        covs = get_daily_covs(r_tildes_var)
        Sigma_hats = get_EWMAs(covs, beta=beta2)
        Sigma_hats = Sigma_hats[start2:]  # TODO: Should be correct to not have the "1 shift" for r_tilde_tilde
        sqrt_V_hats = sqrt_V_hats[start2:] # Also need to discard (the corrsponding) start2 first sqrt_V_hats

        for t, Sigma_hat in enumerate(Sigma_hats):
            Sigma_hats[t] = sqrtm(Sigma_hat) 
        sqrt_Sigma_hats = Sigma_hats

        r_tildes_var = r_tildes_var[start2:] # TODO: Should be correct to not have the "1 shift" for r_tilde_tilde
        r_tildes = get_r_tildes(sqrt_Sigma_hats, r_tildes_var)

        R_new = R_new[start2:]

        skipped_R = start1+start2+1 # 1 is discarded to not have "future look-ahead"; also discard start1+start2
        return (sqrt_V_hats, sqrt_Sigma_hats), (r_tildes_var, r_tildes), R_new, skipped_R  

    elif whitener=='tripple':
        assert len(beta) == 3
        beta1 = beta[0]
        beta2 = beta[1]
        beta3 = beta[2]
        start1 = 100; start2 = 100; start3 = 100

        # Size whitening
        sizes = get_daily_sizes(R) 
        size_hats = get_EWMAs(sizes, beta=beta1)
        size_hats = size_hats[start1:-1] # Start after EWMA initialization period is over
        sqrt_size_hats = np.sqrt(size_hats).reshape(-1,1)
        R_new = R[1+start1:] # The first return is used in the first V_hat (discard)
        r_tildes_size = R_new / sqrt_size_hats
        

        # Diagonal whitening
        variances = get_daily_vars(r_tildes_size)
        V_hats = get_EWMAs(variances, beta=beta2)
        V_hats = V_hats[start2:] # Start after EWMA initialization period is over
        sqrt_size_hats = sqrt_size_hats[start2:] # Also need to discard (the corrsponding) start2 sqrt_size_hats sqrt_V_hats
        for t, V_hat in enumerate(V_hats):
            V_hats[t] = np.linalg.cholesky(V_hat) 
        sqrt_V_hats = V_hats # Rename for cleaner notation

        r_tildes_size = r_tildes_size[start2:] # Should be correct to not have the "1 shift" for r_tilde_tilde
        R_new = R_new[start2:]
        r_tildes_var = get_r_tildes(sqrt_V_hats, r_tildes_size)


        # Full whitening
        covs = get_daily_covs(r_tildes_var)
        Sigma_hats = get_EWMAs(covs, beta=beta3)
        Sigma_hats = Sigma_hats[start3:]  # TODO: Should be correct to not have the "1 shift" for r_tilde_tilde
        sqrt_V_hats = sqrt_V_hats[start3:] # Also need to discard (the corresponding) start3 first sqrt_V_hats
        sqrt_size_hats = sqrt_size_hats[start3:] # Also need to discard (the corresponding) start3 sqrt_size_hats sqrt_V_hats
        for t, Sigma_hat in enumerate(Sigma_hats):
            Sigma_hats[t] = np.linalg.cholesky(Sigma_hat)  
        sqrt_Sigma_hats = Sigma_hats # Rename for cleaner notation
        r_tildes_size = r_tildes_size[start3:]
        r_tildes_var = r_tildes_var[start3:] # TODO: Should be correct to not have the "1 shift" for r_tilde_tilde
        R_new = R_new[start3:]
        r_tildes_full = get_r_tildes(sqrt_Sigma_hats, r_tildes_var)

        

        skipped_R = start1+start2+start3+1 # 1 is discarded to not have "future look-ahead"; also discard start1+start2
        return (sqrt_size_hats, sqrt_V_hats, sqrt_Sigma_hats), (r_tildes_size, r_tildes_var, r_tildes_full), R_new, skipped_R  


    EWMAs = EWMAs[start:-1] # Start after EWMA initialization period is over (last EWMA includes last return; discard)


    # Compute sqrt of EWMAs so that this does not have to be done every time!
    for t, EWMA in enumerate(EWMAs):
        EWMA_temp = EWMA + (EWMA_reg)*np.trace(EWMA)*np.eye(EWMA.shape[0]) 
        
        EWMAs[t] = np.linalg.cholesky(EWMA_temp) 
        # EWMAs[t] = sqrtm(EWMA_temp) 

        # EWMAs[t] = sqrtm(EWMA) 

    R_new = R[1+start:] # The first return is used in the first EWMA (discard)
    r_tildes = get_r_tildes(EWMAs, R_new)

    skipped_R = start+1
    return EWMAs, r_tildes, R_new, skipped_R

def plot_heatmap(metrics, x_lab, y_lab, metric="Correlation"):
    sns.set(font_scale=1.5)

    data = metrics[metric]    

    data_temp = pd.DataFrame({x_lab: data[:,0], y_lab: data[:,1].astype(int), 'Validation correlation': data[:,2]})
    # return data_temp
    data_pivoted = data_temp.pivot(x_lab, y_lab, "Validation correlation")
    ax = sns.heatmap(data_pivoted,  cbar_kws={'label': 'Validation correlation'})

    best_comb = np.argmax(data[:,2])
    
    best_x = data[best_comb][0]
    best_y = data[best_comb][1]
    print("Best x: ", best_y)
    print("Best y: ", best_x)

    # star_coords = (np.where(data[:,2] == best_x)[0].astype(int), np.where(data[:,1] == best_y)[0].astype(int))

    data_temp = data[data[:,0] == best_x]
    data_temp = data_temp[data_temp[:,1] == best_y].flatten()
    star_coords = (data_temp[0], data_temp[1])

    unique_x = list(set(data[:,1]))
    unique_y = list(set(data[:,0]))
    unique_x.sort()
    unique_y.sort()

    x = np.where(unique_x == star_coords[1])[0].astype(int)
    y = np.where(unique_y == star_coords[0])[0].astype(int)

    ax.scatter(x+0.5, y+0.5, marker='*', s=500, color='black');
    print(f"Maximum validation corr: {data[best_comb][2]:.2%}")
    print(f"beta: {data[best_comb][3]:.2}")
    print(f"excess signs: {data[best_comb][4]:.2%}")