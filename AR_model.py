import numpy as np

def predict_periodically(As, Xs, Rs):
    r_hat = None
    R = None

    for p in As.keys():
        A_p = As[p]
        X_p = Xs[p]
        R_p = Rs[p]
        r_hat_p = predict(A_p, X_p)

        if r_hat is None:
            r_hat = r_hat_p
            R = R_p
        else:
            r_hat = np.vstack([r_hat, r_hat_p])
            R = np.vstack([R, R_p])
    return r_hat, R

def predict(A, X_val):
    return (A@X_val).T



def get_As(X_train, y_train, X_val, y_val, R_val, alpha, weighted, update_period):

    As = {}
    X_vals = {}
    R_vals = {}
    
    n_retrains = int(np.ceil(R_val.shape[0] / update_period))

    for p in range(n_retrains): # p for period
        X_train_p = np.hstack([X_train, X_val[:, : p*update_period]])
        y_train_p = np.hstack([y_train, y_val[:, : p*update_period]])
        X_val_p = X_val[:, p*update_period: (p+1)*update_period]
        R_val_p = R_val[p*update_period: (p+1)*update_period]

        A_p = get_A(X_train_p, y_train_p, alpha, weighted=weighted)

        As[p] = A_p
        X_vals[p] = X_val_p
        R_vals[p] = R_val_p

    return As, X_vals, R_vals





def get_A(X_train, y_train, alpha, weighted=True):
    
    I = np.eye(X_train.shape[0], X_train.shape[0])

    if weighted:
        T_half = 2*252
        beta = np.exp(-np.log(2)/T_half)
        W_diag_flipped = np.array([beta**i for i in range(X_train.shape[1])])
        W_diag = np.flip((W_diag_flipped / np.sum(W_diag_flipped))).reshape(-1,1)
        At = np.linalg.solve(X_train@(W_diag*X_train.T) + alpha*I, X_train@ (W_diag*y_train.T))

        return At.T
    else:
    
        I = np.eye(X_train.shape[0], X_train.shape[0])
        At = np.linalg.solve(X_train@X_train.T + alpha*I, X_train@y_train.T)
        return At.T



def get_AR_format_with_features(X, l, features, R=None, EWMAs=None, dates=None, pred_horizon=0):
        """
        param l: look-back horizon in VAR(l) model
        """

        if R is None:
            R = X.copy()

        y_new = X.T[:,l:].copy()

        if pred_horizon >= 1:
            y_new = np.cumsum(y_new, axis=1)
            y_new[:, pred_horizon+1:] = y_new[:, pred_horizon+1:] - y_new[:, :-(pred_horizon+1)]
            y_new = y_new[:, pred_horizon:] / (pred_horizon+1)

        
        R_new = R[l:]
        if pred_horizon >= 1:
            R_new = np.cumsum(R_new, axis=0)
            R_new[pred_horizon+1:] = R_new[pred_horizon+1:] - R_new[:-(pred_horizon+1)]
            R_new = R_new[pred_horizon:] / (pred_horizon+1) 

        X = X.copy()
        X = np.hstack([X, features])

        if EWMAs is not None:
            if dates is not None:
                X_new, _, R_new, EWMAs_new, dates_new = get_AR_format(X, l, R=R, EWMAs=EWMAs, dates=dates, pred_horizon=pred_horizon)
                return X_new, y_new, R_new, EWMAs_new, dates_new
            else:
                X_new, _, R_new, EWMAs_new = get_AR_format(X, l, R=R, EWMAs=EWMAs, dates=dates, pred_horizon=pred_horizon)
                return X_new, y_new, R_new, EWMAs_new
        else:
            if dates is not None:   
                X_new, _, R_new, dates_new = get_AR_format(X, l, R=R, EWMAs=EWMAs, dates=dates, pred_horizon=pred_horizon)
                return X_new, y_new, R_new, dates_new
            else: 
                X_new, _, R_new = get_AR_format(X, l, R=R, EWMAs=EWMAs, dates=dates, pred_horizon=pred_horizon)
                return X_new, y_new, R_new



def get_AR_format(X, l, R=None, EWMAs=None, dates=None, pred_horizon=0):
        """
        param l: look-back horizon in VAR(l) model
        """
        if R is None:
            R = X.copy()

        X = X.T
        T = X.shape[1]
        n_feats = X.shape[0]

        y_new = X[:,l:]

        if pred_horizon >= 1:
            y_new = np.cumsum(y_new, axis=1)
            y_new[:, pred_horizon+1:] = y_new[:, pred_horizon+1:] - y_new[:, :-(pred_horizon+1)]
            y_new = y_new[:, pred_horizon:] / (pred_horizon+1) 

        

        R_new = R[l:]
        if pred_horizon >= 1:
            R_new = np.cumsum(R_new, axis=0)
            R_new[pred_horizon+1:] = R_new[pred_horizon+1:] - R_new[:-(pred_horizon+1)]
            R_new = R_new[pred_horizon:] / (pred_horizon+1) 

        


        X_new = np.zeros((n_feats*l, T-l))
        for i in range(l):
            x_i = X[:,l-i-1:T-i-1]
            X_new[i*n_feats:(i+1)*n_feats, :] = x_i 

        X_new = X_new[:, :X_new.shape[1]-pred_horizon]
        

        if EWMAs is not None:
            if type(EWMAs) != tuple:
                EWMAs_new = EWMAs[l+pred_horizon:]
            elif type(EWMAs) == tuple:
                if len(EWMAs) == 3:
                    EWMAs_new = (EWMAs[0][l+pred_horizon:], EWMAs[1][l+pred_horizon:], EWMAs[2][l+pred_horizon:])
            if dates is not None:
                dates_new = dates[l:dates.shape[0]-pred_horizon]
                return X_new, y_new, R_new, EWMAs_new, dates_new
            return X_new, y_new, R_new, EWMAs_new
            

        else:
            if dates is not None:
                dates_new = dates[l:dates.shape[0]-pred_horizon]
                return X_new, y_new, R_new, dates_new
            return X_new, y_new, R_new

def get_AR_format_old(X, l, R=None, EWMAs=None, dates=None):
        """
        param l: look-back horizon in VAR(l) model
        """
        if R is None:
            R = X.copy()
        X = X.T
        T = X.shape[1]
        n_feats = X.shape[0]

        X_new = np.zeros((n_feats*l, T-l))
        
        for i in range(l):

            x_i = X[:,l-i-1:T-i-1]
            X_new[i*n_feats:(i+1)*n_feats, :] = x_i 
        y_new = X[:,l:]
        R_new = R[l:]

        if EWMAs is not None:
            EWMAs_new = EWMAs[l:]
            if dates is not None:
                dates_new = dates[l:]
                return X_new, y_new, R_new, EWMAs_new, dates_new
            return X_new, y_new, R_new, EWMAs_new
        else:
            if dates is not None:
                dates_new = dates[l:]
                return X_new, y_new, R_new, dates_new
            return X_new, y_new, R_new


