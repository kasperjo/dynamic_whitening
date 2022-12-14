import numpy as np
from utils import *


class Whitened_VAR():
    def __init__(self, R, train_frac=0.5, val_frac=0.5, time_dep=False, features=[]) -> None:
        
        
        self.R = R.values
        self.n = R.shape[1]
        self.dates = R.index

        self.features = features
        self.train_len = int(train_frac*len(R))
        self.val_len = int(val_frac*len(R))

        self.whitened = False
        self.winsorized = False
        self.skipped_R = 0
        self.time_dep = time_dep

    def predict(self, X_val):
        return (self.A@X_val).T
    
    def get_AR_format(self, l):
        """
        param l: look-back horizon in VAR(l) model
        """
        
        R = self.R_win.copy()

        if self.F is not None:
            F = self.F.copy()
            X = np.hstack([R, F])
        else:
            X = R
        
        X = X.T
        T = X.shape[1]
        n_feats = X.shape[0]

        X_new = np.zeros((n_feats*l, T-l))
        
        for i in range(l):

            x_i = X[:,l-i-1:T-i-1]
            X_new[i*n_feats:(i+1)*n_feats, :] = x_i 
        y_new = R.T[:,l:]

        

        
        # Train, val, test sets
        X_train, X_val, X_test = X_new[:, :self.train_len],\
                 X_new[:, self.train_len:self.train_len+self.val_len],\
                     X_new[:, self.train_len+self.val_len:]

        y_train, y_val, y_test = y_new[:, :self.train_len],\
                 y_new[:, self.train_len:self.train_len+self.val_len],\
                     y_new[:, self.train_len+self.val_len:]


        # Store dates
        dates = self.dates[l+self.skipped_R:].copy()  
    
        dates_train, dates_val, dates_test = dates[:self.train_len],\
                 dates[self.train_len:self.train_len+self.val_len],\
                     dates[self.train_len+self.val_len:]

           
        
        if self.whitener is not None:
            if self.whitener != "double":
                EWMAs_new = self.EWMAs[l:]
                EWMAs_train, EWMAs_val, EWMAs_test = EWMAs_new[:self.train_len],\
                    EWMAs_new[self.train_len:self.train_len+self.val_len],\
                        EWMAs_new[self.train_len+self.val_len:]
                self.EWMAs_train, self.EWMAs_val, self.EWMAs_test = EWMAs_train, EWMAs_val, EWMAs_test
                self.EWMAs_lagged[l] = (EWMAs_train, EWMAs_val, EWMAs_test)
            else:
                sqrt_Sigma_hats_new = self.sqrt_Sigma_hats[l:]
                sqrt_V_hats_new = self.sqrt_V_hats[l:]

                sqrt_Sigma_hats_train, sqrt_Sigma_hats_val, sqrt_Sigma_hats_test = \
                    sqrt_Sigma_hats_new[:self.train_len],\
                    sqrt_Sigma_hats_new[self.train_len:self.train_len+self.val_len],\
                        sqrt_Sigma_hats_new[self.train_len+self.val_len:]
                sqrt_V_hats_train, sqrt_V_hats_val, sqrt_V_hats_test = \
                    sqrt_V_hats_new[:self.train_len],\
                    sqrt_V_hats_new[self.train_len:self.train_len+self.val_len],\
                        sqrt_V_hats_new[self.train_len+self.val_len:]
                        
                self.sqrt_Sigma_hats_train, self.sqrt_Sigma_hats_val, self.sqrt_Sigma_hats_test\
                     = sqrt_Sigma_hats_train, sqrt_Sigma_hats_val, sqrt_Sigma_hats_test
                self.sqrt_V_hats_train, self.sqrt_V_hats_val, self.sqrt_V_hats_test\
                     = sqrt_V_hats_train, sqrt_V_hats_val, sqrt_V_hats_test

                self.sqrt_Sigma_hats_lagged[l] = (sqrt_Sigma_hats_train, sqrt_Sigma_hats_val, sqrt_Sigma_hats_test)
                self.sqrt_V_hats_lagged[l] = (sqrt_V_hats_train, sqrt_V_hats_val, sqrt_V_hats_test)

            R_new = self.R_new[l:]
        else:
            R_new = self.R[l:] # TODO: right?


            
        R_train, R_val, R_test = R_new[:self.train_len],\
                R_new[self.train_len:self.train_len+self.val_len],\
                    R_new[self.train_len+self.val_len:]


        if self.whitener is not None:
            self.R_lagged[l] = (R_train, R_val, R_test)


        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

        self.R_train, self.R_val, self.R_test = R_train, R_val, R_test
        self.dates_train, self.dates_val, self.dates_test = dates_train, dates_val, dates_test

        
        self.X_lagged[l] = (X_train, X_val, X_test)
        self.y_lagged[l] = (y_train, y_val, y_test)
        self.R_lagged[l] = (R_train, R_val, R_test)
        self.dates_lagged[l] = (dates_train, dates_val, dates_test)


    def get_AR_model(self, l, alpha=0.1):
        """
        param l: look-back horizon in VAR(l) model
        param alpha: regularization parameter
        """
        self.get_AR_format(l)

        if not self.crossval:
            None
        else:
            n_train = self.X_train.shape[1]
            X_all = np.hstack([self.X_train, self.X_val])
            y_all = np.hstack([self.y_train, self.y_val])

            inds = np.arange(X_all.shape[1])
            np.random.shuffle(inds)
            train_inds = inds[:n_train]
            val_inds = inds[n_train:]


            X_train_new = X_all[:,train_inds]
            y_train_new = y_all[:,train_inds]
            X_val_new = X_all[:,val_inds]
            y_val_new = y_all[:,val_inds]

            R_all = np.vstack([self.R_train, self.R_val])
            self.R_train = R_all[train_inds,:]
            self.R_val = R_all[val_inds,:]

            if self.whitener is not None:
                EWMAs_all = np.vstack([self.EWMAs_train, self.EWMAs_val])
                self.EWMAs_train = EWMAs_all[train_inds]
                self.EWMAs_val = EWMAs_all[val_inds]
                



            X_val_new = X_all[:,val_inds]
            y_val_new = y_all[:,val_inds]

            self.X_train = X_train_new
            self.y_train = y_train_new
            self.X_val = X_val_new
            self.y_val = y_val_new


        
        if self.time_dep:
            p_length=self.p_length
            start = 252*2 + self.validation_start
            A_periods = {}
            X_periods = {}
            y_periods = {}
            R_periods = {}
            dates_periods = {}
            EWMA_periods ={}
            sqrt_Sigma_hat_periods = {}
            sqrt_V_hat_periods = {}


            n_periods = np.floor((self.R_train.shape[0]-start)/p_length).astype(int)
            for p in range(n_periods):
                # X_train = self.X_train[:, p_length*p : start + p_length*p]
                # y_train = self.y_train[:, p_length*p : start + p_length*p]
                X_train = self.X_train[:,  p_length*p+self.validation_start: start + p_length*p]
                y_train = self.y_train[:, p_length*p+self.validation_start: start + p_length*p]

                I = np.eye(X_train.shape[0], X_train.shape[0])
                self.weigted_ls = True
                if not self.weigted_ls:
                    A_T = np.linalg.solve(X_train@X_train.T + alpha*I, X_train@y_train.T)
                else:
                    T_half = 252
                    beta = np.exp(-np.log(2)/T_half)
                    W_diag = np.array([beta**i for i in range(X_train.shape[1])])
                    W_diag = np.flip((W_diag / np.sum(W_diag))).reshape(-1,1)
                    self.W_diag = W_diag
                    A_T = np.linalg.solve(X_train@(self.W_diag*X_train.T) + alpha*I, X_train@ (self.W_diag*y_train.T))
                A = A_T.T

                X_periods[p] = self.X_train[:, start + p_length*p : start + p_length*(p+1)]
                y_periods[p] = self.y_train[:, start + p_length*p : start + p_length*(p+1)]
                R_periods[p] = self.R_train[start + p_length*p : start + p_length*(p+1)]
                dates_periods[p] = self.dates_train[start + p_length*p : start + p_length*(p+1)]
                A_periods[p] = A.copy()

                if self.whitener is not None:
                    if self.whitener != "double":
                        EWMA_periods[p] = self.EWMAs_train[start + p_length*p : start + p_length*(p+1)]
                    else:
                        sqrt_Sigma_hat_periods[p] = self.sqrt_Sigma_hats_train[start + p_length*p : start + p_length*(p+1)]
                        sqrt_V_hat_periods[p] = self.sqrt_V_hats_train[start + p_length*p : start + p_length*(p+1)]

            self.X_periods = X_periods
            self.y_periods = y_periods
            self.R_periods = R_periods
            self.A_periods = A_periods
            self.dates_periods = dates_periods

            if self.whitener is not None:
                if self.whitener != "double":
                    self.EWMA_periods = EWMA_periods
                else:
                    self.sqrt_Sigma_hat_periods = sqrt_Sigma_hat_periods
                    self.sqrt_V_hat_periods = sqrt_V_hat_periods




        else:
            I = np.eye(self.X_train.shape[0], self.X_train.shape[0])
            A_T = np.linalg.solve(self.X_train@self.X_train.T + alpha*I, self.X_train@self.y_train.T)
            A = A_T.T
            self.A = A


    def get_features(self):
        if "std_dev" in self.features:
            std = self.R_win[:self.train_len+self.val_len].std(axis=0)
            above_std = np.maximum((self.R_win - std), 0)
            below_std = np.maximum(-(self.R_win - std), 0)
            self.F = np.hstack([above_std, below_std])
        if "pos_neg" in self.features:
            pos = np.maximum(np.abs(self.R_win), 0)
            neg = np.maximum(np.abs(-self.R_win), 0)
            if "std_dev" in self.features:
                self.F = np.hstack([self.F, pos, neg])
            else:
                self.F = np.hstack([pos, neg])
        else:
            self.F = None


    def train(self, whitener=None, l=1, alpha=0.1, beta=None, train_frac=None, val_frac=None, train_len=None, val_len=None,\
         crossval=False, EWMA_reg=0, time_dep=False, p_length=25, weighted_ls=True, validation_start=0):
        self.crossval=crossval
        self.whitener = whitener
        self.horizon=l
        self.alpha = alpha
        self.beta = beta
        self.time_dep=time_dep
        self.p_length=p_length
        self.weigted_ls = weighted_ls
        self.validation_start = validation_start

        if train_frac is not None:
            self.train_len = int(train_frac*len(self.R))
        if val_frac is not None:
            self.val_len = int(val_frac*len(self.R))
        if train_len is not None:
            self.train_len = train_len
        if val_len is not None:
            self.val_len = val_len


        if self.whitener is None:
            self.R_win = self.R.copy()
            self.R_win[:self.train_len+self.val_len] = np.maximum(self.R_win[:self.train_len+self.val_len], np.percentile(self.R_win[:self.train_len+self.val_len], 1, axis=0))
            self.R_win[:self.train_len+self.val_len] = np.minimum(self.R_win[:self.train_len+self.val_len], np.percentile(self.R_win[:self.train_len+self.val_len], 99, axis=0))
        
            
            

        else:
            if whitener != "double":
                EWMAs, r_tildes, R_new, skipped_R = get_whitened_data(R=self.R.copy(), whitener=whitener, beta=beta, EWMA_reg=EWMA_reg)
                self.EWMAs = EWMAs.copy()
            
            else:
                (sqrt_V_hats, sqrt_Sigma_hats), (r_tildes_var, r_tildes),\
                        R_new, skipped_R =\
                            get_whitened_data(R=self.R.copy(), whitener=whitener, beta=beta, EWMA_reg=EWMA_reg)
                self.sqrt_V_hats = sqrt_V_hats
                self.sqrt_Sigma_hats = sqrt_Sigma_hats
                self.r_tildes_var = r_tildes_var

            # Winsorize
            r_tildes[:self.train_len+self.val_len] = np.maximum(r_tildes[:self.train_len+self.val_len], np.percentile(r_tildes[:self.train_len+self.val_len], 1, axis=0))
            r_tildes[:self.train_len+self.val_len] = np.minimum(r_tildes[:self.train_len+self.val_len], np.percentile(r_tildes[:self.train_len+self.val_len], 99, axis=0))

            self.whitened = True
            self.R_win = r_tildes.copy()
            self.R_new = R_new.copy()


            if skipped_R is not None:
                self.skipped_R = skipped_R
            self.l = l

        self.get_features()
        self.get_AR_model(l, alpha=alpha) 

    def validate(self, verbose=False, test_dates="all", val_type="return"):
        if not self.time_dep:
            R_true = self.R_val
            dates = self.dates_val
            if self.whitener is None:
                R_hat = self.predict(self.X_val)
            else:
                R_hat = get_R(self.EWMAs_val, self.predict(self.X_val))
            if verbose:
                print(f"Correlation:  {get_column_corr(R_hat, R_true):.2%}")
            
            return R_hat, R_true, dates, get_column_corr(R_hat, R_true)
        if self.time_dep:
            R_hat = None
            R_true = None
            EWMAs = None
            dates = None
            sqrt_Sigma_hats = None
            sqrt_V_hats = None
            R_tilde_hat = None
            R_tilde_true = None

            # Predict: (self.A@X_val).T
            for p in self.A_periods.keys():
                A_t = self.A_periods[p]
                X_t = self.X_periods[p]
                date_t = self.dates_periods[p]
                if self.whitener is not None:
                    r_tilde_hat_t = (A_t@X_t).T
                    if val_type == "tilde":
                        r_tilde_true_t = self.y_periods[p].T
                    elif val_type == "return":
                        if self.whitener != "double":
                            EWMA_t = self.EWMA_periods[p]
                            r_hat_t = get_R(EWMA_t, r_tilde_hat_t)
                        else: 
                            sqrt_Sigma_hat_t = self.sqrt_Sigma_hat_periods[p]
                            sqrt_V_hat_t = self.sqrt_V_hat_periods[p]
                            r_hat_t = get_R(sqrt_Sigma_hat_t, r_tilde_hat_t, sqrt_V_hat_t)
                        
                    
                else:
                    r_hat_t = (A_t@X_t).T
                r_true_t = self.R_periods[p]

                try:
                    if val_type == "tilde":
                        R_tilde_hat = np.vstack([R_tilde_hat, r_tilde_hat_t])
                        R_tilde_true = np.vstack([R_tilde_true, r_tilde_true_t])
                    else:
                        R_hat = np.vstack([R_hat, r_hat_t])
                        R_true = np.vstack([R_true, r_true_t])
                        if self.whitener is not None:
                            if self.whitener != "double":
                                EWMAs = np.vstack([EWMAs, EWMA_t])
                            else:
                                sqrt_Sigma_hats = np.vstack([sqrt_Sigma_hats, sqrt_Sigma_hat_t])
                                sqrt_V_hats = np.vstack([sqrt_V_hats, sqrt_V_hat_t])

                    dates = np.hstack([dates, date_t])
                except:
                    if val_type == "tilde":
                        R_tilde_hat = r_tilde_hat_t
                        R_tilde_true = r_tilde_true_t
                    else:
                        R_hat = r_hat_t
                        R_true = r_true_t
                        if self.whitener is not None:
                            if self.whitener != "double":
                                EWMAs = EWMA_t
                            else:
                                sqrt_Sigma_hats = sqrt_Sigma_hat_t
                                sqrt_V_hats = sqrt_V_hat_t
                    
                    dates = date_t
            if type(test_dates) != str:
                inds_to_keep = np.isin(dates, test_dates)
                dates = dates[inds_to_keep]
                R_hat = R_hat[inds_to_keep]
                R_true = R_true[inds_to_keep]
                if self.whitener is not None:
                    if self.whitener != "double":
                        EWMAs = EWMAs[inds_to_keep]
                    else:
                        sqrt_V_hats = sqrt_V_hats[inds_to_keep]
                        sqrt_Sigma_hats = sqrt_Sigma_hats[inds_to_keep]


            # R_hat = np.array(R_hat)  
            # R_true = np.array(R_true)  

            if verbose:
                if val_type == "tilde":
                    print(f"Correlation:  {get_column_corr(R_tilde_hat, R_tilde_true):.2%}")
                    return R_tilde_hat, R_tilde_true, dates, get_column_corr(R_tilde_hat, R_tilde_true)
                print(f"Correlation:  {get_column_corr(R_hat, R_true):.2%}")
            if self.whitener is not None:
                if val_type == "tilde":
                    return R_tilde_hat, R_tilde_true, dates, get_column_corr(R_tilde_hat, R_tilde_true)
                if self.whitener != "double":
                    return R_hat, R_true, EWMAs, dates, get_column_corr(R_hat, R_true)
                else:
                    return R_hat, R_true, (sqrt_V_hats, sqrt_Sigma_hats), dates, get_column_corr(R_hat, R_true)
                    
            else:
                return R_hat, R_true, dates, get_column_corr(R_hat, R_true)

            


                
            



                        


