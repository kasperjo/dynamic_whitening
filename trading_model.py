import numpy as np
import cvxpy as cp
from tqdm import trange
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.dates as mdates




class Trader():
    def __init__(self, R, R_hat, sqrt_Sigma_hats, dates=None, bm=False, R_covs=None, memory=None) -> None:
        self.R = R
        self.R_covs = R_covs
        self.dates = dates
        self.R_hat = R_hat
        self.T = R.shape[0]
        self.n = R.shape[1]
        self.sqrt_Sigma_hats = sqrt_Sigma_hats
        self.memory = memory
        self.bm = bm

    def backtest(self, risk_des=0.05):
        self.successes = 0
        self.fails = 0
        w_t = [0 for _ in range(self.n+1)]
        w_t[0] = 1
        ws = []
        for t in trange(self.T):
            r_hat_t = self.R_hat[t].reshape(-1,1)
            sqrt_Sigma_hat_t = self.sqrt_Sigma_hats[t].T # Transpose for Cholesky

            w_t = self.get_w(w_t, r_hat_t, sqrt_Sigma_hat_t, risk_des)
            ws.append(w_t.reshape(-1,))
        self.ws = np.array(ws)


    def get_w(self, w_t, r_hat_t, sqrt_Sigma_hat_t, risk_des):


        r_hat_t = np.vstack([0,r_hat_t]) # Add cash return

        w = cp.Variable(self.n+1) # First weight is for cash

        ret = r_hat_t.T @ w
        risk = cp.norm(sqrt_Sigma_hat_t@w[1:], 2)
        

        obj = cp.Maximize(ret) # Maximize expected return
         
        # obj = cp.Minimize(w[0]) # Minimize risk

        cons =  [cp.sum(w) == 1, risk <= risk_des/np.sqrt(252)]
        # cons =  [cp.sum(w) == 1, w>=0, risk <= risk_des/np.sqrt(252)]



        
        # cons += [cp.norm(w, 1) <= 1.6]
        # cons = cons + [cp.sum(cp.neg(w)) <= 0.5]
        # cons = cons + [w>=0]
        # cons = cons + [w[1:]<=0.2]

        # cons = cons + [w>= 0]
        cons += [cp.norm(w, 1) <= 2]
        cons += [w >= -0.25]
        cons += [w[1:] <= 0.4]
        cons += [w[0] >= 0]

        
        # cons += [w>=0]
        # cons += [w >= 0]
        cp.Problem(obj, cons).solve()

        # try:
        #     problem = cp.Problem(obj, cons + [cp.norm(w_t-w, 1) <= 0.01])
        #     problem.solve()
        #     assert problem.status == cp.OPTIMAL
        #     self.successes += 1
        # except Exception:
        #     cp.Problem(obj, cons).solve()
        #     self.fails += 1

        return w.value

    def get_total_returns(self):
        rets = []
        for t in range(len(self.ws)):
            w_t = self.ws[t]
            r_t = self.R[t].reshape(-1,1)
            r_t = np.vstack([0, r_t]) # Add cash return
            rets.append(np.dot(w_t, r_t))
        self.rets = np.array(rets)

    def get_portfolio_growth(self):
        Vs = [np.array([1])]
        for i, r_t in enumerate(self.rets):
            # print(Vs[i])
            Vs.append(Vs[i]*(1+r_t))
        self.Vs = np.array(Vs)

    def show_performance(self, benchmarks={}, label="Portfolio performance", title=""):
        self.get_total_returns()
        self.get_portfolio_growth()
        print(f"Mean yearly return (portfolio): {np.mean(self.rets)*252:.2%}")
        print(f"Yearly risk (portfolio): {np.std(self.rets)*np.sqrt(252):.2%}")
        print("Sharpe ratio: ", np.round(np.mean(self.rets)*252 / (np.std(self.rets)*np.sqrt(252)), 2))

        if self.dates is not None:
            x_ticks = pd.to_datetime(self.dates)
        else:
            x_ticks = np.arange(len(self.ws))

        fig, ax = plt.subplots()
        fig.autofmt_xdate()    
        for benchmark in benchmarks.keys():
            ax.plot(x_ticks, benchmarks[benchmark].Vs[1:], label=benchmark)
            # plt.plot(benchmarks[benchmark]["x"], benchmarks[benchmark]["y"][1:], label=benchmark)

            
        ax.plot(x_ticks, self.Vs[1:], label=label)
        # plt.gca().xaxis.set_major_formatter(md.DateFormatter('%Y'))
        plt.legend(bbox_to_anchor=(1, 1),
          ncol=1, fancybox=True, shadow=True)
        ax.set_title(title)
        plt.plot()

    def get_metrics(self):
        self.get_total_returns()
        self.get_portfolio_growth()
        print(f"Mean yearly return (portfolio): {np.mean(self.rets)*252:.2%}")
        print(f"Yearly risk (portfolio): {np.std(self.rets)*np.sqrt(252):.2%}")
        print(f"Sharpe ratio: {np.mean(self.rets)*252 / (np.std(self.rets)*np.sqrt(252)):.2}")

        
        


