"""
MCF(Monte-Carlo Particle Filter)
Copyright (c) 2018 Takuma Sakaki
This module is released under the MIT License.
http://opensource.org/licenses/mit-license.php

"""


####################################
###モンテカルロフィルタ(MCF)クラス
####################################

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
plt.style.use('fivethirtyeight')
from numpy.random import rand,normal

class MCF():
    
    system_equation = None
    obs_equation = None
    system_equation_gen = None
    obs_L = None
    total_step = None
    X_0 = None

    #X = [[x1,x2,...(t=0)],[x1,x2,...,(t=1)],...]
    X = None
    y = None
    sim_X = None


    def __init__(self, system_equation, obs_equation,system_equation_gen, obs_L, state_labels, X_0 ,total_step):
        self.system_equation = system_equation
        self.obs_equation = obs_equation
        self.system_equation_gen = system_equation_gen
        self.obs_L = obs_L
        self.state_labels = state_labels
        self.X_0 = X_0
        self.total_step = total_step
        

    def Set_totalstep(self,total_step):
        self.total_step = total_step

    def Set_obs(self,y):
        if len(y) != self.total_step + 1:
            print("the expected length of y is",(self.total_step + 1), "but the length of inputed y is",len(y))
        else:
            self.y = y

    def Set_truestates(self,X):
        self.X = X

    def Get_results(self):
        return self.sim_X


    #データ生成メソッド
    def DataGenerate(self, true_X_0, plot=False):

        T = self.total_step
        num_variables = len(true_X_0)
        X = np.empty([T+1, num_variables])
        y = np.empty(T+1)
        X[0] = true_X_0
        y[0] = None
        for t in range(1,T+1):
            X[t] = self.system_equation(X[t-1], t)
            y[t] = self.obs_equation(X[t])

        self.X = X
        self.y = y

        if plot == True:
            X = X.T
            colors = ['green', 'yellow', 'pirple']
            for i in range(num_variables):
                label = "true-state: " + self.state_labels[i]
                if i <= 2:
                    plt.plot(X[i], color = colors[i], label=label)
                else:
                    plt.plot(X[i], label=label)

            plt.plot(y, color = 'blue', label='observed-value')
            plt.xlabel("Time")
            plt.legend(loc='upper right', bbox_to_anchor=(1.05,0.5,0.5,.100), borderaxespad=0.)
            plt.show()


    #Calculation Aggregated Values
    def Get_summary(self, Per_CI=0.95, plot=False):

        if Per_CI <= 0 or Per_CI > 1:
            print("Per_CI should be [0,1].")

        if self.sim_X is None:
            print ("This Method can be used only after calculation")
            sys.exit()

        T = self.total_step
        num_variables = len(self.X_0)
        m = len(self.X_0[0])   #mは粒子の個数
        ave_x = np.full([T+1,num_variables], 100, dtype=np.float)
        mid_x = np.full([T+1,num_variables], 100, dtype=np.float)
        lower_CI = np.full([T+1,num_variables], 100, dtype=np.float)
        upper_CI = np.full([T+1,num_variables], 100, dtype=np.float)
        low = int(m * (1- Per_CI) / 2) - 1
        high = int(m - low) - 1

        for t in range(T+1):
            X_t = self.sim_X[t]
            for k in range(num_variables):
                x_t = np.sort(X_t[k])
                ave_x[t][k] = np.average(x_t)
                mid_x[t][k] = np.median(x_t)
                lower_CI[t][k] = x_t[low]
                upper_CI[t][k] = x_t[high]

        if (plot == True):
            #結果のプロット
            plt.figure(figsize=(6,num_variables*4))
            plt.subplots_adjust(hspace=0.5)
            for k in range(num_variables):
                plt.subplot(num_variables,1,k+1)
                plt.title(("Estimation of " + self.state_labels[k]))
                plt.plot(self.X.T[k], color = 'green', linewidth=2, label="true-state")
                plt.plot(ave_x.T[k], color = 'red', linewidth=2, label="estimated-state")
                plt.plot(upper_CI.T[k],'--', color = 'red', linewidth=2, label="upper-CI")
                plt.plot(lower_CI.T[k], '--', color = 'red', linewidth=2, label="lower-CI")
                plt.xlabel("Time")
                plt.legend(loc='upper right', bbox_to_anchor=(1.05,0.5,0.5,.100), borderaxespad=0.)
            
            plt.show()
        

        result = {}
        for k in range(num_variables):
            result[self.state_labels[k]] = {'ave_x':ave_x.T[k],'mid_x':mid_x.T[k],'lower_CI':lower_CI.T[k], 'upper_CI':upper_CI.T[k]}
        
        return result


    #平均二乗誤差
    def MSE(self, estimation_method = "average"):
        if self.sim_X is None:
            print ("This Method can be used only after calculation")
            sys.exit()
        if self.X is None:
            print ("This Method can be used only after defining true-state")
            sys.exit()

        result = {}
        for k in range(len(self.X_0)):

            MSE = 0
            T = self.total_step
            
            for t in range(1,T+1):
                if estimation_method == "average":
                    est_x = np.average(self.sim_X[t][k])
                elif estimation_method == "median":
                    est_x = np.median(self.sim_X[t][k])
                else:
                    print("error: estimation_method:",estimation_method)
                    sys.exit()

                MSE = MSE + ((self.X[t][k] - est_x) ** 2) / T
            
            result[self.state_labels[k]] = MSE

        return result

        

    #Total Calculation
    def Filtering(self, resampling_method = "original", smoothing_lag = 0):
        
        #変数チェックとセッティング
        if self.y is None:
            print("observed value (y) is not defined. Define it with the methods 'set_obs' or 'DataGenerate'")
        if resampling_method != "original" and resampling_method != "stratified":
            print("'original' or 'stratified' is only usable as resampling_method")
        T = self.total_step
        self.sim_X = np.empty((T+1,len(self.X_0), len(self.X_0[0])))

        #初期値
        X_samples = self.X_0
        self.sim_X[0] = X_samples
 
        #Main calc
        for t in tqdm(range(1, T+1)):
            X_samples = self.__Step(X_samples, t, resampling_method, smoothing_lag)

        print("Simulation finished successfully.")



    #1step
    def __Step(self, X_samples,t, resampling_method, smoothing_lag):

        num_variables = len(X_samples)
        m = len(X_samples[0])
        
        #RandomGenerate
        X_samples = self.system_equation_gen(X_samples, t)
        
        #likelihood
        y_t_array = np.full(m, self.y[t])
        w_t = self.obs_L(X_samples, y_t_array)

        #Resampling -- with high speed
        new_samples = np.empty([num_variables,m])
        if resampling_method == "original":
            u_t = rand(m)
        elif resampling_method == "stratified":
            u_t = rand(m) / m  + np.linspace(0, 1, m+1)[:-1]

        w_t = w_t / np.sum(w_t)
        w_t = w_t.cumsum()
        
        for i in range(m):
            u = u_t[i]
            try:
                j = np.where(w_t > u)[0][0]
            except:
                print(u,"is a unexpected number. plz debug accessing self.w")
                self.w = w_t
                sys.exit()
            
            for k in range(num_variables):
                new_samples[k][i] = X_samples[k][j]

            #Smoothing
            if smoothing_lag > 0:
                for past_time in range(max(0,t - smoothing_lag),t):
                    for k in range(num_variables):
                        self.sim_X[past_time][k][i] = self.sim_X[past_time][k][j]

        self.sim_X[t] = new_samples

        return new_samples


#テスト,線形ガウス型状態空間モデル
if __name__ == "__main__":
    print("Conducting test program...")

    #システム方程式、観測方程式は以下。
    #X=[x1, x2, ...]の形式
    def system_equation_1(X, t):
        v = normal(0,1,1)
        return np.array([X[0] + v])
    def obs_equation_1(X):
        e = normal(0,0.5)
        return X[0] + e
    
    #予測分布の発生式
    #X_samplesは[x1_samples, x2_samples, ...]の形式
    def system_equation_gen_1(X_samples, t):
        x_samples = X_samples[0]
        v = normal(0,1,len(x_samples))
        return np.array([x_samples + v])
    #尤度の計算式
    def obs_L_1(X,y):
        x = X[0]
        t = y-x
        return (1/np.sqrt(2*np.pi*0.25))*np.exp(-1 * (t**2) / 0.5)

    #10000個の0の値の粒子から、50ステップのシミュレーションを行う。
    MCF = MCF(system_equation_1, obs_equation_1, system_equation_gen_1, obs_L_1, ['x'], np.array([np.zeros(10000)]),50)

    #初期値x_0を0として、T=50までの真の状態x_tと観測値y_tを発生させる
    MCF.DataGenerate([0])
    #計算を行う
    MCF.Filtering()
    summary = MCF.Get_summary(plot=True)


"""
Reference:
Kitagawa G (1996) Monte Carlo filter and smoother for non-Gaussian nonlinear state space models. J Comput Graph Stat 5:1–25
矢野浩一（2014）粒子フィルタの基礎と応用：フィルタ・平滑化・パラメータ推定
"""