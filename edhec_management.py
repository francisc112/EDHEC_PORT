#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 19:20:53 2021

@author: francisco
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from scipy.optimize import minimize



class Risk:
    
    def __init__(self):
        pass

    def max_drawdown(self,returns):
        wealth_index = 1000 * (1 + returns).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        return drawdowns.min()

    def wealth_to_date(self,returns,initial_investment = 1000):
        wealth_index = initial_investment * (1 + returns).cumprod()
        return wealth_index.iloc[-1]

        
    def drawdown(self,returns,plot_drawdown=False):
          """
          Takes a time series of asset returns 
          Computes and returns a DataFrame that contains:
  
              The wealth index
              The previous peaks
              Percent drawdowns
  
          """
          
          wealth_index = 1000 * (1+ returns).cumprod()
          previous_peaks = wealth_index.cummax()
          drawdowns = (wealth_index - previous_peaks) / previous_peaks    
          
              
          if plot_drawdown == True:
              drawdowns.plot(kind='line')
              plt.title(f'Drawdown for {returns.name}')
              print(f"Worst drawdown is {round(drawdowns.min(),2)} and it was in {drawdowns.idxmin()}")
          
          return pd.DataFrame({
          'wealth index':wealth_index,
          'previous peaks':previous_peaks,
          'drawdown':drawdowns
          })
      
        
      
    def kurtosis(self,return_series:pd.Series):
          
          """ 
          Alternative to scipy.stats.kurtosis()
          Computes the kurtosis of the supplied series or DataFrame (exess kurtosis (already substracts 3))
          Returns a float or a series
          """
          demeaned_returns = return_series - return_series.mean()

          # Use the population standard deviation, so set dof=0
          sigma_r = return_series.std(ddof=0)
          exp  = (demeaned_returns ** 4).mean()
          return (exp/sigma_r ** 4) - 3
      
        
    def skewness(self,return_series:pd.Series):
        """ 
        Alternative to scipy.stats.skew()
        Computes the skewness of the supplied series or DataFrame
        A negative skewness means you get more negative returns than you would normally expect
        If the median return is less than the mean return (you have negatively skewed returns)
        Returns a float or a series
        """ 

        demeaned_returns = return_series - return_series.mean()

        # Use the population standard deviation, so set dof=0
        sigma_r = return_series.std(ddof=0)
        exp  = (demeaned_returns ** 3).mean()
        return exp/sigma_r ** 3
    
    
    
    # Sharpe Ratio
    
    
    def annualize_vol(self,r, periods_per_year):
        """ 
        Annualizes the vol of a set of returns 
        
        """
        return r.std() * (periods_per_year ** 0.5)
    
    def annualize_rets(self,r,periods_per_year):
        
        compounded_growth = (1+r).prod()
        
        n_periods = r.shape[0]

        return compounded_growth ** (periods_per_year / n_periods) - 1
    
    def sharpe_ratio(self,r, riskfree_rate, periods_per_year):
        """ 
        Computes the annualized sharpe ratio of a set of returns
        
        """
        # Convert the annualized riskfree rate to period 
        
        rf_per_period = (1+riskfree_rate) ** (1/periods_per_year) - 1
        
        excess_ret = r - rf_per_period
        
        ann_ex_ret = self.annualize_rets(excess_ret, periods_per_year)
        
        ann_vol = self.annualize_vol(r, periods_per_year)

        return ann_ex_ret / ann_vol
    
    def portfolio_returns(self,weights,returns):
        """ 
        Weights -> Returns
    
        It behaves like a =SUMPRODUCT() in Excel
    
        """
    
        return weights.T @ returns

    def portfolio_vol(self,weights, covmat):
        
        """ 
        Weights -> Vol
        """
    
        return (weights.T @ covmat @ weights) ** 0.5
    
      
        
    def plot_ef2(self,er,cov,n_points):
        
        """ 
        
        Plots the 2 asset efficient frontier
        
        """
        
        if er.shape[0] != 2:
            raise ValueError('Plot ef2 can only plot two asset efficient frontier')
            
        
        weights = [np.array([w, 1-w]) for w in np.linspace(0,1,n_points)]
        
        rets = [self.portfolio_returns(w,er) for w in weights]
        
        vols = [self.portfolio_vol(w,cov) for w in weights]
        
        ef = pd.DataFrame({
            'Returns':rets,
            'Volatility':vols
        })
        
        return ef.plot.line(x='Volatility',y='Returns',style='.-')
    
    def minimize_vol(self,target_return,er,cov):

        """
        target_return -> Weight Vector
        """
    
        n = er.shape[0]
        init_guess = np.repeat(1/n, n)
        bounds = ((0.0,1),) * n # Go as low as 0 and go as high as 1 in terms of weights. The result is a tuple of tuples (that return constantly (0.0,1))
    
        return_is_target = {
            'type':'eq',
            'args':(er, ),
            'fun': lambda weights, er: target_return - self.portfolio_returns(weights,er)
        }
    
        weights_sum_to_1 = {
            'type':'eq',
            'fun': lambda weights: np.sum(weights) - 1
        }
    
        results = minimize(self.portfolio_vol, init_guess, args=(cov, ), method = "SLSQP", options={'disp':False}, constraints=(return_is_target,weights_sum_to_1), bounds=bounds)
    
        return results.x
    
    def optimal_weights(self,n_points,er,cov):
        """ 
        -> list of weights to run the optimizer to minizme the vol
        """
        
        target_rs = np.linspace(er.min(), er.max(), n_points)
        weights = [self.minimize_vol(target_return, er, cov) for target_return in target_rs]
        
        return weights
        
        
    
    def plot_ef(self,er,cov,n_points):
        
        """ 
        
        Plots the 2 asset efficient frontier
        
        """
        
        weights = self.optimal_weights(n_points,er,cov)
        
        rets = [self.portfolio_returns(w,er) for w in weights]
        
        vols = [self.portfolio_vol(w,cov) for w in weights]
        
        ef = pd.DataFrame({
            'Returns':rets,
            'Volatility':vols
        })
        
        return ef.plot.line(x='Volatility',y='Returns',style='.-')
    


    

    
    
        
      
      
# stocks = get_ticker_returns(['AAPL','AMZN','IBM','NFLX'])

# risk = Risk(stocks['AAPL'])

# risk.drawdown(plot_drawdown=True)['drawdown'].plot(kind='line')

        
        