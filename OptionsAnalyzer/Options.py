import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
from dateutil.parser import parse
import yfinance as yf
import pandas_datareader.data as web

class Option:

    def __init__(self, ticker, strike, price, call_put, short_long, expiration, Sk_onPurchase, r_onPurchase, stock_beta, date_purch):
        self.ticker = ticker
        self.strike = strike
        self.price = price
        self.call_put = call_put
        self.short_long = short_long
        self.expiration = expiration if isinstance(expiration, datetime) else parse(expiration)
        self.Sk_onPurchase = Sk_onPurchase
        self.r_onPurchase = r_onPurchase
        self.stock_beta = stock_beta
        self.date_purch = date_purch

    def impl_vol(self, stock_price, r, date):

        '''
        Takes stock price, r, and date and returns the IV of an option.
        '''

        date = date if isinstance(date, datetime) else parse(date)
        tol = 0.0001
        max_iterations=1000
        vol = 0.5

        for i in range(max_iterations):
            theos = self.theo_values(stock_price, r, vol, date)
            diff = theos.loc['Theo Price'] - self.price
            if abs(diff) < tol:
                break
            vol = vol - diff / theos.loc['Vega']
        return vol

    def get_current_values(self):

        '''
        Returns the current price and IV of an option from the yahoo finance API.
        '''

        ticker = yf.Ticker(self.ticker)
        if self.call_put == 'call':
            chain = ticker.option_chain(self.expiration.strftime('%Y-%m-%d')).calls
        if self.call_put == 'put':
            chain = ticker.option_chain(self.expiration.strftime('%Y-%m-%d')).puts
        chain = chain[chain['strike'] == self.strike]
        implVol = chain['impliedVolatility']
        currentPrice = chain['lastPrice'].iloc[0]
        return pd.Series([currentPrice, implVol], index=['currentPrice', 'implVol'])

        

    def get_exp_pl(self, stock_price):

        '''
        Takes stock price and returns the P/L of the option upon expiration.
        '''

        if self.call_put == 'call' and self.short_long == 'long':
            if self.strike < stock_price: return (stock_price - self.strike) - self.price
            if self.strike >= stock_price: return -self.price
        
        if self.call_put == 'put' and self.short_long == 'long':
            if self.strike > stock_price: return (self.strike - stock_price) - self.price
            if self.strike <= stock_price: return -self.price
            
        if self.call_put == 'call' and self.short_long == 'short':
            if self.strike < stock_price: return (self.strike - stock_price) + self.price
            if self.strike >= stock_price: return self.price

        if self.call_put == 'put' and self.short_long == 'short':
            if self.strike > stock_price: return (stock_price - self.strike) + self.price
            if self.strike <= stock_price: return self.price

    def theo_values(self, stock_price, r, vol, date):

        '''
        Takes stock price, r, IV and date; returns a Series with index: Theo Value, Delta, Gamma, Theta, Vega and Rho.
        '''

        date = date if isinstance(date, datetime) else parse(date)
        t = (self.expiration - date).days / 365
        N = norm.cdf
        theo_values = []
        d1 = (np.log(stock_price/self.strike) + (r + vol**2/2) * t) / (vol*np.sqrt(t))
        nnd1 = (1/np.sqrt(2*np.pi)) * np.exp(-d1**2/2)
        d2 = d1 - vol * np.sqrt(t)
        nnd2 = (1/np.sqrt(2*np.pi)) * np.exp(-d2**2/2) 

        if self.call_put == 'call':
            theo_value = stock_price * N(d1) - self.strike * np.exp(-r*t) * N(d2)
            delta = N(d1)
            gamma = nnd1/(stock_price*vol*np.sqrt(t))
            theta = -(stock_price * nnd1 * vol/2* np.sqrt(t)) - r * self.strike * np.exp(-r*t) * N(d2)
            vega = stock_price * np.sqrt(t) * nnd1
            rho = self.strike * t * np.exp(-r*t) * N(d2)
            theo_values.extend((theo_value, delta, gamma, theta, vega, rho))
            theo_values = pd.Series(theo_values, index=['Theo Price', 'Delta', 'Gamma', 'Theta', 'Vega', 'Rho'])
            return theo_values
        if self.call_put == 'put':
            theo_value = self.strike * np.exp(-r*t) * N(-d2) - stock_price*N(-d1)
            delta = N(d1) - 1
            gamma = nnd1/(stock_price*vol*np.sqrt(t))
            theta = -(stock_price * nnd1 * vol / 2 * np.sqrt(t)) + r * self.strike * np.exp(-r*t) * (1 - N(d2))
            vega = stock_price * np.sqrt(t) * nnd1
            rho = -self.strike * t * np.exp(-r*t) * (1-N(d2))
            theo_values.extend((theo_value, delta, gamma, theta, vega, rho))
            theo_values = pd.Series(theo_values, index=['Theo Price', 'Delta', 'Gamma', 'Theta', 'Vega', 'Rho'])
            return theo_values

def graph_hockeystick(option_list):

    '''
    Takes a list of Option instances and creates a graph of P/L on expiration.
    '''

    option_strikes = [x.strike for x in option_list]

    min_stock_price = round(min(option_strikes)) - 5 if round(min(option_strikes)) - 5 >= 0 else 0 
    max_stock_price = round(max(option_strikes)) + 5

    stock_range = range(min_stock_price, max_stock_price + 1)

    pLs = {}

    for option in option_list:
        pL_list = []
        for price in stock_range:
            pL = option.get_exp_pl(price) 
            pL_list.append(pL)
        
        pLs[option] = pL_list

    pL_df = pd.DataFrame(pLs, index=stock_range)

    pL_df['Total P/L'] = pL_df.sum(axis=1)

    pL_df['Total P/L'].plot(title='P/L of Options(s) on Expiration')
    plt.xticks(stock_range)
    plt.axhline(y=0, color='k')
    plt.show()











        



