from Options import Option
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date as dt
import datetime
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from dateutil.parser import parse

class Transactions:

    def __init__(self, path):
        self.path = path
        self.orig_table = pd.read_excel(self.path)
        self.table = self.clean()
        self.open_positions = self.open_positions()
        self.closed_positions = self.closed_positions()

    def clean(self):

        '''
        Returns cleaned transactions table.
        '''

        cleaned = self.orig_table.drop(['REG FEE', 'SHORT-TERM RDM FEE', 'FUND REDEMPTION FEE', ' DEFERRED SALES CHARGE'], axis=1)
        cleaned.drop(cleaned.tail(1).index, inplace=True)
        mapper = {'COMMISSION': 'Commission', 'AMOUNT': 'Amount', 'DATE': 'Date', 'TRANSACTION ID': 'Tran ID', 
                   'DESCRIPTION': 'Desc', 'QUANTITY': 'Quantity', 'SYMBOL': 'Symbol', 'PRICE': 'Price'}
        cleaned.rename(columns=mapper, inplace=True)
        cleaned.set_index('Date', inplace=True)
        return cleaned

    def open_positions(self):

        '''
        Returns dataframe with open positions (does not currently work correctly for stock positions).
        '''

        symbols = self.table['Symbol'].tolist()
        opens = []
        for sym in symbols:
            count = symbols.count(sym)
            if count % 2 != 0:
                opens.append(sym)
        while np.nan in opens: opens.remove(np.nan)
        dates = []
        tran_ids = []
        descs = [] 
        quants = [] 
        syms = [] 
        prices = [] 
        comms = [] 
        amts = [] 
        for sym in opens:
            row = self.table[self.table['Symbol'] == sym]
            row.reset_index(inplace=True)
            dates.append(row['Date'].iloc[0])
            tran_ids.append(row['Tran ID'].iloc[0])
            descs.append(row['Desc'].iloc[0])
            quants.append(row['Quantity'].iloc[0])
            syms.append(row['Symbol'].iloc[0])
            prices.append(row['Price'].iloc[0])
            comms.append(row['Commission'].iloc[0])
            amts.append(row['Amount'].iloc[0])

        open_pos = {'Date': dates, 'Tran ID': tran_ids, 'Desc': descs, 'Quantity':quants, 'Symbol': syms, 
                    'Price': prices, 'Commission': comms, 'Amount': amts}
        open_positions = pd.DataFrame(open_pos)
        open_positions.set_index('Date', inplace=True)

        # Remove options that have are past expiration (expired or assigned)
        open_positions = create_options(open_positions)
        expired = []
        for index, row in open_positions.iterrows():
            if row['Option Object'].expiration.date() < dt.today():
                exp_option = row['Desc']
                expired.append(exp_option)
        open_positions = open_positions[~open_positions['Desc'].isin(expired)]

        return open_positions.dropna()
    
    def current_exposure(self):

        '''
        Takes Transactions instance and returns datafrane with current (theoretical) greeks for every ticker in portfolio. Assumes IV and r
        has remained constant since each option acquisition.
        '''
        open_positions = self.open_positions.dropna()
        date = str(dt.today())
        tickers = []
        deltas = [] 
        gammas = [] 
        thetas = [] 
        vegas = [] 
        rhos = []
        for index, row in open_positions.iterrows():
            ticker = str.split(str(row['Symbol']))[0]
            option = row['Option Object']
            vol = option.impl_vol(option.Sk_onPurchase, option.r_onPurchase, option.date_purch)
            stock_price = yf.Ticker(ticker).info['regularMarketPrice']
            r = option.r_onPurchase

            theo_values = option.theo_values(stock_price, r, vol, date)
            theo_values = theo_values.to_list()[1:]
            if option.short_long == 'short': #invert greeks if short
                theo_values = [-x for x in theo_values]
            deltas.append(theo_values[0]); gammas.append(theo_values[1]); thetas.append(theo_values[2]); vegas.append(theo_values[3]); rhos.append(theo_values[4])
            tickers.append(ticker)
               
        theo_values = {}
        theo_values['Ticker'] = tickers
        theo_values['Delta'] = deltas 
        theo_values['Gamma'] = gammas 
        theo_values['Theta'] = thetas 
        theo_values['Vega'] = vegas 
        theo_values['Rho'] = rhos
        theo_values = pd.DataFrame(theo_values)
        theo_values = theo_values.groupby('Ticker').sum()
        return theo_values
    
    def b_w_delta(self):

        '''
        Creates graph of the portfolio's beta weighted delta for different SPY prices. ** Assumes IV and r remains constant from option's purchase
        to expiration and that beta is an exact predictor of stock prices. ** 
        '''

        # Get all open option positions
        open_positions = self.open_positions.dropna()
        
        #Get 'risk free rate'
        r = yf.Ticker('^TNX').info['regularMarketPrice'] / 100

        #Add betas to table
        betas = []
        for index, row in open_positions.iterrows():
            betas.append(row['Option Object'].stock_beta)
        open_positions['beta'] = betas

        #Add vols (IV on purchase) to table
        vols = []
        stock_prices = []
        for index, row in open_positions.iterrows():
            option = row['Option Object']
            ticker = option.ticker
            stock_price = yf.Ticker(ticker).info['regularMarketPrice']
            stock_prices.append(stock_price)
            vol = option.impl_vol(option.Sk_onPurchase, option.r_onPurchase, option.date_purch)
            vols.append(vol)
        open_positions['Implied Volatility'] = vols
        open_positions['Stock Prices'] = stock_prices

        #get spy price and create spy price range
        spy_price = yf.Ticker('SPY').info['regularMarketPrice']
        mult_range = np.linspace(-0.1, 0.1, 21).tolist()
        SPY_prices = []
        for mult in mult_range:
            SPY_prices.append(round(spy_price + spy_price * mult, 2))

        #Calculate beta-weighted delta
        delta_sums = []
        today = str(dt.today())
        for price in SPY_prices:
            deltas = []
            for index, row in open_positions.iterrows():
                theo_pct_change = (price/spy_price - 1) * row['beta']
                theo_stock_price = row['Stock Prices'] - (row['Stock Prices'] * theo_pct_change)
                delta = row['Option Object'].theo_values(theo_stock_price, r, row['Implied Volatility'], today)['Delta']
                if row['Option Object'].short_long == 'short':
                    delta = -delta
                b_w_delta = (theo_stock_price / price) * row['beta'] * delta
                deltas.append(b_w_delta)

            delta_sums.append(np.sum(b_w_delta))

        b_w_delta = pd.Series(delta_sums, index=SPY_prices)
        plt.plot(b_w_delta)
        plt.show()
        return b_w_delta
            
    def current_mkt_exposure(self):

        '''
        Creates graph of theoretical P/L at different SPY prices through time. ** Assumes that IV and r of options remains constant from purchase 
        to expiration and that beta is an exactly accurate. **
        '''
        open_positions = self.open_positions.dropna()
        current_spy_price = yf.Ticker('SPY').info['regularMarketPrice']
        r = yf.Ticker('^TNX').info['regularMarketPrice'] / 100

        #add tickers, vols (IV on purchase) and betas to open_positions DataFrame
        tickers = []
        vols = []
        stock_prices = []
        betas = []
        for index, row in open_positions.iterrows():
            option = row['Option Object']
            ticker = option.ticker
            tickers.append(option.ticker)
            stock_price = yf.Ticker(ticker).info['regularMarketPrice']
            stock_prices.append(stock_price)
            vol = option.impl_vol(option.Sk_onPurchase, option.r_onPurchase, option.date_purch)
            vols.append(vol)
            betas.append(option.stock_beta)
        open_positions['Ticker'] = tickers
        open_positions['Implied Volatility'] = vols
        open_positions['Stock Price'] = stock_prices
        open_positions['Beta'] = betas
        
        #readjust dataframe for analysis
        open_positions.reset_index(inplace=True)
        open_positions.drop(['Date', 'Tran ID', 'Desc', 'Quantity', 'Symbol', 'Price', 'Commission', 'Amount'], axis=1, inplace=True)
        
        #get last option expiration and create date range from today until then
        expirations = []
        for index, row in open_positions.iterrows():
            expirations.append(row['Option Object'].expiration)
        last_exp = max(expirations)
        dates = pd.date_range(start=dt.today(), end=last_exp)
        
        mult_range = np.linspace(-0.2, 0.2, 99).tolist()
        SPY_prices = []
        for mult in mult_range:
            SPY_prices.append(round(current_spy_price + current_spy_price * mult, 2))
        vols.insert(0, np.nan)

        total_exposure = pd.DataFrame(index=SPY_prices)
        for date in dates:
            #convert open_positions DataFrame to stock-prices over 50% SPY loss to 50% spy gain with option object as index --> returns mkt_exposure DataFrame
            mkt_exposure = {} #add back below here
            mkt_exposure['SPY Prices'] = SPY_prices

            for index, row in open_positions.iterrows():
                new_prices = [] 
                for mult in mult_range:
                    new_price = round(row['Stock Price'] + row['Stock Price'] * mult * row['Beta'], 2)
                    new_prices.append(new_price if new_price > 0 else 0)
                mkt_exposure[row['Option Object']] = new_prices

            mkt_exposure = pd.DataFrame(mkt_exposure).transpose()
            mkt_exposure['Implied Vol'] = vols
            columnNum = len(mkt_exposure.columns)
            colRange = range(0, columnNum - 1)
            for index, row in mkt_exposure.iterrows():
                for colIndex in colRange:
                    if index != 'SPY Prices':
                        if date < index.expiration:
                            theo_value = index.theo_values(row[colIndex], r, row['Implied Vol'], str(date))['Theo Price']
                            if index.short_long == 'short':
                                row[colIndex] = index.price - theo_value
                            else:
                                row[colIndex] = theo_value - index.price
                        else:
                            row[colIndex] = index.get_exp_pl(row[colIndex])
            mkt_exposure.drop(['Implied Vol'], axis=1, inplace=True)
            mkt_exposure.drop(['SPY Prices'], axis=0, inplace=True)

            mkt_exposure = mkt_exposure.sum(axis=0)
            mkt_exposure.index = SPY_prices
            total_exposure[date] = mkt_exposure
        
        x = np.arange(len(total_exposure.columns))
        y = total_exposure.index
        X,Y = np.meshgrid(x,y)
        Z = total_exposure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z)
        ax.set_zlabel('P/L of Options Portfolio')
        plt.title('Theoretical P/L for SPY Price through Time')
        plt.xlabel('Dates ' + str(dt.today()) + ' through ' + str(last_exp)[:-9])
        plt.ylabel('SPY Price')
        fig.set_size_inches(10, 10)
        plt.show()
        return total_exposure

    def closed_positions(self):

        '''
        Get's all closed positions (does not currently work with stock positions). Does not create option objects by default!
        '''

        symbols = self.table['Symbol'].tolist()
        closed = []
        for sym in symbols:
            count = symbols.count(sym)
            if count % 2 == 0:
                closed.append(sym)
        while np.nan in closed: closed.remove(np.nan)
        dates = []
        tran_ids = []
        descs = [] 
        quants = [] 
        syms = [] 
        prices = [] 
        comms = [] 
        amts = [] 
        for sym in closed:
            row = self.table[self.table['Symbol'] == sym]
            row.reset_index(inplace=True)
            dates.append(row['Date'].iloc[0])
            tran_ids.append(row['Tran ID'].iloc[0])
            descs.append(row['Desc'].iloc[0])
            quants.append(row['Quantity'].iloc[0])
            syms.append(row['Symbol'].iloc[0])
            prices.append(row['Price'].iloc[0])
            comms.append(row['Commission'].iloc[0])
            amts.append(row['Amount'].iloc[0])

        closed_pos = {'Date': dates, 'Tran ID': tran_ids, 'Desc': descs, 'Quantity':quants, 'Symbol': syms, 
                    'Price': prices, 'Commission': comms, 'Amount': amts}
        closed_positions = pd.DataFrame(closed_pos)
        closed_positions.set_index('Date', inplace=True)
        return closed_positions
        
    def analyze(self):

        '''
        WILL eventually analyze statistical metrics of different options strategies. Thinking about how to do pairing logic.
        '''

        hist_table = self.closed_positions
        print(hist_table.to_string())


class Balances:
    
    def __init__(self, path):
        self.path = path
        self.table = pd.read_excel(path)

    def summary(self, transactions, start=None, end=None):

        '''
        WILL create an equity chart from balance history and transactions that shows returns of account vs returns of SPY.
        '''

        bal_table = self.table.copy()
        bal_table.dropna(inplace=True)
        tran_table = transactions.orig_table

        #see if dates passed to function, if not, default
        if not start:
            start = bal_table['Date'].iloc[0]
        else:
            start = parse(start)
        if not end:
            end = bal_table['Date'].iloc[len(bal_table)-1]
        else:
            end = parse(end)

        #resample/ffill to cover weekends too and filter by date arguments if needed
        bal_table.set_index('Date', inplace=True)
        bal_table = bal_table.asfreq('D', method='ffill')
        bal_table.reset_index(inplace=True)
        bal_table = bal_table[bal_table['Date'] >= start]
        bal_table = bal_table[bal_table['Date'] <= end]
        bal_table.reset_index(inplace=True)

        #Get all contributions from transactions and pull date and amt into dict
        cont_keys = ['ELECTRONIC NEW ACCOUNT FUNDING',
                     'CLIENT REQUESTED ELECTRONIC FUNDING RECEIPT (FUNDS NOW)']
        conts_table = tran_table[tran_table['DESCRIPTION'].isin(cont_keys)]
        conts = {row['DATE']: row['AMOUNT'] for index, row in conts_table.iterrows()}
        
        #get daily (returns + 1) !arithmetic and geometric factors
        returns = []
        arith_factors = []
        total_cont = bal_table['Account value'].iloc[0]
        for index, row in bal_table.iterrows():
            if index == 0: 
                factor = 1.0
                arith_factor = 1.0
            elif row['Date'] in (conts.keys()): #if date had contribution, add contribution to previous date's acc val in calc
                factor = row['Account value'] / (bal_table['Account value'].iloc[index - 1] + conts[row['Date']])
                total_cont += conts[row['Date']]
                arith_factor = row['Account value'] / total_cont
            else:
                factor = row['Account value'] / bal_table['Account value'].iloc[index - 1]
                if total_cont:
                    arith_factor = row['Account value'] / total_cont
                else: arith_factor = 1.0
            returns.append(factor)
            arith_factors.append(arith_factor)

        #Calculate equity over time ($100,000 initial) both geo and arith
        bal_table['cont adj returns'] = returns
        bal_table['arith factor'] = arith_factors
        bal_table['returns'] = bal_table['arith factor'] - 1
        bal_table['factor'] = bal_table['cont adj returns'].cumprod()
        bal_table['cont adj returns'] = bal_table['cont adj returns'] - 1
        bal_table['equity'] = bal_table['factor'] * 100000
        bal_table['arith equity'] = bal_table['arith factor'] * 100000

        #Get corresponding SPY equity
        SPY_data = web.get_data_yahoo('SPY', start - datetime.timedelta(days=5), end + datetime.timedelta(days=5))
        SPY_data = SPY_data.asfreq('D', method='ffill')
        SPY_data.reset_index(inplace=True)
        SPY_data = SPY_data[SPY_data['Date'] >= start] #this is done to make the indices match easier
        SPY_data = SPY_data[SPY_data['Date'] <= end]

        bal_table['SPY price'] = SPY_data['Open'].to_list()
        bal_table['SPY returns'] = (bal_table['SPY price'].shift() / bal_table['SPY price']) - 1
        bal_table['SPY returns'].iloc[0] = 0
        bal_table['SPY factor'] = bal_table['SPY price'] / bal_table['SPY price'].iloc[0]
        bal_table['SPY equity'] = bal_table['SPY factor'] * 100000
        bal_table.set_index('Date', inplace=True)

        #Graph SPY equity vs equity
        fig, (ax1, ax2) = plt.subplots(2)

        #Geometric graph
        ax1.plot(bal_table.index, bal_table['equity'].to_list())
        ax1.plot(bal_table.index, bal_table['SPY equity'].to_list())
        ax1.set_title('Contribution Adjusted Portfolio Returns vs SPY Returns ($100,000)')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Equity ($)')

        #Arithmetic graph
        ax2.plot(bal_table.index, bal_table['arith equity'].to_list())
        ax2.plot(bal_table.index, bal_table['SPY equity'].to_list())
        ax2.set_title('Portfolio Returns vs SPY Returns ($100,000)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Equity ($)')

        plt.show()

        #Get some summary statistics
        stats = pd.DataFrame(index=['Contribution Adjusted', 'Not Adjusted'])
        stats['Total Return'] = [bal_table['factor'].iloc[-1] - 1, bal_table['arith factor'].iloc[-1] - 1]

        geo_cov = np.cov(bal_table['cont adj returns'].to_list(), bal_table['SPY returns'].to_list())[0][1]
        geo_var = np.var(bal_table['cont adj returns'].to_list())
        arith_cov = np.cov(bal_table['returns'].to_list(), bal_table['SPY returns'].to_list())[0][1]
        arith_var = np.var(bal_table['returns'].to_list())
        stats['Beta'] = [geo_cov/geo_var, arith_cov/arith_var]

        return stats

        
              


def create_options(table):

    '''
    Takes (cleaned) dataframe and creates a new column of Option instances.
    '''

    options = []
    for index, row in table.iterrows():
            symbol = str(row['Symbol'])
            symbol = str.split(symbol)
            if len(symbol) > 1: #exclude nan values
                strike = float(symbol[-2])
                price = row['Price']
                call_put = symbol[-1].lower()
                short_long = 'long' if str.split(str(row['Desc']))[0] == 'Bought' else 'short'
                expiration = ' '.join(symbol[1:4])
                ticker = str.split(str(row['Desc']))[2]

                #query for more info
                date_purch = index
                if date_purch == dt.today():
                    Sk_onPurchase = yf.Ticker(ticker).info['regularMarketPrice']
                    r_onPurchase = r
                else:
                    Sk_onPurchase = web.get_data_yahoo(ticker, start=date_purch, end=date_purch)['Open'].iloc[0]
                    r_onPurchase = web.get_data_yahoo('^TNX', start=date_purch, end=date_purch)['Open'].iloc[0] / 100

                stock_beta = get_beta(ticker)

                options.append(Option(ticker, strike, price, call_put, short_long, expiration, Sk_onPurchase, r_onPurchase, stock_beta, date_purch))

            else:
                options.append(np.nan)
    table['Option Object'] = options
    return table.dropna()
                                     
def get_beta(ticker):

    '''
    Takes arg ticker and get's beta with SPY.
    '''

    start = dt.today() - datetime.timedelta(days=5*365)
    tickers = [ticker, 'SPY']
    data = web.get_data_yahoo(tickers, start, interval='m')
    data = data['Adj Close']
    log_returns = np.log(data/data.shift())

    cov = log_returns.cov()
    var = log_returns['SPY'].var()

    return cov.loc[ticker, 'SPY']/var