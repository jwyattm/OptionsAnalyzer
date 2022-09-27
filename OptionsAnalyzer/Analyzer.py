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

class transactions:

    def __init__(self, path):
        self.path = path
        self.orig_table = pd.read_excel(self.path)
        self.table = self.clean()
        self.open_positions = self.open_positions()
        self.closed_positions = self.closed_positions()

    #return cleaned transaction table
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

    #return open positions
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
        options = create_options(open_positions)
        expired = []
        for index, row in options.iterrows():
            if row['Option Object'].expiration.date() < dt.today():
                exp_option = row['Desc']
                expired.append(exp_option)
        open_positions = open_positions[~open_positions['Desc'].isin(expired)]

        return open_positions
    
    def current_exposure(self):

        '''
        Takes transactions instance and r and returns datafrane with current (theoretical) greeks for every ticker in portfolio.
        '''

        open_positions = create_options(self.open_positions)
        open_positions = open_positions.dropna()
        r = yf.Ticker('^TNX').info['regularMarketPrice'] / 100
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
            vol = option.get_current_values()['implVol']
            stock_price = yf.Ticker(ticker).info['regularMarketPrice']

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
        open_positions = create_options(self.open_positions)
        
        #Get 'risk free rate'
        r = yf.Ticker('^TNX').info['regularMarketPrice'] / 100

        #Add betas to table
        betas = []
        for index, row in open_positions.iterrows():
            betas.append(get_beta(row['Option Object'].ticker))
        open_positions['beta'] = betas

        #Add vols (IV on purchase) to table
        vols = []
        stock_prices = []
        for index, row in open_positions.iterrows():
            date_purch = index
            desc = str.split(row['Desc'])
            ticker = desc[2]
            stock_price = yf.Ticker(ticker).info['regularMarketPrice']
            stock_prices.append(stock_price)
            option = row['Option Object']
            if date_purch == pd.Timestamp(dt.today()):
                SK_onPurchase = stock_price
                r_onPurchase = r
            else:
                Sk_onPurchase = web.get_data_yahoo(option.ticker, start=date_purch, end=date_purch)['Open'].iloc[0]
                r_onPurchase = web.get_data_yahoo('^TNX', start=date_purch, end=date_purch)['Open'].iloc[0] / 100
            vol = option.impl_vol(Sk_onPurchase, r_onPurchase, date_purch)
            vols.append(vol)
            print(str(ticker) + str(option.strike) + str(date_purch) + ' data loaded')
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
                print(theo_stock_price)
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

        open_positions = create_options(self.open_positions)
        current_spy_price = yf.Ticker('SPY').info['regularMarketPrice']
        r = yf.Ticker('^TNX').info['regularMarketPrice'] / 100

        #add tickers and vols (IV on purchase) to open_positions DataFrame
        tickers = []
        vols = []
        stock_prices = []
        for index, row in open_positions.iterrows():
            date_purch = index
            desc = str.split(row['Desc'])
            ticker = desc[2]
            tickers.append(ticker)
            stock_price = yf.Ticker(ticker).info['regularMarketPrice']
            stock_prices.append(stock_price)
            option = row['Option Object']
            if date_purch == dt.today():
                SK_onPurchase = stock_price
                r_onPurchase = r
            else:
                Sk_onPurchase = web.get_data_yahoo(option.ticker, start=date_purch, end=date_purch)['Open'].iloc[0]
                r_onPurchase = web.get_data_yahoo('^TNX', start=date_purch, end=date_purch)['Open'].iloc[0] / 100
            vol = option.impl_vol(Sk_onPurchase, r_onPurchase, date_purch)
            vols.append(vol)
            print(str(ticker) + str(option.strike) + str(date_purch) + ' data loaded')
        open_positions['Ticker'] = tickers
        open_positions['Implied Volatility'] = vols
        open_positions['Stock Price'] = stock_prices
        
        #add betas to open_positions DataFrame
        betas = []
        for ticker in tickers:
            beta = get_beta(ticker)
            betas.append(beta)
        open_positions['Beta'] = betas
        open_positions['Implied Volatility'] = vols
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
        
        print(total_exposure.transpose().to_string())
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
        plt.show()
        return total_exposure

    def closed_positions(self):

        '''
        Get's all closed positions (does not currently work with stock positions).
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
        closed_positions = create_options(closed_positions)
        return closed_positions
        
    def analyze(self):

        '''
        WILL eventually analyze statistical metrics of different options strategies. Thinking about how to do pairing logic.
        '''

        hist_table = self.closed_positions
        print(hist_table.to_string())


class bal_hist:
    
    def __init__(self, path):
        self.path = path
        self.table = pd.read_excel(path)

    def equity_chart(self, transactions):

        '''
        WILL create an equity chart from balance history and transactions that shows geometric returns of account vs geometric returns of SPY.
        '''

        pass


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
                options.append(Option(ticker, strike, price, call_put, short_long, expiration))
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