from Options import Option
from Options import graph_hockeystick
from yahoo_fin import options
import pandas as pd

print('Ticker?')
ticker = input().upper()

print(str(options.get_expiration_dates(ticker)))

n = 0
option_list = [] 
expirations = []

while n == 0:
    print('What expiration(s)?')
    expiration = input()
    expirations.append(expiration)
    print('Another? (y/n)')
    if input() == 'y':
        pass
    else:
        n=1

for exp in expirations:
    c_chain = options.get_calls(ticker, exp)
    p_chain = options.get_puts(ticker, exp)

    print('Want calls in ' + exp + '? (y/n)')
    if input() == 'y':
        print(c_chain[['Strike', 'Last Price']].to_string())
        while n == 1:
            print('State index for desired call in ' + exp +'?')
            indx = int(input())
            strike = c_chain['Strike'].loc[indx]
            price = c_chain['Last Price'].loc[indx]
            call_put = 'call'
            print('long/short?')
            long_short = input()
            option_list.append(Option(strike, price, call_put, long_short, exp))

            print('Another call in ' + exp + '? (y/n)')
            if input() == 'y':
                pass
            else:
                n = 0
    n = 1

    print('Want puts in ' + exp + '? (y/n)')
    if input() == 'y':
        print(p_chain[['Strike', 'Last Price']].to_string())
        while n == 1:
            print('State index for desired put in ' + exp +'?')
            indx = int(input())
            strike = c_chain['Strike'].loc[indx]
            price = c_chain['Last Price'].loc[indx]
            call_put = 'put'
            print('long/short?')
            long_short = input()
            option_list.append(Option(strike, price, call_put, long_short, exp))
        

            print('Another put in ' + exp + '? (y/n)')
            if input() == 'y':
                pass
            else:
                n = 0

graph_hockeystick(option_list)
        

        





    

