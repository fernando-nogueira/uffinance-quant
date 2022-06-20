import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
import seaborn as sns
import plotly.graph_objects as go
from PIL import Image

CB91_Blue = '#012a4a'
CB91_Green = '#b89000'
CB91_Pink = '#2c7da0'
CB91_Purple = '#ffda1f'
CB91_Violet = '#1360e2'
CB91_Amber = '#fc7802'

color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
              CB91_Purple, CB91_Violet]

sns.set_context("notebook", rc={"font.size":16,
                                "axes.titlesize":20,
                                "axes.labelsize":18})

sns.set(font='Franklin Gothic Book',
        rc={
 'axes.axisbelow': False,
 'axes.edgecolor': 'lightgrey',
 'axes.facecolor': 'None',
 'axes.grid': False,
 'axes.labelcolor': 'dimgrey',
 'axes.spines.right': False,
 'axes.spines.top': False,
 'figure.facecolor': 'white',
 'lines.solid_capstyle': 'round',
 'patch.edgecolor': 'w',
 'patch.force_edgecolor': True,
 'text.color': 'dimgrey',
 'xtick.bottom': False,
 'xtick.color': 'dimgrey',
 'xtick.direction': 'out',
 'xtick.top': False,
 'ytick.color': 'dimgrey',
 'ytick.direction': 'out',
 'ytick.left': False,
 'ytick.right': False})

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

def bacen_api(serie):
    url = f'http://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie}/dados?formato=json'
    df = pd.read_json(url)
    df['data'] = pd.to_datetime(df['data'], dayfirst=True)
    df = df.set_index('data')
    return df

def mean_reversion(df, column='Close', window=21, n_std = [1, 1]):
    dfc = df.copy()
    mean = dfc[column].ewm(span=window, min_periods=window).mean()
    std = dfc[column].rolling(window=window).std()
    dfc['MEAN'] = mean
    dfc['MEAN_MORE_STD'] = mean + (n_std[0] * std) 
    dfc['MEAN_LESS_STD'] = mean - (n_std[0] * std)
    return dfc.dropna()

def strategy(df):
    buy_points = []
    sell_points = []
    strategy_lst = []
    returns = df['Close'].pct_change()
    i = 0
    while i + 1 <= len(df):
        price = df['Close'][i]
        c = 0
        # Checar se entramos em alerta da estratégia
        if price <= df['MEAN_LESS_STD'][i]:
            buy_points.append(price) # Efetuamos o ponto de compra
            sell_points.append(np.nan)
            strategy_lst.append(0)
            print('[BUY POINT]')
            i = i + 1
            # Entramos em um loop dado o ponto de entrada até o alerta de ponto de saída
            while price < df['MEAN_MORE_STD'][i + c]:
                c = c + 1
                if i + c + 1 >= len(df):
                    print('[OUT STRATEGY]')
                    price = df['Close'][i + c]
                    break
                else:
                    price = df['Close'][i + c]
                    buy_points.append(np.nan) # Estamos dentro da estratégia
                    sell_points.append(np.nan)
                    strategy_lst.append(returns[i + c])
                    i = i + 1
                    print('[IN STRATEGY]')
            # Fim do loop de entrada, foi dado o alerta do ponto de saída
            buy_points.append(np.nan)
            sell_points.append(price)
            strategy_lst.append(returns[i + c - 1])
            i = i + 1
            print('[SELL POINT]')
        else:
            #  Não temos a estratégia nesse dia
            print('[OUT STRATEGY]')
            buy_points.append(np.nan)
            sell_points.append(np.nan)
            strategy_lst.append(0)
            i = i + 1
            
    returns_strategy = pd.Series(strategy_lst)
    returns_strategy.index = df.index
    return buy_points, sell_points, returns_strategy

def year_filter(df, year):
    return df[df.index.year == year]
    
def ipca_series(df_reindex, mult = None):
    year = df_reindex.index.max().year
    df = bacen_api(433)
    df = df[df.index.year == year]
    ipca_daily = []
    for month in df.index.month:
        n_days = len(df_reindex[df_reindex.index.month == month])
        days_in_month = [d.day for d in df_reindex[df_reindex.index.month == month].index]
        for day in days_in_month:
            ipca_month = df[df.index.month == month]['valor'][0] / 100
            ipca_daily.append((1 + ipca_month) ** (1/n_days) - 1)        
            
    ipca_daily_series = pd.Series(ipca_daily)
    ipca_daily_series.index = df_reindex.index
    
    if mult == None:
        return pd.DataFrame(ipca_daily_series).reindex(df_reindex.index)
        
    else:
        daily_mult = (1 + mult) ** (1/len(ipca_daily_series)) - 1
        return pd.DataFrame((1 + ipca_daily_series) * (1 + daily_mult) - 1).reindex(df_reindex.index)

def cdi_series(df_reindex):
    year = df_reindex.index.max().year
    df = bacen_api(12) / 100
    df = df[df.index.year == year]
    return pd.DataFrame(df).reindex(df_reindex.index)

def drawdown_series(returns):
    acum_returns = (1 + returns).cumprod()
    peaks = acum_returns.cummax()
    ddw = (acum_returns - peaks)/peaks 
    ddw_series = pd.Series(ddw)
    return ddw_series

def volatility_series(returns, window=21):
    return returns.rolling(window=window).std() * np.sqrt(252)
    
def alpha_series(returns1, returns2):
    return (1 + returns1)/(1+returns2) - 1

def beta_series(dfv, dfv_bench, window=21):
    df = pd.concat([dfv, dfv_bench],axis=1)
    name_assets = df.columns.values.tolist()
    df['Beta'] = 0
    for row in range(0, len(dfv)-window):
        df['Beta'].iloc[row+window] = df.iloc[row:row+window][name_assets[0]].cov(df.iloc[row:row+window][name_assets[1]]) / df.iloc[row:row+window][name_assets[1]].var()
    return df['Beta'][window:len(df)]

# def sharpe_rolling(returns, selic, window=21):
#     risk_return_lst = []
#     for row in range(len(returns) - window):
#         return_asset = (1 + returns.iloc[row:row+window]).cumprod().iloc[-1] - 1
#         return_risk_free = (1 + selic.iloc[row:row+window]).cumprod().iloc[-1]['valor'] - 1
#         vol_asset = returns.iloc[row:row+window].std() * np.sqrt(window)
#         if vol_asset == 0:
#             sharpe = (return_asset - return_risk_free)
#         else:
#             sharpe = (return_asset - return_risk_free)/vol_asset
#         risk_return_lst.append(sharpe)

#     sharpe_series = pd.Series(risk_return_lst)
#     sharpe_series.index = returns.index[window:]
#     return sharpe_series

def profit_ratio(buy_points, sell_points):
    # np.isnan(buy_p[0]) -> Podemos ter mais pontos de compra e de venda (fazer função de profit)
    bp = [b for b in buy_points if not np.isnan(b)]
    sp = [s for s in sell_points if not np.isnan(s)]
    if len(bp) > len(sp):
        bp.pop()
    signal_lst = []
    for n_signal in range(len(bp)):
        signal_sign = sp[n_signal] / bp[n_signal] - 1
        if signal_sign > 0:
            signal = 1
        elif signal_sign < 0:
            signal = 0 
        signal_lst.append(signal)
    return sum(signal_lst) / len(signal_lst)

def operation_date_match(df_operation, df_info):
    df, dfc = df_operation.copy(), df_info.copy()
    lst = []
    df_t = dfc.copy()
    for i in range(len(df)):
        for ii in range(len(df_t)):
            if df_t.index[ii] == df.index[i]:
                lst.append(df[df.columns[0]].iloc[i])
                df_t = df_t.iloc[ii:]
                break
            else:
                lst.append(np.nan)
    if len(lst) < len(dfc):
        for i in range(len(dfc) - len(lst)):
            lst.append(np.nan)
    if len(lst) > len(dfc):
        print(lst)
        for i in range(len(lst) - len(dfc)):
            lst.pop(-2)
            
    return lst

def mean_reversion_strategy_points(df):
    dfc = df.copy()
    entry_points = []
    date_entry_operation = []
    date_exit_operation = []
    exit_points = []
    for i in range(len(dfc)):
        if len(entry_points) == len(exit_points):
            # Podemos comprar
            if dfc['MEAN_LESS_STD'].iloc[i] >= dfc['Close'].iloc[i]:
                date_entry_operation.append(dfc.index[i])
                entry_points.append(dfc['Close'].iloc[i])
            # Podemos vender
        elif len(entry_points) > len(exit_points):
            if dfc['MEAN_MORE_STD'].iloc[i] <= dfc['Close'].iloc[i]:
                date_exit_operation.append(dfc.index[i])
                exit_points.append(dfc['Close'].iloc[i])
    df_exit = pd.DataFrame({"EXIT": exit_points})
    df_exit.index = date_exit_operation
    df_entry = pd.DataFrame({"ENTRY" : entry_points})
    df_entry.index = date_entry_operation
    lst_entry_points = operation_date_match(df_entry, dfc)
    lst_exit_points = operation_date_match(df_exit, dfc)
    dfc['ENTRY'] = lst_entry_points
    dfc['EXIT'] = lst_exit_points
    return dfc

# df = yf.download(asset)
# dfc = year_filter(mean_reversion(df, n_std=[2,2]), year)
# dfc = dfc.reset_index()
# dfc = mean_reversion_strategy_points(dfc)

# fig = go.Figure()
# traces = []
# for column in ['Close', 'MEAN', 'MEAN_MORE_STD', 'MEAN_LESS_STD']:
#     trace = go.Scatter(x=dfc['Date'], y=dfc[column], name=column.capitalize().replace('_',' '))
#     traces.append(trace)

# traces.append(go.Scatter(mode='markers',x=dfc['Date'], y=dfc['ENTRY'] * 0.99, marker_symbol='triangle-up', marker=dict(size=10,color='rgba(1, 152, 117, 1)')))
# traces.append(go.Scatter(mode='markers',x=dfc['Date'], y=dfc['EXIT'] * 1.01, marker_symbol='triangle-down', marker=dict(size=10,color='rgba(178,34,34, 1)')))
# # 'triangle-up
# # triangle-down'
# fig.add_traces(traces)
# fig.show()