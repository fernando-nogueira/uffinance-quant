import pandas as pd
import numpy as np
import datetime as dt
import pandas_datareader as yf
import matplotlib.pyplot as plt

assets = ["VALE3.SA", "PETR4.SA", "ITUB4.SA", "BBDC4.SA", "PETR3.SA", "B3SA3.SA", "ABEV3.SA", "HAPV3.SA", "WEGE3.SA",
"ITSA4.SA", "JBSS3.SA", "BBAS3.SA", "SUZB3.SA", "GGBR4.SA", "RENT3.SA", "BPAC11.SA", "EQTL3.SA", "RDOR3.SA", "CSAN3.SA", "VBBR3.SA"]

end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=3652)
df = pd.concat([yf.get_data_yahoo(assets,
                                   start=start_date,
                                   end=end_date)['Close']], axis=1)

dfv = pd.concat([yf.get_data_yahoo(assets,
                                   start=start_date,
                                   end=end_date)['Volume']], axis=1)

for column in df.columns:
    if pd.isna(df[column].iloc[0]) == True:
        del df[column]
    
for column in dfv.columns:
    if pd.isna(dfv[column].iloc[0]) == True:
        del dfv[column]

del df['SUZB3.SA']
del dfv['SUZB3.SA']

dfr = df.pct_change().dropna()
dfv = dfv.iloc[1:]

def annually_rebalancing(df):
    dfc = df.copy()
    dfc = dfc.set_index(pd.to_datetime(dfc.index))
    dfc['DateAux'] = dfc.index
    dfc['DateAux'] = dfc['DateAux'].dt.year.astype(str)
    return dfc

def monthly_rebalancing(df):
    dfc = df.copy()
    dfc = dfc.set_index(pd.to_datetime(dfc.index))
    dfc['DateAux'] = dfc.index
    dfc['DateAux'] = dfc['DateAux'].dt.month.astype(str) + '| ' + dfc['DateAux'].dt.year.astype(str)
    return dfc

# Primeira parte do estudo, criar uma lista de dataframes por ano, perder o primeiro para criar  o retorno acumulado e a vol
dfrc = dfr.copy()
dfrc_y = annually_rebalancing(dfrc)
dfrc_m = monthly_rebalancing(dfrc)

lst_df_year = []
lst_df_month = []

for year in dfrc_y['DateAux'].unique():
    lst_df_year.append(dfrc_y[dfrc_y['DateAux'] == year])

for month in dfrc_m['DateAux'].unique():
    lst_df_month.append(dfrc_m[dfrc_m['DateAux'] == month])

lst_returns_year = [] 
lst_vol_year = []
for df in lst_df_year:
    df_acumulado = (1 + df[df.columns[:-1]]).cumprod() - 1
    lst_returns_year.append(df_acumulado.iloc[-1])
    lst_vol_year.append(df[df.columns[:-1]].std())

lst_quartil_momentum_assets = []
for df in lst_returns_year:
    num = df.describe()['75%']
    lst_quartil_momentum_assets.append(df[df >= num].index)
    
lst_quartil_low_volatility_assets = []
for df in lst_vol_year:
    num = df.describe()['25%']
    lst_quartil_low_volatility_assets.append(df[df <= num].index)
lst_quartil_momentum_assets[-1]
lst_quartil_low_volatility_assets[-1]
# Não existe 2023
lst_quartil_momentum_assets = lst_quartil_momentum_assets[:-1]
lst_quartil_low_volatility_assets = lst_quartil_low_volatility_assets[:-1]

# 100% momentum
lst_aux = list(range(1, len(lst_quartil_momentum_assets) + 1))

df_momentum = pd.DataFrame()
for num, group_assets in enumerate(lst_quartil_momentum_assets):
    serie_acumulada = lst_df_year[lst_aux[num]][group_assets]
    serie_acumulada.columns = ['ativo_1', 'ativo_2', 'ativo_3', 'ativo_4']
    df_momentum = pd.concat([df_momentum, serie_acumulada], axis=0)

# df_momentum_acumulado = (1 + df_momentum).cumprod() - 1

def fixed_portfolio(df, weights, name_portfolio = 'Portfolio'):
    # Getting a copy of a dataframe, reseting your index for a safe loc iteration 
    dfc = df.copy().reset_index(drop=True)
    num_assets = len(dfc.columns)
    # Iteration of w_i * r_i vectors
    for row in range(0, len(dfc)):
        dfc.loc[row, name_portfolio] = np.transpose(weights) @ np.array(dfc[dfc.columns[0:num_assets]].iloc[row])
    return dfc.set_index(df.index)

df_momentum_portfolio = fixed_portfolio(df_momentum, [0.25, 0.25, 0.25, 0.25], name_portfolio='Momentum')['Momentum']
df_momentum_portfolio_acumulado = (1 + df_momentum_portfolio).cumprod() - 1

df_low_volatility = pd.DataFrame()
for num, group_assets in enumerate(lst_quartil_low_volatility_assets):
    serie_acumulada = lst_df_year[lst_aux[num]][group_assets]
    serie_acumulada.columns = ['ativo_1', 'ativo_2', 'ativo_3', 'ativo_4']
    df_low_volatility = pd.concat([df_low_volatility, serie_acumulada], axis=0)

df_low_volatility_portfolio = fixed_portfolio(df_low_volatility, [0.25, 0.25, 0.25, 0.25], name_portfolio='Low-Volatility')['Low-Volatility']
df_low_volatility_portfolio_acumulado = (1 + df_low_volatility_portfolio).cumprod() - 1

end_date = dt.datetime.now()
start_date = dt.datetime(2012, 12, 28, 0, 0)
df_ibov = yf.get_data_yahoo('^BVSP',
                        start=start_date,
                        end=end_date)['Close']
df_ibov
df_ibov_acumulado = (1 + df_ibov.pct_change().dropna()).cumprod() - 1

resultado_estrategias = pd.concat([df_ibov_acumulado, df_momentum_portfolio_acumulado, df_low_volatility_portfolio_acumulado], axis=1)

fig, ax = plt.subplots()
ax.plot(resultado_estrategias)
ax.legend(resultado_estrategias.columns)
plt.show()

