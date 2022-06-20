import streamlit as st
import pandas as pd
import fibonacci as fb
import yfinance as yf
# import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image

sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].values.tolist()
ibovespa_tickers = [f'{tick}.SA' for tick in pd.read_html('https://en.wikipedia.org/wiki/List_of_companies_listed_on_B3')[0]['Ticker'].values.tolist()]

st.title("UFFinance")
st.write(
    """

    **Mean Reversion**  \n
    """
)

dct_tickers = {'S&P500':sp500_tickers,
               'IBOVESPA' : ibovespa_tickers}

dp_choices_lst = [1, 1.25, 1.50, 1.75, 2, 2.25, 2.5, 2.75, 3]
st.sidebar.image(Image.open('somepic.png'))
st.sidebar.header("OPTIONS:")
market_choices = st.sidebar.selectbox("MARKET:", list(dct_tickers.keys()))
stock_choices = st.sidebar.selectbox("ASSET:", dct_tickers[market_choices])
year_choices = st.sidebar.selectbox("YEAR:", list(range(2015, 2023)))
window_choices = st.sidebar.selectbox("WINDOW:", [21, 42, 63, 74])
more_dp = st.sidebar.selectbox("(+) STD:", dp_choices_lst)
less_dp = st.sidebar.selectbox("(-) STD:", dp_choices_lst)
# ipca_choices = st.sidebar.selectbox("IPCA +:", [0.03, 0.04, 0.05, 0.06, 0.07, 0.08])

# ----------------------------------- BackEND
asset = stock_choices
year = year_choices
dp_lst = [more_dp, less_dp]
df = yf.download(asset)
dfc = fb.year_filter(fb.mean_reversion(df, n_std=dp_lst, window=window_choices), year).reset_index()
dfc = fb.mean_reversion_strategy_points(dfc)
profit_ratio_number = fb.profit_ratio(dfc['ENTRY'], dfc['EXIT'])
string_profit_ratio = '{:.2f} %'.format(profit_ratio_number * 100)
# ipca_string = '{:.2f} %'.format(ipca_choices * 100)

fig = go.Figure()
traces = []
for column in ['Close', 'MEAN', 'MEAN_MORE_STD', 'MEAN_LESS_STD']:
    trace = go.Scatter(x=dfc['Date'], y=dfc[column], name=column.capitalize().replace('_',' '))
    traces.append(trace)

traces.append(go.Scatter(mode='markers',x=dfc['Date'], y=dfc['ENTRY'], marker_symbol='triangle-up', marker=dict(size=10,color='rgba(1, 152, 117, 1)')))
traces.append(go.Scatter(mode='markers',x=dfc['Date'], y=dfc['EXIT'], marker_symbol='triangle-down', marker=dict(size=10,color='rgba(178,34,34, 1)')))
fig.add_traces(traces)
fig.update_layout(title_text=f"{asset} MEAN REVERSION STRATEGY IN {year} | PROFIT RATIO: {string_profit_ratio}",
                  )

st.plotly_chart(fig)



# fig1, ax1 = plt.subplots(figsize=(12, 6))
# ax1.plot(dfc['Date'], dfc['Close'], label='PRICE')
# ax1.plot(dfc['Date'], dfc['MEAN'], label='MEAN')
# ax1.plot(dfc['Date'], dfc['MEAN_MORE_STD'], label='MEAN + STD')
# ax1.plot(dfc['Date'], dfc['MEAN_LESS_STD'], label='MEAN - STD')
# ax1.plot(dfc['Date'], fb.strategy_points['BUY'], marker = '^', markersize = 5, color = 'green', label = 'BUY SIGNAL')
# ax1.plot(dfc['Date'], fb.streategy_points['SELL'], marker = 'v', markersize = 5, color = 'r', label = 'SELL SIGNAL')
# ax1.legend(fontsize=8)
# ax1.set_title(f"{asset} MEAN REVERSION STRATEGY IN {year} | PROFIT RATIO: {string_profit_ratio}")
# fig1.autofmt_xdate()

# fig2, ax2 = plt.subplots(figsize=(12, 6))
# ax2.plot((1 + dfc['Close'].pct_change()).cumprod(), label = 'CARRY')
# ax2.plot((1 + ret).cumprod(), label = 'STRATEGY')
# ax2.plot((1 + fb.ipca_series(ret, ipca_choices)).cumprod(), label = f'IPCA + {ipca_string}')
# ax2.plot((1 + fb.cdi_series(ret)).cumprod(), label = 'CDI')
# ax2.legend(fontsize=8)
# fig2.autofmt_xdate()

# fig3, ax3 = plt.subplots(figsize=(12, 6))
# ax3.plot(fb.alpha_series(ret, dfc['Close'].pct_change()), label='ALPHA STRATEGY')
# ax3.plot(fb.beta_series(ret, dfc['Close'].pct_change()), label='BETA STRATEGY')
# ax3.legend(fontsize=8)
# fig3.autofmt_xdate()

# fig4, ax4 = plt.subplots(figsize=(12, 6))
# ax4.plot(fb.drawdown_series(dfc['Close'].pct_change()), label=f'DDW {asset}')
# ax4.plot(fb.drawdown_series(ret), label='DDW STRATEGY')
# ax4.legend(fontsize=8)
# fig4.autofmt_xdate()

# fig5, ax5 = plt.subplots(figsize=(12, 6))
# ax5.plot(fb.volatility_series(dfc['Close'].pct_change()), label=f'VOL {asset}')
# ax5.plot(fb.volatility_series(ret), label='VOL STRATEGY')
# ax5.legend(fontsize=8)
# fig5.autofmt_xdate()

# # fig6, ax6 = plt.subplots(figsize=(12, 6))
# # ax6.plot(fb.sharpe_rolling(ret, fb.cdi_series(ret)), label='SHARPE RATIO STRATEGY')
# # ax6.plot(fb.sharpe_rolling(dfc['Close'].pct_change(), fb.cdi_series(dfc['Close'].pct_change())), label=f'SHARPE RATIO {asset}')
# # ax6.legend(fontsize=8)
# # fig6.autofmt_xdate()

# st.pyplot(fig1)
# st.pyplot(fig2)
# st.pyplot(fig3)
# st.pyplot(fig4)
# st.pyplot(fig5)
# # st.pyplot(fig6)

