import pandas as pd
import numpy as np
import akshare as ak
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def get_index_data(symbol, startdate, enddate):
    index_data = ak.index_zh_a_hist(
        symbol=symbol, 
        period="daily", 
        start_date=startdate, 
        end_date=enddate
    )
    index_data.index = pd.to_datetime(index_data['日期'])
    index_data = index_data[['开盘', '最高', '最低', '收盘', '成交量']]
    index_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    index_data = index_data.sort_index()
    return index_data

def calculate_rsrs_slope(index_data, N=16):
    rsrs_slope = []
    rsquared_values = []
    for i in range(N, len(index_data)):
        high_prices = index_data['High'].iloc[i-N:i].values.reshape(-1, 1)
        low_prices = index_data['Low'].iloc[i-N:i].values.reshape(-1, 1)
        
        model = LinearRegression()
        model.fit(low_prices, high_prices)
        
        beta = model.coef_[0][0]
        rsrs_slope.append(beta)
        
        # 计算决定系数 R^2
        rsquared = model.score(low_prices, high_prices)
        rsquared_values.append(rsquared)
    
    rsrs_slope_series = pd.Series(rsrs_slope, index=index_data.index[N:])
    rsquared_series = pd.Series(rsquared_values, index=index_data.index[N:])
    return rsrs_slope_series, rsquared_series

def calculate_rsrs_zscore(rsrs_slope_series, rsquared_series, M=300):
    rsrs_zscore = []
    for i in range(M, len(rsrs_slope_series)):
        slope_window = rsrs_slope_series.iloc[i-M:i]
        mean_slope = slope_window.mean()
        std_slope = slope_window.std()
        
        current_slope = rsrs_slope_series.iloc[i]
        z_score = (current_slope - mean_slope) / std_slope
        
        # 修正标准分
        rsquared = rsquared_series.iloc[i]
        adjusted_z_score = z_score * rsquared
        
        rsrs_zscore.append(adjusted_z_score)
    
    rsrs_zscore_series = pd.Series(rsrs_zscore, index=rsrs_slope_series.index[M:])
    return rsrs_zscore_series

def backtest_strategy(index_data, rsrs_zscore_series, S=0.7):
    signals = pd.Series(index=rsrs_zscore_series.index)
    position = 0  # 初始空仓
    for i in range(1, len(rsrs_zscore_series)):
        if rsrs_zscore_series.iloc[i] > S and position == 0:
            signals.iloc[i] = 1  # 买入信号
            position = 1  # 持仓
        elif rsrs_zscore_series.iloc[i] < -S:
            signals.iloc[i] = 0  # 卖出信号
            position = 0  # 空仓
        else:
            signals.iloc[i] = signals.iloc[i-1]  # 保持现有状态
    
    signals = signals.ffill().fillna(0)  # 持仓信号
    
    returns = index_data['Close'].pct_change()
    strategy_returns = returns[signals.index] * signals.shift(1)
    
    cumulative_returns = (1 + strategy_returns).cumprod()
    return cumulative_returns, strategy_returns, signals

def calculate_performance_metrics(cumulative_returns, strategy_returns, signals):
    final_value = cumulative_returns.iloc[-1]
    total_return = final_value - 1
    annualized_return = cumulative_returns.pct_change().mean() * 252
    annualized_volatility = strategy_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility
    max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
    holding_days = signals.sum()
    trade_count = signals.diff().abs().sum()
    winning_days = (strategy_returns > 0).sum()
    losing_days = (strategy_returns < 0).sum()
    
    return {
        'Final Value': final_value,
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Holding Days': holding_days,
        'Trade Count': trade_count,
        'Winning Days': winning_days,
        'Losing Days': losing_days
    }

# 示例使用
symbol = '000300'  # 沪深300指数代码
startdate = '20020301'
enddate = '20170401'

index_data = get_index_data(symbol, startdate, enddate)
rsrs_slope_series, rsquared_series = calculate_rsrs_slope(index_data, N=16)
rsrs_zscore_series = calculate_rsrs_zscore(rsrs_slope_series, rsquared_series, M=300)

# 回测策略
cumulative_returns, strategy_returns, signals = backtest_strategy(index_data, rsrs_zscore_series, S=0.7)

# 计算沪深300指数的累计收益
index_cumulative_returns = (1 + index_data['Close'].pct_change()).cumprod()

# 绘制累计收益曲线
plt.figure(figsize=(10, 6))
cumulative_returns.plot(label='RSRS Strategy')
index_cumulative_returns.plot(label='HS300 Index')
plt.title('Cumulative Returns of RSRS Strategy vs HS300 Index')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()

# 计算并显示绩效指标
performance_metrics = calculate_performance_metrics(cumulative_returns, strategy_returns, signals)
for metric, value in performance_metrics.items():
    print(f'{metric}: {value:.2f}')

# 打印信号
print(signals)