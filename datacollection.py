import pandas as pd
import yfinance as yf
import numpy as np

def fetch_stock_data(tickers, start_date, end_date):
    """
    Fetch historical stock price data for the given tickers and date range.
    
    Parameters:
    - tickers (list): List of stock tickers (e.g., ['TCS.BO', 'RELIANCE.BO']).  
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.
    
    Returns:
    - DataFrame: DataFrame containing adjusted close prices for each stock.
    """
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

def calculate_returns(data):
    """
    Calculate daily returns from the historical stock price data.
    
    Parameters:
    - data (DataFrame): DataFrame containing adjusted close prices.
    
    Returns:
    - DataFrame: DataFrame containing daily returns for each stock.
    """
    returns = data.pct_change().dropna()
    return returns

def calculate_cumulative_returns(data, period):
    """
    Calculate cumulative returns over the specified period for each stock.
    
    Parameters:
    - data (DataFrame): DataFrame containing daily returns for each stock.
    - period (int): Number of trading days in the period.
    
    Returns:
    - Series: Series containing cumulative returns for each stock.
    """
    cumulative_returns = (data.iloc[-period:] / data.iloc[-period]).fillna(1).prod() - 1
    return cumulative_returns

def rank_stocks(cumulative_returns):
    """
    Rank stocks based on their cumulative returns.
    
    Parameters:
    - cumulative_returns (Series): Series containing cumulative returns for each stock.
    
    Returns:
    - Series: Series containing ranks for each stock.
    """
    ranked_stocks = cumulative_returns.rank(ascending=False)
    return ranked_stocks

# Example usage
tickers = ['TCS.BO', 'RELIANCE.BO', 'HDFCBANK.BO', 'INFY.BO', 'ITC.BO']
start_date = '2023-01-01'
end_date = '2024-01-01'

# Fetch historical stock price data
stock_data = fetch_stock_data(tickers, start_date, end_date)

# Calculate daily returns
returns = calculate_returns(stock_data)

# Calculate cumulative returns over 1 year and rank stocks
period = 252  # 1 year (assuming 252 trading days)
cumulative_returns = calculate_cumulative_returns(returns, period)
ranked_stocks = rank_stocks(cumulative_returns)
print(ranked_stocks)

def select_top_n_stocks(ranked_stocks, n):
    """
    Select the top N stocks with the highest momentum scores.
    
    Parameters:
    - ranked_stocks (Series): Series containing ranks for each stock.
    - n (int): Number of top stocks to select.
    
    Returns:
    - list: List of top N stocks.
    """
    top_stocks = ranked_stocks.nsmallest(n).index.tolist()
    return top_stocks

# Example usage
n_top = 3  # Number of top stocks to select
top_stocks = select_top_n_stocks(ranked_stocks, n_top)
print("Top momentum stocks:")
print(top_stocks)

def rebalance_portfolio(returns, period, n_top):
    """
    Rebalance the portfolio by selecting the top N stocks with the highest momentum scores.
    
    Parameters:
    - returns (DataFrame): DataFrame containing daily returns for each stock.
    - period (int): Number of trading days in the rebalancing period.
    - n_top (int): Number of top stocks to select.
    
    Returns:
    - DataFrame: DataFrame containing the rebalanced portfolio.
    """
    portfolio = pd.DataFrame(columns=returns.columns)
    
    for i in range(period, len(returns), period):
        data_slice = returns.iloc[i-period:i]
        cumulative_returns = calculate_cumulative_returns(data_slice, period)
        ranked_stocks = rank_stocks(cumulative_returns)
        top_stocks = select_top_n_stocks(ranked_stocks, n_top)
        portfolio = portfolio.append(data_slice.loc[:, top_stocks])
    
    return portfolio

# Example usage
rebalanced_portfolio = rebalance_portfolio(returns, period=252, n_top=3)
print("Rebalanced portfolio:")
print(rebalanced_portfolio)



def backtest_strategy(returns, period, n_top):
    """
    Backtest the momentum strategy using historical data.
    
    Parameters:
    - returns (DataFrame): DataFrame containing daily returns for each stock.
    - period (int): Number of trading days in the rebalancing period.
    - n_top (int): Number of top stocks to select.
    
    Returns:
    - dict: Dictionary containing backtest results.
    """
    portfolio_values = []
    
    for i in range(period, len(returns), period):
        data_slice = returns.iloc[i-period:i]
        cumulative_returns = calculate_cumulative_returns(data_slice, period)
        ranked_stocks = rank_stocks(cumulative_returns)
        top_stocks = select_top_n_stocks(ranked_stocks, n_top)
        portfolio_value = data_slice.loc[:, top_stocks].mean(axis=1).cumsum().iloc[-1]
        portfolio_values.append(portfolio_value)
    
    # Calculate performance metrics
    cumulative_return = (np.prod(portfolio_values) - 1) * 100
    annualized_return = ((np.prod(portfolio_values) ** (252 / len(returns))) - 1) * 100
    sharpe_ratio = np.mean(portfolio_values) / np.std(portfolio_values) * np.sqrt(252)
    max_drawdown = np.min([(portfolio_values[i] / np.max(portfolio_values[:i+1]) - 1) * 100 
                           for i in range(len(portfolio_values))])
    
    backtest_results = {
        'Cumulative Return (%)': cumulative_return,
        'Annualized Return (%)': annualized_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown
    }
    
    return backtest_results

# Example usage
backtest_results = backtest_strategy(returns, period=252, n_top=3)
print("Backtest results:")
for key, value in backtest_results.items():
    print(f"{key}: {value}")

def optimize_strategy(returns, parameter_ranges):
    """
    Optimize the momentum strategy by sweeping through parameter ranges and evaluating performance.
    
    Parameters:
    - returns (DataFrame): DataFrame containing daily returns for each stock.
    - parameter_ranges (dict): Dictionary containing ranges for parameters to be optimized.
    
    Returns:
    - dict: Dictionary containing optimized parameter values and corresponding performance metrics.
    """
    best_params = {}
    best_performance = -float('inf')
    
    for period in parameter_ranges['period']:
        for n_top in parameter_ranges['n_top']:
            backtest_results = backtest_strategy(returns, period, n_top)
            performance_metric = backtest_results['Sharpe Ratio']  # You can choose any performance metric
            
            if performance_metric > best_performance:
                best_performance = performance_metric
                best_params['period'] = period
                best_params['n_top'] = n_top
    
    best_params['performance_metric'] = best_performance
    return best_params

# Example usage
parameter_ranges = {
    'period': [126, 252, 504],  # Different momentum calculation periods
    'n_top': [3, 5, 10]          # Different number of top stocks to select
}

optimized_params = optimize_strategy(returns, parameter_ranges)
print("Optimized parameters:")
print(optimized_params)


# Assume you have obtained the optimized parameters from the optimization process
optimized_params = {'period': 252, 'n_top': 5}  # Example optimized parameters

# Backtest the strategy using the optimized parameters
backtest_results_optimized = backtest_strategy(returns, optimized_params['period'], optimized_params['n_top'])

# Print the backtest results
print("Backtest results with optimized parameters:")
for key, value in backtest_results_optimized.items():
    print(f"{key}: {value}")
