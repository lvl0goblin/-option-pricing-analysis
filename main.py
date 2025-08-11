from data.data_preparation import Nifty50Data
from models.black_scholes import BlackScholes
from models.monte_carlo import MonteCarloPricer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

class OptionBacktester:
    """
    Framework for backtesting option pricing models against market data
    """
    def __init__(self, risk_free_rate=0.05):
        self.bs = BlackScholes(risk_free_rate)
        self.mc = MonteCarloPricer(risk_free_rate)
        self.results = None
        
    def load_data(self, filepath):
        """Load option data from CSV file"""
        df = pd.read_csv(filepath)
        df['expiry'] = pd.to_datetime(df['expiry'])
        df['date'] = pd.to_datetime(df['date'])
        df['days_to_expiry'] = (df['expiry'] - df['date']).dt.days
        df['T'] = df['days_to_expiry'] / 365
        
        # Calculate mid_price if not already present
        if 'mid_price' not in df.columns and 'bid' in df.columns and 'ask' in df.columns:
            df['mid_price'] = (df['bid'] + df['ask']) / 2
        elif 'last_price' in df.columns:
            df['mid_price'] = df['last_price']
        else:
            raise ValueError("Data must contain either 'bid/ask' or 'last_price' columns")
            
        return df
    
    def run_backtest(self, data_file, sample_size=None, use_mc=False):
        """
        Run backtest comparing model prices to market prices
        
        Parameters:
        data_file (str): path to option data CSV
        sample_size (int): number of options to test (None for all)
        use_mc (bool): whether to use Monte Carlo for pricing
        
        Returns:
        DataFrame: containing backtest results
        """
        df = self.load_data(data_file)
        
        if sample_size is not None:
            df = df.sample(min(sample_size, len(df)), random_state=42)
        
        results = []
        for _, row in df.iterrows():
            S = row['underlying']
            K = row['strike']
            T = row['T']
            market_price = row['mid_price']
            option_type = row['type']
            sigma = row['implied_volatility']
            
            # Calculate model prices
            bs_price = self.bs.calculate(S, K, T, sigma, option_type)['price']
            
            if use_mc:
                mc_result = self.mc.price_option(S, K, T, sigma, option_type, 
                                                num_simulations=10000, num_steps=100)
                model_price = mc_result['price']
                model_type = 'Monte Carlo'
            else:
                model_price = bs_price
                model_type = 'Black-Scholes'
            
            # Calculate errors
            error = model_price - market_price
            pct_error = 100 * error / market_price if market_price != 0 else 0
            
            results.append({
                'date': row['date'],
                'expiry': row['expiry'],
                'strike': K,
                'type': option_type,
                'moneyness': K / S,
                'days_to_expiry': row['days_to_expiry'],
                'market_price': market_price,
                'model_price': model_price,
                'model_type': model_type,
                'error': error,
                'pct_error': pct_error,
                'implied_volatility': sigma
            })
            
        self.results = pd.DataFrame(results)
        return self.results
    
    def calculate_metrics(self):
        """Calculate performance metrics for the backtest"""
        if self.results is None:
            raise ValueError("No backtest results available. Run run_backtest() first.")
            
        metrics = {
            'MAE': mean_absolute_error(self.results['market_price'], self.results['model_price']),
            'RMSE': np.sqrt(mean_squared_error(self.results['market_price'], self.results['model_price'])),
            'Mean Error': np.mean(self.results['error']),
            'Mean Abs Pct Error': np.mean(np.abs(self.results['pct_error'])),
            'Median Error': np.median(self.results['error']),
            'Std Dev of Errors': np.std(self.results['error']),
            'Correlation': np.corrcoef(self.results['market_price'], self.results['model_price'])[0, 1]
        }
        
        return pd.Series(metrics).to_frame('Value')
    
    def plot_results(self, save_path=None):
        """Visualize backtest results"""
        if self.results is None:
            raise ValueError("No backtest results available. Run run_backtest() first.")
            
        plt.figure(figsize=(18, 12))
        
        # Scatter plot of market vs model prices
        plt.subplot(2, 2, 1)
        sns.scatterplot(x='market_price', y='model_price', hue='type', data=self.results)
        plt.plot([0, self.results['market_price'].max()], [0, self.results['market_price'].max()], 'k--')
        plt.xlabel('Market Price')
        plt.ylabel('Model Price')
        plt.title('Market vs Model Prices')
        
        # Error distribution
        plt.subplot(2, 2, 2)
        sns.histplot(self.results['error'], kde=True)
        plt.axvline(0, color='k', linestyle='--')
        plt.xlabel('Pricing Error')
        plt.title('Error Distribution')
        
        # Error by moneyness
        plt.subplot(2, 2, 3)
        sns.scatterplot(x='moneyness', y='error', hue='type', data=self.results)
        plt.axhline(0, color='k', linestyle='--')
        plt.xlabel('Moneyness (K/S)')
        plt.ylabel('Pricing Error')
        plt.title('Error by Moneyness')
        
        # Error by days to expiry
        plt.subplot(2, 2, 4)
        sns.scatterplot(x='days_to_expiry', y='error', hue='type', data=self.results)
        plt.axhline(0, color='k', linestyle='--')
        plt.xlabel('Days to Expiry')
        plt.ylabel('Pricing Error')
        plt.title('Error by Time to Expiry')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def analyze_by_category(self):
        """Analyze performance by different option categories"""
        if self.results is None:
            raise ValueError("No backtest results available. Run run_backtest() first.")
            
        analysis = {}
        
        # By option type
        for opt_type in ['call', 'put']:
            subset = self.results[self.results['type'] == opt_type]
            analysis[f'{opt_type.title()} Options'] = {
                'MAE': mean_absolute_error(subset['market_price'], subset['model_price']),
                'RMSE': np.sqrt(mean_squared_error(subset['market_price'], subset['model_price'])),
                'Mean Error': np.mean(subset['error']),
                'Count': len(subset)
            }
        
        # By moneyness categories
        bins = [0, 0.9, 1.0, 1.1, np.inf]
        labels = ['Deep OTM', 'OTM', 'ITM', 'Deep ITM']
        self.results['moneyness_category'] = pd.cut(
            self.results['moneyness'], bins=bins, labels=labels)
        
        for category in labels:
            subset = self.results[self.results['moneyness_category'] == category]
            if len(subset) > 0:
                analysis[f'{category} Options'] = {
                    'MAE': mean_absolute_error(subset['market_price'], subset['model_price']),
                    'RMSE': np.sqrt(mean_squared_error(subset['market_price'], subset['model_price'])),
                    'Mean Error': np.mean(subset['error']),
                    'Count': len(subset)
                }
        
        # By time to expiry
        bins = [0, 7, 30, 90, np.inf]
        labels = ['Weekly', 'Monthly', 'Quarterly', 'Long-term']
        self.results['expiry_category'] = pd.cut(
            self.results['days_to_expiry'], bins=bins, labels=labels)
        
        for category in labels:
            subset = self.results[self.results['expiry_category'] == category]
            if len(subset) > 0:
                analysis[f'{category} Options'] = {
                    'MAE': mean_absolute_error(subset['market_price'], subset['model_price']),
                    'RMSE': np.sqrt(mean_squared_error(subset['market_price'], subset['model_price'])),
                    'Mean Error': np.mean(subset['error']),
                    'Count': len(subset)
                }
        
        return pd.DataFrame(analysis).T

if __name__ == "__main__":
    # Step 1: Generate sample data
    print("Generating sample data...")
    data_handler = Nifty50Data()
    hist_data = data_handler.download_historical_data()
    option_data = data_handler.simulate_option_data(num_options=1000)
    cleaned_data = data_handler.clean_data(option_data)
    
    # Step 2: Run Black-Scholes backtest
    print("\nRunning Black-Scholes backtest...")
    backtester = OptionBacktester()
    bs_results = backtester.run_backtest('data/nifty50_options.csv', sample_size=500)
    
    print("\nBlack-Scholes Performance Metrics:")
    print(backtester.calculate_metrics())
    
    # Step 3: Run Monte Carlo backtest (smaller sample due to computation time)
    print("\nRunning Monte Carlo backtest...")
    mc_results = backtester.run_backtest('data/nifty50_options.csv', sample_size=100, use_mc=True)
    
    print("\nMonte Carlo Performance Metrics:")
    print(backtester.calculate_metrics())
    
    # Step 4: Save plots and analysis
    print("\nGenerating plots...")
    backtester.plot_results(save_path='analysis/backtest_results.png')
    
    print("\nCategory Analysis:")
    category_results = backtester.analyze_by_category()
    print(category_results)
    
    print("\nAnalysis complete! Check the 'analysis/' folder for results.")