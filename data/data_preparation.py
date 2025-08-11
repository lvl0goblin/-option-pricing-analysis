import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os

class Nifty50Data:
    def __init__(self):
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        
    def download_historical_data(self, start_date="2020-01-01", end_date=None):
        """Download historical Nifty50 index data"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        ticker = "^NSEI"
        data = yf.download(ticker, start=start_date, end=end_date)
        data.to_csv(f"{self.data_dir}/nifty50_historical.csv")
        print(f"Historical data saved to {self.data_dir}/nifty50_historical.csv")
        return data
    
    def simulate_option_data(self, num_options=1000):
        """Generate simulated option data for testing"""
        np.random.seed(42)
        
        # Base parameters
        current_price = 18000  # Approximate Nifty50 level
        today = datetime.now()
        expiries = [today + timedelta(days=i) for i in [7, 14, 30, 60, 90]]
        
        # Generate random options
        data = []
        for _ in range(num_options):
            expiry = np.random.choice(expiries)
            days_to_expiry = (expiry - today).days
            strike = current_price * (0.9 + 0.2 * np.random.random())
            option_type = np.random.choice(['call', 'put'])
            iv = 0.15 + 0.1 * np.random.random()  # 15-25% IV
            price = current_price * 0.01 * np.random.random()  # 0-1% of index
            
            data.append({
                'date': today.strftime('%Y-%m-%d'),
                'expiry': expiry.strftime('%Y-%m-%d'),
                'strike': round(strike, 2),
                'type': option_type,
                'bid': round(price * 0.95, 2),
                'ask': round(price * 1.05, 2),
                'last_price': round(price, 2),
                'underlying': current_price,
                'implied_volatility': round(iv, 4),
                'days_to_expiry': days_to_expiry
            })
            
        df = pd.DataFrame(data)
        df.to_csv(f"{self.data_dir}/nifty50_options.csv", index=False)
        print(f"Option data saved to {self.data_dir}/nifty50_options.csv")
        return df
    
    def clean_data(self, df):
        """Clean and preprocess option data"""
        # Calculate mid price
        df['mid_price'] = (df['bid'] + df['ask']) / 2
        
        # Filter for reasonable options
        df = df[(df['bid'] > 0) & (df['ask'] > df['bid'])]
        
        # Calculate moneyness
        df['moneyness'] = df['strike'] / df['underlying']
        
        return df

if __name__ == "__main__":
    data_handler = Nifty50Data()
    hist_data = data_handler.download_historical_data()
    option_data = data_handler.simulate_option_data()
    cleaned_data = data_handler.clean_data(option_data)
