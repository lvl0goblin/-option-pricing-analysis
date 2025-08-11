import numpy as np
from scipy.stats import norm
from math import log, sqrt, exp

class BlackScholes:
    """
    Black-Scholes option pricing model implementation with Greeks calculation
    """
    def __init__(self, risk_free_rate=0.05):
        self.r = risk_free_rate  # risk-free interest rate
        
    def calculate(self, S, K, T, sigma, option_type='call'):
        """
        Calculate option price using Black-Scholes formula
        
        Parameters:
        S (float): underlying asset price
        K (float): strike price
        T (float): time to maturity in years
        sigma (float): volatility
        option_type (str): 'call' or 'put'
        
        Returns:
        dict: containing price and Greeks
        """
        d1 = (log(S / K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * exp(-self.r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
        else:
            price = K * exp(-self.r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = -norm.cdf(-d1)
        
        # Calculate Greeks
        gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
        vega = S * norm.pdf(d1) * sqrt(T)
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * sqrt(T)) - 
                self.r * K * exp(-self.r * T) * norm.cdf(d2)) if option_type == 'call' else (
                -(S * norm.pdf(d1) * sigma) / (2 * sqrt(T)) + 
                self.r * K * exp(-self.r * T) * norm.cdf(-d2))
        rho = K * T * exp(-self.r * T) * norm.cdf(d2) if option_type == 'call' else (
              -K * T * exp(-self.r * T) * norm.cdf(-d2))
        
        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta / 365,  # per day
            'rho': rho
        }
    
    def implied_volatility(self, S, K, T, market_price, option_type='call', max_iter=100, precision=1e-6):
        """
        Calculate implied volatility using Newton-Raphson method
        
        Parameters:
        S, K, T: as above
        market_price (float): observed option price
        option_type (str): 'call' or 'put'
        max_iter (int): maximum iterations
        precision (float): desired accuracy
        
        Returns:
        float: implied volatility
        """
        sigma = 0.5  # initial guess
        
        for _ in range(max_iter):
            price = self.calculate(S, K, T, sigma, option_type)['price']
            vega = self.calculate(S, K, T, sigma, option_type)['vega']
            
            diff = market_price - price
            
            if abs(diff) < precision:
                return sigma
            
            sigma = sigma + diff / vega  # Newton-Raphson update
            
        return sigma  # return best estimate if not converged

if __name__ == "__main__":
    bs = BlackScholes()
    
    # Example usage
    S = 18000  # Underlying price
    K = 18200  # Strike price
    T = 30/365  # 30 days to expiry
    sigma = 0.20  # 20% volatility
    
    call = bs.calculate(S, K, T, sigma, 'call')
    put = bs.calculate(S, K, T, sigma, 'put')
    
    print("Call Option:")
    print(f"Price: {call['price']:.2f}")
    print(f"Delta: {call['delta']:.4f}")
    print(f"Gamma: {call['gamma']:.6f}")
    print(f"Vega: {call['vega']:.4f}")
    print(f"Theta: {call['theta']:.4f} per day")
    print(f"Rho: {call['rho']:.4f}")
    
    print("\nPut Option:")
    print(f"Price: {put['price']:.2f}")
    print(f"Delta: {put['delta']:.4f}")
    print(f"Gamma: {put['gamma']:.6f}")
    print(f"Vega: {put['vega']:.4f}")
    print(f"Theta: {put['theta']:.4f} per day")
    print(f"Rho: {put['rho']:.4f}")
    
    # Implied volatility example
    market_price = 350.0
    iv = bs.implied_volatility(S, K, T, market_price, 'call')
    print(f"\nImplied Volatility: {iv:.4f} or {iv*100:.2f}%")
