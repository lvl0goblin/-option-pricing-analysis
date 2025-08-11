import numpy as np
from math import exp, sqrt
from scipy.stats import norm

class MonteCarloPricer:
    """
    Option pricing using Monte Carlo simulation with variance reduction techniques
    """
    def __init__(self, risk_free_rate=0.05):
        self.r = risk_free_rate
        
    def simulate_paths(self, S0, sigma, T, num_steps, num_simulations, antithetic=True, seed=None):
        """
        Simulate geometric Brownian motion paths
        
        Parameters:
        S0 (float): initial price
        sigma (float): volatility
        T (float): time to maturity in years
        num_steps (int): number of time steps
        num_simulations (int): number of paths to simulate
        antithetic (bool): whether to use antithetic variates
        seed (int): random seed for reproducibility
        
        Returns:
        ndarray: simulated paths (num_simulations x num_steps+1)
        """
        if seed is not None:
            np.random.seed(seed)
            
        dt = T / num_steps
        if antithetic:
            # Generate half the paths and use antithetic variates
            num_simulations_half = num_simulations // 2
            paths = np.zeros((num_simulations_half * 2, num_steps + 1))
            paths[:, 0] = S0
            
            # Random shocks
            shocks = np.random.normal(0, 1, (num_simulations_half, num_steps))
            
            # Generate paths
            for t in range(1, num_steps + 1):
                # Standard paths
                paths[:num_simulations_half, t] = (
                    paths[:num_simulations_half, t-1] * 
                    np.exp((self.r - 0.5 * sigma**2) * dt + sigma * sqrt(dt) * shocks[:, t-1])
                )
                # Antithetic paths
                paths[num_simulations_half:, t] = (
                    paths[num_simulations_half:, t-1] * 
                    np.exp((self.r - 0.5 * sigma**2) * dt - sigma * sqrt(dt) * shocks[:, t-1])
                )
        else:
            # Generate all paths independently
            paths = np.zeros((num_simulations, num_steps + 1))
            paths[:, 0] = S0
            shocks = np.random.normal(0, 1, (num_simulations, num_steps))
            
            for t in range(1, num_steps + 1):
                paths[:, t] = (
                    paths[:, t-1] * 
                    np.exp((self.r - 0.5 * sigma**2) * dt + sigma * sqrt(dt) * shocks[:, t-1])
                )
                
        return paths
    
    def price_option(self, S0, K, T, sigma, option_type='call', 
                     num_simulations=10000, num_steps=252, control_variate=True):
        """
        Price option using Monte Carlo simulation
        
        Parameters:
        S0, K, T, sigma, option_type: as in Black-Scholes
        num_simulations (int): number of simulations
        num_steps (int): number of time steps
        control_variate (bool): whether to use control variate
        
        Returns:
        dict: containing price and standard error
        """
        # Simulate paths
        paths = self.simulate_paths(S0, sigma, T, num_steps, num_simulations, antithetic=True)
        ST = paths[:, -1]
        
        # Calculate payoffs
        if option_type == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
            
        # Discount payoffs
        discounted_payoffs = np.exp(-self.r * T) * payoffs
        
        # Basic Monte Carlo estimate
        mc_price = np.mean(discounted_payoffs)
        mc_std = np.std(discounted_payoffs) / sqrt(num_simulations)
        
        if control_variate:
            # Control variate technique using Black-Scholes as control
            d1 = (np.log(S0 / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
            d2 = d1 - sigma * sqrt(T)
            
            if option_type == 'call':
                bs_price = S0 * norm.cdf(d1) - K * exp(-self.r * T) * norm.cdf(d2)
            else:
                bs_price = K * exp(-self.r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
            
            # Calculate optimal coefficient
            cov = np.cov(np.stack((discounted_payoffs, payoffs)))[0, 1]
            var = np.var(payoffs)
            theta = cov / var
            
            # Control variate estimate
            cv_price = mc_price - theta * (np.mean(payoffs) - bs_price)
            
            # Estimate standard error (simplified)
            cv_std = mc_std * sqrt(1 - (cov**2)/(var * np.var(discounted_payoffs)))
            
            return {
                'price': cv_price,
                'std_error': cv_std,
                'method': 'Monte Carlo with Antithetic Variates and Control Variate',
                'bs_price': bs_price,
                'basic_mc_price': mc_price,
                'basic_mc_std': mc_std
            }
        else:
            return {
                'price': mc_price,
                'std_error': mc_std,
                'method': 'Basic Monte Carlo',
                'bs_price': None
            }

if __name__ == "__main__":
    mc = MonteCarloPricer()
    
    # Parameters
    S0 = 18000
    K = 18200
    T = 30/365
    sigma = 0.20
    option_type = 'call'
    
    # Price option
    result = mc.price_option(S0, K, T, sigma, option_type, 
                            num_simulations=100000, num_steps=100)
    
    print(f"\n{result['method']}")
    print(f"Monte Carlo Price: {result['price']:.2f}")
    print(f"Standard Error: {result['std_error']:.4f}")
    print(f"Black-Scholes Price: {result['bs_price']:.2f}")
    print(f"Difference: {result['price'] - result['bs_price']:.2f}")
    
    # Convergence study
    print("\nConvergence Study:")
    for sims in [1000, 10000, 50000, 100000]:
        res = mc.price_option(S0, K, T, sigma, option_type, 
                            num_simulations=sims, num_steps=100)
        print(f"Simulations: {sims:6d} | Price: {res['price']:.2f} | "
              f"Std Error: {res['std_error']:.4f} | "
              f"Diff to BS: {res['price'] - res['bs_price']:.2f}")
