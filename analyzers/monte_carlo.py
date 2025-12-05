"""Monte Carlo Simulation for Options Positions"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation results"""
    paths: int
    model: str
    pop: float  # Probability of Profit
    pot_lower: float  # Probability of Touch (lower)
    pot_upper: float  # Probability of Touch (upper)
    expected_pl: float
    median_pl: float
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    expected_shortfall_95: float
    optimal_exit_dte: int
    
    def to_dict(self) -> Dict:
        return {
            'paths': self.paths,
            'model': self.model,
            'pop': round(self.pop, 1),
            'pot_lower': round(self.pot_lower, 1),
            'pot_upper': round(self.pot_upper, 1),
            'expected_pl': round(self.expected_pl, 2),
            'median_pl': round(self.median_pl, 2),
            'var_95': round(self.var_95, 2),
            'var_99': round(self.var_99, 2),
            'expected_shortfall_95': round(self.expected_shortfall_95, 2),
            'optimal_exit_dte': self.optimal_exit_dte
        }


class MonteCarloSimulator:
    """Monte Carlo simulation for options strategies"""
    
    def __init__(self, n_paths: int = 50000, seed: int = None):
        self.n_paths = n_paths
        if seed:
            np.random.seed(seed)
    
    def simulate_gbm(
        self,
        S0: float,
        mu: float,
        sigma: float,
        T: float,
        n_steps: int = None
    ) -> np.ndarray:
        """
        Geometric Brownian Motion simulation
        
        Args:
            S0: Initial stock price
            mu: Expected return (annualized)
            sigma: Volatility (annualized)
            T: Time to expiration in years
            n_steps: Number of time steps (default: days to expiration)
        
        Returns:
            Array of simulated price paths (n_paths, n_steps+1)
        """
        if n_steps is None:
            n_steps = max(1, int(T * 252))  # Trading days
        
        dt = T / n_steps
        
        # Generate random shocks
        Z = np.random.standard_normal((self.n_paths, n_steps))
        
        # Calculate price paths
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        
        log_returns = drift + diffusion
        log_prices = np.zeros((self.n_paths, n_steps + 1))
        log_prices[:, 0] = np.log(S0)
        log_prices[:, 1:] = np.log(S0) + np.cumsum(log_returns, axis=1)
        
        return np.exp(log_prices)
    
    def simulate_heston(
        self,
        S0: float,
        v0: float,
        mu: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        T: float,
        n_steps: int = None
    ) -> np.ndarray:
        """
        Heston stochastic volatility model simulation
        
        Args:
            S0: Initial stock price
            v0: Initial variance
            mu: Expected return
            kappa: Mean reversion speed
            theta: Long-term variance
            xi: Volatility of volatility
            rho: Correlation between price and variance
            T: Time to expiration
            n_steps: Number of steps
        
        Returns:
            Array of simulated price paths
        """
        if n_steps is None:
            n_steps = max(1, int(T * 252))
        
        dt = T / n_steps
        
        # Correlation matrix for correlated Brownian motions
        Z1 = np.random.standard_normal((self.n_paths, n_steps))
        Z2 = np.random.standard_normal((self.n_paths, n_steps))
        W1 = Z1
        W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
        
        # Initialize arrays
        S = np.zeros((self.n_paths, n_steps + 1))
        v = np.zeros((self.n_paths, n_steps + 1))
        S[:, 0] = S0
        v[:, 0] = v0
        
        for t in range(n_steps):
            # Ensure variance stays positive
            v_pos = np.maximum(v[:, t], 0)
            sqrt_v = np.sqrt(v_pos)
            
            # Update variance (Euler discretization)
            v[:, t+1] = v[:, t] + kappa * (theta - v_pos) * dt + xi * sqrt_v * np.sqrt(dt) * W2[:, t]
            v[:, t+1] = np.maximum(v[:, t+1], 0)  # Reflection scheme
            
            # Update stock price
            S[:, t+1] = S[:, t] * np.exp(
                (mu - 0.5 * v_pos) * dt + sqrt_v * np.sqrt(dt) * W1[:, t]
            )
        
        return S
    
    def calculate_option_payoff(
        self,
        final_prices: np.ndarray,
        positions: List[Dict],
        entry_credit: float
    ) -> np.ndarray:
        """Calculate P&L for each simulated path"""
        
        n_paths = len(final_prices)
        payoffs = np.zeros(n_paths)
        
        for pos in positions:
            strike = pos['strike']
            qty = pos['qty']
            is_long = pos['position'] == 'long'
            is_call = pos['type'] == 'call'
            
            if is_call:
                intrinsic = np.maximum(final_prices - strike, 0)
            else:
                intrinsic = np.maximum(strike - final_prices, 0)
            
            # Long pays premium, short receives premium
            if is_long:
                payoffs -= intrinsic * qty * 100  # We lose when long options are ITM at expiry for credit spreads
            else:
                payoffs += intrinsic * qty * 100  # We lose when short options are ITM
        
        # Add entry credit (positive for credit spreads)
        # Note: For credit spreads, we received credit upfront
        # P&L = Credit received - Cost to close
        # If options expire worthless, payoffs = 0, so P&L = entry_credit
        # If options are ITM, payoffs < 0, reducing our profit
        
        # Actually, let's recalculate properly:
        # For a short option: we received premium, at expiry we owe intrinsic value
        # For a long option: we paid premium, at expiry we receive intrinsic value
        
        # Reset and recalculate
        payoffs = np.full(n_paths, entry_credit)  # Start with credit received
        
        for pos in positions:
            strike = pos['strike']
            qty = pos['qty']
            is_long = pos['position'] == 'long'
            is_call = pos['type'] == 'call'
            
            if is_call:
                intrinsic = np.maximum(final_prices - strike, 0)
            else:
                intrinsic = np.maximum(strike - final_prices, 0)
            
            if is_long:
                # Long option: we receive intrinsic at expiry
                payoffs += intrinsic * qty * 100
            else:
                # Short option: we owe intrinsic at expiry
                payoffs -= intrinsic * qty * 100
        
        return payoffs
    
    def run_simulation(
        self,
        current_price: float,
        positions: List[Dict],
        dte: int,
        volatility: float,
        entry_credit: float,
        breakeven_lower: float = None,
        breakeven_upper: float = None,
        risk_free_rate: float = 0.05,
        use_heston: bool = False
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation for options position
        
        Args:
            current_price: Current underlying price
            positions: List of option positions
            dte: Days to expiration
            volatility: Implied volatility (annualized)
            entry_credit: Net credit received
            breakeven_lower: Lower breakeven price
            breakeven_upper: Upper breakeven price
            risk_free_rate: Risk-free rate
            use_heston: Use Heston model instead of GBM
        
        Returns:
            MonteCarloResult with simulation statistics
        """
        T = dte / 365.0
        
        if use_heston:
            # Heston parameters (typical values)
            v0 = volatility ** 2
            kappa = 2.0  # Mean reversion speed
            theta = volatility ** 2  # Long-term variance
            xi = 0.3  # Vol of vol
            rho = -0.7  # Correlation (typically negative for equities)
            
            paths = self.simulate_heston(
                S0=current_price,
                v0=v0,
                mu=risk_free_rate,
                kappa=kappa,
                theta=theta,
                xi=xi,
                rho=rho,
                T=T
            )
            model = "Heston"
        else:
            paths = self.simulate_gbm(
                S0=current_price,
                mu=risk_free_rate,
                sigma=volatility,
                T=T
            )
            model = "GBM"
        
        # Get final prices
        final_prices = paths[:, -1]
        
        # Calculate P&L for each path
        payoffs = self.calculate_option_payoff(final_prices, positions, entry_credit)
        
        # Calculate statistics
        pop = np.mean(payoffs > 0) * 100
        
        # Probability of touch (price touching breakeven during the path)
        if breakeven_lower:
            pot_lower = np.mean(np.min(paths, axis=1) <= breakeven_lower) * 100
        else:
            pot_lower = 0
        
        if breakeven_upper:
            pot_upper = np.mean(np.max(paths, axis=1) >= breakeven_upper) * 100
        else:
            pot_upper = 0
        
        expected_pl = np.mean(payoffs)
        median_pl = np.median(payoffs)
        
        # Value at Risk (negative values represent losses)
        var_95 = np.percentile(payoffs, 5)
        var_99 = np.percentile(payoffs, 1)
        
        # Expected Shortfall (average loss when VaR is breached)
        losses_beyond_var = payoffs[payoffs <= var_95]
        expected_shortfall_95 = np.mean(losses_beyond_var) if len(losses_beyond_var) > 0 else var_95
        
        # Optimal exit DTE (simplified: when theta decay slows)
        # Generally, exit at 50% profit or around 21 DTE, whichever comes first
        optimal_exit_dte = min(max(dte - 21, 0), dte // 2)
        
        return MonteCarloResult(
            paths=self.n_paths,
            model=model,
            pop=pop,
            pot_lower=pot_lower,
            pot_upper=pot_upper,
            expected_pl=expected_pl,
            median_pl=median_pl,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall_95=expected_shortfall_95,
            optimal_exit_dte=optimal_exit_dte
        )
    
    def find_optimal_exit(
        self,
        current_price: float,
        positions: List[Dict],
        dte: int,
        volatility: float,
        entry_credit: float,
        current_value: float
    ) -> Tuple[int, float]:
        """
        Find optimal exit DTE by simulating different exit points
        
        Returns:
            Tuple of (optimal_dte, expected_pnl_at_exit)
        """
        best_dte = dte
        best_sharpe = float('-inf')
        best_expected = 0
        
        # Test different exit points
        exit_points = [max(1, dte - i * 5) for i in range(dte // 5 + 1)]
        
        for exit_dte in exit_points:
            if exit_dte <= 0:
                continue
                
            result = self.run_simulation(
                current_price=current_price,
                positions=positions,
                dte=exit_dte,
                volatility=volatility,
                entry_credit=entry_credit
            )
            
            # Simple Sharpe-like ratio
            if result.var_95 != 0:
                sharpe = result.expected_pl / abs(result.var_95)
            else:
                sharpe = result.expected_pl
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_dte = exit_dte
                best_expected = result.expected_pl
        
        return best_dte, best_expected

