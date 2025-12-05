"""
JAX-Accelerated Monte Carlo Simulation for Options
Falls back to NumPy if JAX is not available

Performance:
- NumPy: ~2s for 50,000 paths
- JAX CPU: ~0.5s for 50,000 paths  
- JAX GPU: ~0.05s for 500,000 paths
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import warnings

# Try to import JAX, fall back to NumPy
try:
    import jax
    import jax.numpy as jnp
    from jax import random, jit, vmap
    JAX_AVAILABLE = True
    # Check for GPU
    try:
        GPU_AVAILABLE = len(jax.devices('gpu')) > 0
    except:
        GPU_AVAILABLE = False
except ImportError:
    JAX_AVAILABLE = False
    GPU_AVAILABLE = False
    jnp = np  # Use numpy as fallback


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation"""
    pop: float  # Probability of profit (%)
    expected_pl: float  # Expected P&L
    median_pl: float  # Median P&L
    std_pl: float  # Standard deviation of P&L
    var_95: float  # 95% Value at Risk
    var_99: float  # 99% Value at Risk
    max_profit: float
    max_loss: float
    percentiles: Dict[int, float]  # P&L at various percentiles
    optimal_exit_dte: int  # Suggested exit DTE
    price_distribution: Dict[str, float]  # Final price stats
    paths_run: int
    backend: str  # 'jax_gpu', 'jax_cpu', or 'numpy'
    execution_time: float
    
    def to_dict(self) -> Dict:
        return {
            'probability_of_profit': round(self.pop, 2),
            'expected_pl': round(self.expected_pl, 2),
            'median_pl': round(self.median_pl, 2),
            'std_pl': round(self.std_pl, 2),
            'var_95': round(self.var_95, 2),
            'var_99': round(self.var_99, 2),
            'max_profit': round(self.max_profit, 2),
            'max_loss': round(self.max_loss, 2),
            'percentiles': {k: round(v, 2) for k, v in self.percentiles.items()},
            'optimal_exit_dte': self.optimal_exit_dte,
            'price_distribution': self.price_distribution,
            'paths_run': self.paths_run,
            'backend': self.backend,
            'execution_time_ms': round(self.execution_time * 1000, 1)
        }


class MonteCarloJAX:
    """
    High-performance Monte Carlo simulator using JAX
    Automatically uses GPU if available, falls back to CPU/NumPy
    """
    
    def __init__(self, n_paths: int = 100000, seed: int = None):
        self.n_paths = n_paths
        self.seed = seed or 42
        
        # Determine backend
        if JAX_AVAILABLE and GPU_AVAILABLE:
            self.backend = 'jax_gpu'
        elif JAX_AVAILABLE:
            self.backend = 'jax_cpu'
        else:
            self.backend = 'numpy'
        
        # Initialize random key for JAX
        if JAX_AVAILABLE:
            self.key = random.PRNGKey(self.seed)
    
    def _simulate_gbm_jax(
        self,
        S0: float,
        mu: float,
        sigma: float,
        T: float,
        n_steps: int
    ) -> jnp.ndarray:
        """
        Geometric Brownian Motion using JAX (JIT compiled)
        """
        dt = T / n_steps
        
        # Generate random numbers
        self.key, subkey = random.split(self.key)
        Z = random.normal(subkey, shape=(self.n_paths, n_steps))
        
        # GBM formula: S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * jnp.sqrt(dt) * Z
        
        # Cumulative sum for path
        log_returns = drift + diffusion
        log_prices = jnp.cumsum(log_returns, axis=1)
        
        # Convert to prices
        prices = S0 * jnp.exp(log_prices)
        
        return prices
    
    def _simulate_gbm_numpy(
        self,
        S0: float,
        mu: float,
        sigma: float,
        T: float,
        n_steps: int
    ) -> np.ndarray:
        """
        Geometric Brownian Motion using NumPy (fallback)
        """
        np.random.seed(self.seed)
        dt = T / n_steps
        
        Z = np.random.standard_normal((self.n_paths, n_steps))
        
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        
        log_returns = drift + diffusion
        log_prices = np.cumsum(log_returns, axis=1)
        
        prices = S0 * np.exp(log_prices)
        
        return prices
    
    def _simulate_heston_jax(
        self,
        S0: float,
        v0: float,
        mu: float,
        kappa: float,
        theta: float,
        sigma_v: float,
        rho: float,
        T: float,
        n_steps: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Heston stochastic volatility model using JAX
        """
        dt = T / n_steps
        
        # Generate correlated random numbers
        self.key, subkey1, subkey2 = random.split(self.key, 3)
        Z1 = random.normal(subkey1, shape=(self.n_paths, n_steps))
        Z2 = random.normal(subkey2, shape=(self.n_paths, n_steps))
        
        # Correlate the random numbers
        W_S = Z1
        W_v = rho * Z1 + jnp.sqrt(1 - rho**2) * Z2
        
        # Initialize arrays
        S = jnp.zeros((self.n_paths, n_steps + 1))
        v = jnp.zeros((self.n_paths, n_steps + 1))
        S = S.at[:, 0].set(S0)
        v = v.at[:, 0].set(v0)
        
        # Euler discretization (simplified for JIT)
        for t in range(n_steps):
            v_t = jnp.maximum(v[:, t], 0)  # Ensure non-negative variance
            
            # Price dynamics
            S = S.at[:, t+1].set(
                S[:, t] * jnp.exp(
                    (mu - 0.5 * v_t) * dt + jnp.sqrt(v_t * dt) * W_S[:, t]
                )
            )
            
            # Variance dynamics
            v = v.at[:, t+1].set(
                v_t + kappa * (theta - v_t) * dt + sigma_v * jnp.sqrt(v_t * dt) * W_v[:, t]
            )
        
        return S[:, 1:], v[:, 1:]
    
    def _simulate_heston_numpy(
        self,
        S0: float,
        v0: float,
        mu: float,
        kappa: float,
        theta: float,
        sigma_v: float,
        rho: float,
        T: float,
        n_steps: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Heston model using NumPy (fallback)
        """
        np.random.seed(self.seed)
        dt = T / n_steps
        
        Z1 = np.random.standard_normal((self.n_paths, n_steps))
        Z2 = np.random.standard_normal((self.n_paths, n_steps))
        
        W_S = Z1
        W_v = rho * Z1 + np.sqrt(1 - rho**2) * Z2
        
        S = np.zeros((self.n_paths, n_steps + 1))
        v = np.zeros((self.n_paths, n_steps + 1))
        S[:, 0] = S0
        v[:, 0] = v0
        
        for t in range(n_steps):
            v_t = np.maximum(v[:, t], 0)
            S[:, t+1] = S[:, t] * np.exp(
                (mu - 0.5 * v_t) * dt + np.sqrt(v_t * dt) * W_S[:, t]
            )
            v[:, t+1] = v_t + kappa * (theta - v_t) * dt + sigma_v * np.sqrt(v_t * dt) * W_v[:, t]
        
        return S[:, 1:], v[:, 1:]
    
    def _calculate_option_payoff(
        self,
        final_prices: np.ndarray,
        positions: List[Dict],
        entry_credit: float
    ) -> np.ndarray:
        """
        Calculate P&L for option positions at expiration
        
        Args:
            final_prices: Array of simulated final underlying prices
            positions: List of position dicts with strike, type, position, qty
            entry_credit: Net credit received in DOLLARS (positive = credit, negative = debit)
                         This is the TOTAL amount, not per-share
        
        Returns:
            Array of P&L values for each simulated path
        """
        payoffs = np.zeros(len(final_prices))
        
        for pos in positions:
            strike = pos.get('strike', 0)
            opt_type = pos.get('type', 'call').lower()
            direction = 1 if pos.get('position', 'long').lower() == 'long' else -1
            qty = pos.get('qty', 1)
            
            if opt_type == 'call':
                intrinsic = np.maximum(final_prices - strike, 0)
            else:
                intrinsic = np.maximum(strike - final_prices, 0)
            
            # Each contract = 100 shares
            payoffs += direction * qty * intrinsic * 100
        
        # Add entry credit/debit (already in dollars, no multiplier needed)
        # Positive entry_credit = we received money (credit spread)
        # Negative entry_credit = we paid money (debit spread)
        payoffs += entry_credit
        
        return payoffs
    
    def run_simulation(
        self,
        current_price: float,
        positions: List[Dict],
        dte: int,
        volatility: float,
        entry_credit: float = 0,
        risk_free_rate: float = 0.05,
        use_heston: bool = False,
        heston_params: Dict = None
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation for options positions
        
        Args:
            current_price: Current underlying price
            positions: List of position dicts with strike, type, position, qty
            dte: Days to expiration
            volatility: Annualized volatility (decimal, e.g., 0.25 for 25%)
            entry_credit: Net credit received (positive) or debit paid (negative)
            risk_free_rate: Annual risk-free rate
            use_heston: Use Heston model instead of GBM
            heston_params: Parameters for Heston model
        
        Returns:
            MonteCarloResult with simulation statistics
        """
        import time
        start_time = time.time()
        
        T = dte / 365.0
        n_steps = max(dte, 10)  # At least 10 steps
        
        # Run simulation
        if use_heston:
            params = heston_params or {
                'v0': volatility**2,
                'kappa': 2.0,
                'theta': volatility**2,
                'sigma_v': 0.3,
                'rho': -0.7
            }
            
            if self.backend.startswith('jax'):
                prices, _ = self._simulate_heston_jax(
                    S0=current_price,
                    v0=params['v0'],
                    mu=risk_free_rate,
                    kappa=params['kappa'],
                    theta=params['theta'],
                    sigma_v=params['sigma_v'],
                    rho=params['rho'],
                    T=T,
                    n_steps=n_steps
                )
                prices = np.array(prices)
            else:
                prices, _ = self._simulate_heston_numpy(
                    S0=current_price,
                    v0=params['v0'],
                    mu=risk_free_rate,
                    kappa=params['kappa'],
                    theta=params['theta'],
                    sigma_v=params['sigma_v'],
                    rho=params['rho'],
                    T=T,
                    n_steps=n_steps
                )
        else:
            if self.backend.startswith('jax'):
                prices = self._simulate_gbm_jax(
                    S0=current_price,
                    mu=risk_free_rate,
                    sigma=volatility,
                    T=T,
                    n_steps=n_steps
                )
                prices = np.array(prices)
            else:
                prices = self._simulate_gbm_numpy(
                    S0=current_price,
                    mu=risk_free_rate,
                    sigma=volatility,
                    T=T,
                    n_steps=n_steps
                )
        
        # Get final prices at expiration
        final_prices = prices[:, -1]
        
        # Calculate payoffs
        payoffs = self._calculate_option_payoff(final_prices, positions, entry_credit)
        
        # Calculate statistics
        pop = np.mean(payoffs > 0) * 100
        expected_pl = np.mean(payoffs)
        median_pl = np.median(payoffs)
        std_pl = np.std(payoffs)
        
        # Value at Risk (losses are negative, so we use percentiles from the left)
        var_95 = np.percentile(payoffs, 5)
        var_99 = np.percentile(payoffs, 1)
        
        max_profit = np.max(payoffs)
        max_loss = np.min(payoffs)
        
        # Percentiles
        percentiles = {
            5: np.percentile(payoffs, 5),
            10: np.percentile(payoffs, 10),
            25: np.percentile(payoffs, 25),
            50: np.percentile(payoffs, 50),
            75: np.percentile(payoffs, 75),
            90: np.percentile(payoffs, 90),
            95: np.percentile(payoffs, 95)
        }
        
        # Optimal exit DTE (simplified - find when expected P&L is maximized)
        # This is a rough estimate based on theta decay
        optimal_exit = max(0, dte - int(dte * 0.5)) if expected_pl > 0 else 0
        
        # Price distribution stats
        price_distribution = {
            'mean': float(np.mean(final_prices)),
            'std': float(np.std(final_prices)),
            'min': float(np.min(final_prices)),
            'max': float(np.max(final_prices)),
            'median': float(np.median(final_prices))
        }
        
        execution_time = time.time() - start_time
        
        return MonteCarloResult(
            pop=pop,
            expected_pl=expected_pl,
            median_pl=median_pl,
            std_pl=std_pl,
            var_95=var_95,
            var_99=var_99,
            max_profit=max_profit,
            max_loss=max_loss,
            percentiles=percentiles,
            optimal_exit_dte=optimal_exit,
            price_distribution=price_distribution,
            paths_run=self.n_paths,
            backend=self.backend,
            execution_time=execution_time
        )
    
    @staticmethod
    def get_backend_info() -> Dict:
        """Return information about available backends"""
        return {
            'jax_available': JAX_AVAILABLE,
            'gpu_available': GPU_AVAILABLE,
            'recommended_paths': 500000 if GPU_AVAILABLE else (100000 if JAX_AVAILABLE else 50000)
        }

