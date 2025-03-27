"""
Linear Marxian Model with fixed technology matrices.

This implements the basic linear model from Section 1 of Roemer's paper.
"""

import numpy as np
import warnings
from typing import List, Tuple, Optional
from scipy.optimize import linprog


class MarxianLinearModel:
    """
    Linear Marxian model with fixed technology matrices.
    
    Parameters:
        A (ndarray): Input-output matrix (n x n)
        L (ndarray): Direct labor input vector (n)
        b (ndarray): Workers' subsistence bundle (n)
        omega (ndarray): Initial endowments (n)
    """
    
    def __init__(self, A: np.ndarray, L: np.ndarray, b: np.ndarray, omega: np.ndarray):
        """Initialize the linear Marxian model."""
        self.A = np.array(A)
        self.L = np.array(L)
        self.b = np.array(b)
        self.omega = np.array(omega)
        self.n = len(L)
        
        # Validate model parameters
        self._validate_parameters()
        
        # Augmented input matrix (M = A + bL)
        self.M = self.A + np.outer(self.b, self.L)
    
    def _validate_parameters(self):
        """Validate model parameters."""
        # Check dimensions
        if self.A.shape != (self.n, self.n):
            raise ValueError(f"A should be a {self.n}x{self.n} matrix")
        if len(self.L) != self.n:
            raise ValueError(f"L should have length {self.n}")
        if len(self.b) != self.n:
            raise ValueError(f"b should have length {self.n}")
        if len(self.omega) != self.n:
            raise ValueError(f"omega should have length {self.n}")
        
        # Check if A is productive (i.e., there exists x > 0 s.t. x > Ax)
        try:
            I = np.eye(self.n)
            x = np.linalg.solve(I - self.A, np.ones(self.n))
            if not np.all(x > 0):
                warnings.warn("Matrix A may not be productive")
        except np.linalg.LinAlgError:
            warnings.warn("Matrix A may not be productive")
    
    def compute_labor_values(self) -> np.ndarray:
        """
        Compute labor values (Λ) by solving Λ = ΛA + L.
        
        Returns:
            ndarray: Vector of labor values
        """
        I = np.eye(self.n)
        try:
            labor_values = np.linalg.solve(I - self.A.T, self.L)
            return labor_values
        except np.linalg.LinAlgError:
            warnings.warn("Failed to compute labor values, matrix I-A' may be singular")
            return np.ones(self.n) * np.nan
    
    def compute_exploitation_rate(self) -> float:
        """
        Compute the rate of exploitation e = (1/Λb) - 1.
        
        Returns:
            float: Rate of exploitation
        """
        labor_values = self.compute_labor_values()
        subsistence_value = np.dot(labor_values, self.b)
        return (1 / subsistence_value) - 1
    
    def find_equal_profit_rate_prices(self) -> Tuple[np.ndarray, float]:
        """
        Find prices that generate equal profit rates in all sectors.
        
        Solves for p and π such that:
        p = (1 + π)(pA + L) and pb = 1
        
        Returns:
            Tuple[ndarray, float]: (prices, profit rate)
        """
        # Find eigenvalues and right eigenvectors of M.T
        eigenvalues, eigenvectors = np.linalg.eig(self.M.T)
        
        # Find the Frobenius eigenvalue (smallest positive real eigenvalue)
        valid_indices = np.where(np.isclose(eigenvalues.imag, 0) & (eigenvalues.real > 0))[0]
        if len(valid_indices) == 0:
            warnings.warn("No valid Frobenius eigenvalue found")
            return np.ones(self.n), 0.0
            
        idx = valid_indices[np.argmin(eigenvalues.real[valid_indices])]
        eigenvector = eigenvectors[:, idx].real
        
        # Normalize eigenvector so pb = 1
        if np.dot(eigenvector, self.b) == 0:
            warnings.warn("Cannot normalize prices: pb = 0")
            return np.ones(self.n), 0.0
            
        prices = eigenvector / np.dot(eigenvector, self.b)
        profit_rate = (1 / eigenvalues[idx].real) - 1
        
        return prices, profit_rate
    
    def get_profit_maximizing_activity(self, prices: np.ndarray, capitalist_omega: np.ndarray) -> np.ndarray:
        """
        Find profit-maximizing activity levels for a capitalist.
        
        Parameters:
            prices (ndarray): Price vector
            capitalist_omega (ndarray): Capitalist's endowment vector
            
        Returns:
            ndarray: Profit-maximizing activity levels
        """
        # Solve the linear program:
        # max p(I-A)x - Lx
        # s.t. (pA + L)x ≤ p·omega
        #      x ≥ 0
        
        c = -(prices @ (np.eye(self.n) - self.A) - self.L)
        
        A_ub = np.array([prices @ self.A + self.L])
        b_ub = np.array([prices @ capitalist_omega])
        
        bounds = [(0, None) for _ in range(self.n)]
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            return result.x
        else:
            warnings.warn(f"Failed to find profit-maximizing activity: {result.message}")
            return np.zeros(self.n)
    
    def is_reproducible(self, prices: np.ndarray, capitalist_omegas: List[np.ndarray]) -> bool:
        """
        Check if the economy is reproducible at given prices.
        
        Parameters:
            prices (ndarray): Price vector
            capitalist_omegas (List[ndarray]): List of capitalists' endowment vectors
            
        Returns:
            bool: True if economy is reproducible, False otherwise
        """
        # Get aggregate profit-maximizing activity
        x = np.zeros(self.n)
        for omega_v in capitalist_omegas:
            x += self.get_profit_maximizing_activity(prices, omega_v)
        
        # Check reproducibility: x ≥ Ax + (Lx)b
        total_labor = np.sum(self.L * x)
        return np.all(x >= (self.A @ x + total_labor * self.b))
    
    def compute_cone_C_star(self) -> np.ndarray:
        """
        Compute a representation of the cone C* of reproducible endowments.
        
        Returns:
            ndarray: Matrix whose columns span the cone C*
        """
        prices, profit_rate = self.find_equal_profit_rate_prices()
        
        # Find balanced growth vector x* such that x* = (1 + π)Mx*
        # This is the right eigenvector of M with eigenvalue 1/(1+π)
        eigenvalues, eigenvectors = np.linalg.eig(self.M)
        target_eigenvalue = 1 / (1 + profit_rate)
        
        idx = np.argmin(np.abs(eigenvalues - target_eigenvalue))
        x_star = eigenvectors[:, idx].real
        
        # Normalize x_star to be positive
        if np.any(x_star < 0):
            x_star = -x_star
            
        # Scale x_star to be strictly positive
        x_star = x_star + 1e-6
        
        # The cone C* is generated by Mx*
        # We return a basis for the cone
        return self.M @ x_star.reshape(-1, 1)
