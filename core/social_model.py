"""
Social Determination Model with workers' consumption determined by social factors.

This implements the mechanism described in Section 4 of Roemer's paper.
"""

import numpy as np
import warnings
from typing import List, Tuple, Callable, Dict, Any
from scipy.optimize import minimize


class SocialDeterminationModel:
    """
    Model with social determination of workers' consumption.
    
    Parameters:
        production_functions (List[Callable]): List of production functions
        labor_functions (List[Callable]): List of labor requirement functions
        omega (ndarray): Initial endowments
        consumption_function (Callable): Function that determines worker consumption
        class_power (float): Parameter representing relative power of workers vs capitalists
    """
    
    def __init__(self, 
                 production_functions: List[Callable], 
                 labor_functions: List[Callable],
                 omega: np.ndarray,
                 consumption_function: Callable,
                 class_power: float = 0.5):
        """Initialize the social determination model."""
        self.production_functions = production_functions
        self.labor_functions = labor_functions
        self.omega = np.array(omega)
        self.n = len(omega)
        self.consumption_function = consumption_function
        self.class_power = class_power
        
        # Calculate initial subsistence bundle
        self.b = self._determine_subsistence()
    
    def _determine_subsistence(self, activity_levels=None):
        """
        Determine subsistence bundle based on production and class power.
        
        Parameters:
            activity_levels (ndarray): Activity levels for production
            
        Returns:
            ndarray: Determined subsistence bundle
        """
        if activity_levels is None:
            # Default to even activity across sectors
            activity_levels = np.ones(self.n) / self.n
        
        # Calculate reference output and labor
        total_output = np.zeros(self.n)
        total_labor = 0
        
        for i, prod_func in enumerate(self.production_functions):
            # Use activity level as input for each production function
            output = prod_func(activity_levels)
            labor = self.labor_functions[i](activity_levels)
            
            total_output += output
            total_labor += labor
        
        # Apply consumption function to determine subsistence
        subsistence = self.consumption_function(
            total_output=total_output,
            total_labor=total_labor,
            class_power=self.class_power,
            activity_levels=activity_levels
        )
        
        return subsistence
    
    def update_subsistence_bundle(self, activity_levels=None):
        """
        Update the subsistence bundle based on current production.
        
        Parameters:
            activity_levels (ndarray): Activity levels for production
        """
        self.b = self._determine_subsistence(activity_levels)
    
    def compute_labor_value(self, commodity_bundle: np.ndarray) -> float:
        """
        Compute the labor value of a commodity bundle.
        
        Parameters:
            commodity_bundle (ndarray): Bundle of commodities to evaluate
        
        Returns:
            float: Labor value of the commodity bundle
        """
        def objective(x):
            # x represents activity levels for each production process
            total_labor = 0
            for i, labor_func in enumerate(self.labor_functions):
                # Extract inputs for this production function
                inputs = x[i*self.n:(i+1)*self.n]
                total_labor += labor_func(inputs)
            
            return total_labor
        
        def constraint(x):
            # Calculate net output for the given activity levels
            net_output = np.zeros(self.n)
            
            for i, prod_func in enumerate(self.production_functions):
                # Extract inputs for this production function
                inputs = x[i*self.n:(i+1)*self.n]
                
                # Add outputs
                outputs = prod_func(inputs)
                net_output += outputs
                
                # Subtract inputs
                net_output -= inputs
            
            # Constraint: net_output ≥ commodity_bundle
            return net_output - commodity_bundle
        
        # Initial guess: spread activity evenly across processes
        x0 = np.ones(len(self.production_functions) * self.n) / (len(self.production_functions) * self.n)
        
        # Constraints: net_output[i] ≥ commodity_bundle[i] for each i
        constraints = [{'type': 'ineq', 'fun': lambda x: constraint(x)[i]} for i in range(self.n)]
        
        # Non-negativity constraint
        bounds = [(0, None) for _ in range(len(self.production_functions) * self.n)]
        
        # Solve minimization problem
        result = minimize(objective, x0, bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.fun
        else:
            warnings.warn(f"Failed to compute labor value: {result.message}")
            return np.nan
    
    def compute_exploitation_rate(self) -> float:
        """
        Compute the rate of exploitation.
        
        Returns:
            float: Rate of exploitation
        """
        subsistence_labor_value = self.compute_labor_value(self.b)
        
        # For the total labor, use a reference production level
        reference_input = np.ones(self.n) * np.mean(self.omega) * 0.1
        total_labor = sum(labor_func(reference_input) for labor_func in self.labor_functions)
        
        return (total_labor / subsistence_labor_value) - 1
    
    def find_equilibrium_prices(self) -> Tuple[np.ndarray, float]:
        """
        Find equilibrium prices that approximate equal profit rates.
        
        Returns:
            Tuple[ndarray, float]: (prices, average profit rate)
        """
        # Define test inputs for different sectors
        test_inputs = []
        for i in range(self.n):
            input_i = np.zeros(self.n)
            input_i[i] = 1.0
            test_inputs.append(input_i)
        
        def profit_rate_variance(prices):
            # Normalize prices so pb = 1
            prices = prices / np.dot(prices, self.b)
            
            profit_rates = []
            for input_vector in test_inputs:
                # Calculate costs
                labor = sum(labor_func(input_vector) for labor_func in self.labor_functions)
                input_cost = np.dot(prices, input_vector) + labor
                
                # Calculate outputs
                outputs = sum(prod_func(input_vector) for prod_func in self.production_functions)
                output_value = np.dot(prices, outputs)
                
                # Calculate profit rate
                if input_cost > 0:
                    profit_rate = (output_value - input_cost) / input_cost
                    profit_rates.append(profit_rate)
            
            # Calculate variance of profit rates
            if profit_rates:
                return np.var(profit_rates)
            else:
                return float('inf')
        
        # Initial guess: equal prices
        p0 = np.ones(self.n) / self.n
        
        # Constraint: prices should be non-negative
        bounds = [(0, None) for _ in range(self.n)]
        
        # Solve minimization problem
        result = minimize(profit_rate_variance, p0, bounds=bounds)
        
        if result.success:
            # Normalize prices
            prices = result.x / np.dot(result.x, self.b)
            
            # Calculate average profit rate
            avg_profit_rate = 0.0
            count = 0
            
            for input_vector in test_inputs:
                labor = sum(labor_func(input_vector) for labor_func in self.labor_functions)
                input_cost = np.dot(prices, input_vector) + labor
                
                outputs = sum(prod_func(input_vector) for prod_func in self.production_functions)
                output_value = np.dot(prices, outputs)
                
                if input_cost > 0:
                    profit_rate = (output_value - input_cost) / input_cost
                    avg_profit_rate += profit_rate
                    count += 1
            
            if count > 0:
                avg_profit_rate /= count
            
            return prices, avg_profit_rate
        else:
            warnings.warn(f"Failed to find equilibrium prices: {result.message}")
            return np.ones(self.n) / np.sum(self.b), 0.0
    
    def simulate_class_struggle_dynamics(self, periods: int = 10, class_power_trajectory: List[float] = None) -> Dict[str, Any]:
        """
        Simulate economy with changing class power relations.
        
        Parameters:
            periods (int): Number of periods to simulate
            class_power_trajectory (List[float]): Class power parameters for each period
            
        Returns:
            Dict[str, Any]: Simulation results
        """
        if class_power_trajectory is None:
            # Default: constant class power
            class_power_trajectory = [self.class_power] * periods
        
        # Initialize time series
        time_series = {
            'periods': list(range(periods)),
            'exploitation_rate': [],
            'profit_rate': [],
            'subsistence': [],
            'endowments': [],
            'class_power': class_power_trajectory,
            'reproducible': []
        }
        
        # Current endowment and activity levels
        curr_omega = self.omega.copy()
        activity_levels = np.ones(self.n) / self.n
        
        for t in range(periods):
            # Update class power
            self.class_power = class_power_trajectory[t]
            
            # Update subsistence bundle
            self.update_subsistence_bundle(activity_levels)
            time_series['subsistence'].append(self.b.copy())
            
            # Calculate prices
            prices, profit_rate = self.find_equilibrium_prices()
            time_series['profit_rate'].append(profit_rate)
            
            # Calculate exploitation rate
            exploitation_rate = self.compute_exploitation_rate()
            time_series['exploitation_rate'].append(exploitation_rate)
            
            # Get profit-maximizing activities
            # We use a simplified approach for simulation
            total_output = np.zeros(self.n)
            total_labor = 0
            
            for i, prod_func in enumerate(self.production_functions):
                output = prod_func(activity_levels)
                labor = self.labor_functions[i](activity_levels)
                
                total_output += output
                total_labor += labor
            
            # Calculate worker consumption
            worker_consumption = total_labor * self.b
            
            # Check reproducibility
            reproducible = np.all(total_output >= worker_consumption + activity_levels)
            time_series['reproducible'].append(reproducible)
            
            # Update endowments
            if reproducible:
                curr_omega = total_output - worker_consumption
            else:
                # If not reproducible, adjust to maximum possible consumption
                scale_factor = min([
                    (total_output[i] - activity_levels[i]) / worker_consumption[i] 
                    for i in range(self.n) if worker_consumption[i] > 0
                ])
                curr_omega = total_output - worker_consumption * scale_factor
            
            time_series['endowments'].append(curr_omega.copy())
            
            # Update activity levels for next period (simple adjustment rule)
            for i in range(self.n):
                # Increase activity in profitable sectors
                output_i = total_output[i] / self.n  # Simplified allocation
                input_cost_i = activity_levels[i] * prices[i] + total_labor / self.n
                
                if input_cost_i > 0:
                    profit_i = output_i * prices[i] - input_cost_i
                    profit_rate_i = profit_i / input_cost_i
                    
                    # Adjustment rule: increase activity in profitable sectors
                    activity_levels[i] *= (1 + 0.1 * profit_rate_i)
            
            # Normalize activity levels
            activity_levels = activity_levels / np.sum(activity_levels)
        
        return time_series
