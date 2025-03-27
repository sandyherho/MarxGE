"""
Fundamental Marxian Theorem analysis.

This module implements tools for testing the Fundamental Marxian Theorem,
which states that positive exploitation is necessary and sufficient for
positive profits under Assumption A7 (Independence of Production).
"""

import numpy as np
import copy
from typing import Dict, List, Any, Union, Optional


class FundamentalMarxianTheorem:
    """
    Tools for testing the Fundamental Marxian Theorem.
    """
    
    @staticmethod
    def test_theorem(model, parameter_name=None, values=None):
        """
        Test the Fundamental Marxian Theorem.
        
        Parameters:
            model: Economic model (MarxianLinearModel, ConvexProductionModel, or SocialDeterminationModel)
            parameter_name (str): Name of parameter to vary
            values (List): List of parameter values to test
            
        Returns:
            Dict[str, Any]: Test results
        """
        if parameter_name is None:
            # Just test the base model
            exploitation_rate = model.compute_exploitation_rate()
            
            # Compute profit rate
            if hasattr(model, 'find_equal_profit_rate_prices'):
                _, profit_rate = model.find_equal_profit_rate_prices()
            elif hasattr(model, 'find_equilibrium_prices'):
                _, profit_rate = model.find_equilibrium_prices()
            else:
                profit_rate = 0.0
            
            return {
                'exploitation_rate': exploitation_rate,
                'profit_rate': profit_rate,
                'fmt_holds': (exploitation_rate > 0) == (profit_rate > 0)
            }
        
        # Vary parameter and test FMT
        results = {
            'parameter_values': values,
            'exploitation_rates': [],
            'profit_rates': [],
            'fmt_holds': []
        }
        
        for value in values:
            # Create modified model based on parameter to vary
            modified_model = copy.deepcopy(model)
            
            if parameter_name == 'subsistence':
                if hasattr(modified_model, 'b'):
                    modified_model.b = model.b * value
                    
                    # Update M matrix for linear model
                    if hasattr(modified_model, 'M'):
                        modified_model.M = modified_model.A + np.outer(modified_model.b, modified_model.L)
            
            elif parameter_name == 'productivity':
                if hasattr(modified_model, 'A'):
                    # For linear model, decrease input requirements
                    modified_model.A = model.A / value
                    
                    # Update M matrix
                    modified_model.M = modified_model.A + np.outer(modified_model.b, modified_model.L)
                
                elif hasattr(modified_model, 'production_functions'):
                    # For convex model, scale production functions
                    orig_production_functions = model.production_functions
                    modified_model.production_functions = [
                        lambda x, orig_func=func, v=value: orig_func(x) * v
                        for func in orig_production_functions
                    ]
            
            # Test FMT
            fmt_result = FundamentalMarxianTheorem.test_theorem(modified_model)
            
            results['exploitation_rates'].append(fmt_result['exploitation_rate'])
            results['profit_rates'].append(fmt_result['profit_rate'])
            results['fmt_holds'].append(fmt_result['fmt_holds'])
        
        return results
    
    @staticmethod
    def test_independence_assumption(model, num_samples: int = 10) -> Dict[str, Any]:
        """
        Test if the production technology satisfies Assumption A7 (Independence of Production).
        
        This assumption is crucial for the Fundamental Marxian Theorem.
        
        Parameters:
            model: Economic model with production functions
            num_samples (int): Number of random samples to test
            
        Returns:
            Dict[str, Any]: Test results
        """
        if not hasattr(model, 'production_functions') or not hasattr(model, 'labor_functions'):
            return {'assumption_holds': 'Not applicable for this model type'}
            
        from scipy.optimize import minimize
        
        test_cases = []
        
        for _ in range(num_samples):
            # Generate random output target
            target_output = np.random.rand(model.n) * np.mean(model.omega) * 0.2
            
            # Find minimum labor for full output
            def objective_full(x):
                labor = 0
                for i in range(len(model.production_functions)):
                    input_vector = x[i*model.n:(i+1)*model.n]
                    labor += model.labor_functions[i](input_vector)
                return labor
            
            def constraint_full(x):
                net_output = np.zeros(model.n)
                for i in range(len(model.production_functions)):
                    input_vector = x[i*model.n:(i+1)*model.n]
                    output = model.production_functions[i](input_vector)
                    net_output += output - input_vector
                return net_output - target_output
            
            x0 = np.ones(len(model.production_functions) * model.n) / (len(model.production_functions) * model.n)
            constraints_full = [{'type': 'ineq', 'fun': lambda x: constraint_full(x)[i]} for i in range(model.n)]
            bounds = [(0, None) for _ in range(len(model.production_functions) * model.n)]
            
            result_full = minimize(objective_full, x0, bounds=bounds, constraints=constraints_full)
            
            if not result_full.success:
                continue
            
            full_output_x = result_full.x
            full_labor = result_full.fun
            
            # Generate partial output (some fraction of full output)
            partial_output = target_output * (0.3 + np.random.rand(model.n) * 0.5)
            
            # Find minimum labor for partial output
            def constraint_partial(x):
                net_output = np.zeros(model.n)
                for i in range(len(model.production_functions)):
                    input_vector = x[i*model.n:(i+1)*model.n]
                    output = model.production_functions[i](input_vector)
                    net_output += output - input_vector
                return net_output - partial_output
            
            constraints_partial = [{'type': 'ineq', 'fun': lambda x: constraint_partial(x)[i]} for i in range(model.n)]
            
            result_partial = minimize(objective_full, x0, bounds=bounds, constraints=constraints_partial)
            
            if not result_partial.success:
                continue
            
            partial_labor = result_partial.fun
            
            # Check if assumption A7 holds
            a7_holds = partial_labor < full_labor
            
            test_cases.append({
                'full_output': target_output,
                'partial_output': partial_output,
                'full_labor': full_labor,
                'partial_labor': partial_labor,
                'a7_holds': a7_holds
            })
        
        # Check if assumption A7 holds for all test cases
        assumption_holds = all(case['a7_holds'] for case in test_cases) if test_cases else False
        
        return {
            'assumption_holds': assumption_holds,
            'test_cases': test_cases
        }
