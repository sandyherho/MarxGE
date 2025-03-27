"""
General Equilibrium Analysis.

This module provides tools for analyzing general equilibrium effects
and performing comparative statics in Marxian models.
"""

import numpy as np
import copy
from typing import Dict, List, Any, Union, Optional


class GeneralEquilibriumAnalysis:
    """
    Tools for analyzing general equilibrium effects and performing comparative statics.
    """
    
    @staticmethod
    def analyze_general_equilibrium_effects(model, parameter_name, values):
        """
        Analyze general equilibrium effects of changing a parameter.
        
        Parameters:
            model: Economic model (MarxianLinearModel, ConvexProductionModel, or SocialDeterminationModel)
            parameter_name (str): Name of parameter to vary
            values (List): List of parameter values to test
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        results = {
            'parameter_values': values,
            'prices': [],
            'exploitation_rates': [],
            'profit_rates': [],
            'reproducibility': []
        }
        
        for value in values:
            # Modify model based on parameter
            modified_model = copy.deepcopy(model)
            
            if parameter_name == 'subsistence':
                if hasattr(modified_model, 'b'):
                    modified_model.b = model.b * value
                    
                    # Update M matrix for linear model
                    if hasattr(modified_model, 'M'):
                        modified_model.M = modified_model.A + np.outer(modified_model.b, modified_model.L)
            
            elif parameter_name == 'endowments':
                if hasattr(modified_model, 'omega'):
                    modified_model.omega = model.omega * value
            
            elif parameter_name == 'productivity':
                if hasattr(modified_model, 'A'):
                    # For linear model, decrease input requirements
                    modified_model.A = model.A / value
                    
                    # Update M matrix
                    if hasattr(modified_model, 'M'):
                        modified_model.M = modified_model.A + np.outer(modified_model.b, modified_model.L)
                
                elif hasattr(modified_model, 'production_functions'):
                    # For convex model, scale production functions
                    orig_production_functions = model.production_functions
                    modified_model.production_functions = [
                        lambda x, orig_func=func, v=value: orig_func(x) * v
                        for func in orig_production_functions
                    ]
            
            # Calculate prices and profit rate
            try:
                if hasattr(modified_model, 'find_equal_profit_rate_prices'):
                    prices, profit_rate = modified_model.find_equal_profit_rate_prices()
                elif hasattr(modified_model, 'find_equilibrium_prices'):
                    prices, profit_rate = modified_model.find_equilibrium_prices()
                else:
                    prices = np.ones(modified_model.n) / modified_model.n
                    profit_rate = 0.0
            except Exception as e:
                # Default values if calculation fails
                prices = np.ones(modified_model.n) / modified_model.n
                profit_rate = 0.0
                print(f"Warning: Failed to calculate prices - {e}")
            
            # Calculate exploitation rate
            try:
                exploitation_rate = modified_model.compute_exploitation_rate()
            except Exception as e:
                exploitation_rate = 0.0
                print(f"Warning: Failed to calculate exploitation rate - {e}")
            
            # Check reproducibility
            try:
                if hasattr(modified_model, 'is_reproducible'):
                    # For linear model
                    reproducible = modified_model.is_reproducible(prices, [modified_model.omega])
                else:
                    # For convex models, perform simplified check
                    total_output = np.zeros(modified_model.n)
                    total_labor = 0
                    
                    activity_levels = np.ones(modified_model.n) / modified_model.n
                    
                    for i, prod_func in enumerate(modified_model.production_functions):
                        output = prod_func(activity_levels)
                        labor = modified_model.labor_functions[i](activity_levels)
                        
                        total_output += output
                        total_labor += labor
                    
                    worker_consumption = total_labor * modified_model.b
                    reproducible = np.all(total_output >= worker_consumption)
            except Exception as e:
                reproducible = False
                print(f"Warning: Failed to check reproducibility - {e}")
            
            # Store results
            results['prices'].append(prices)
            results['exploitation_rates'].append(exploitation_rate)
            results['profit_rates'].append(profit_rate)
            results['reproducibility'].append(reproducible)
        
        return results
    
    @staticmethod
    def analyze_class_power_effects(model, class_power_values):
        """
        Analyze the effects of varying class power on economic outcomes.
        
        This is applicable only for SocialDeterminationModel.
        
        Parameters:
            model: SocialDeterminationModel
            class_power_values (List[float]): Class power values to test
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        if not hasattr(model, 'class_power'):
            return {'error': 'Model does not support class power analysis'}
        
        results = {
            'class_power_values': class_power_values,
            'subsistence_bundles': [],
            'exploitation_rates': [],
            'profit_rates': [],
            'reproducibility': []
        }
        
        original_class_power = model.class_power
        
        for value in class_power_values:
            # Modify model
            model.class_power = value
            
            # Update subsistence bundle
            model.update_subsistence_bundle()
            
            # Calculate prices and profit rate
            try:
                prices, profit_rate = model.find_equilibrium_prices()
            except Exception as e:
                prices = np.ones(model.n) / model.n
                profit_rate = 0.0
                print(f"Warning: Failed to calculate prices - {e}")
            
            # Calculate exploitation rate
            try:
                exploitation_rate = model.compute_exploitation_rate()
            except Exception as e:
                exploitation_rate = 0.0
                print(f"Warning: Failed to calculate exploitation rate - {e}")
            
            # Check reproducibility
            try:
                # For SocialDeterminationModel, perform simplified check
                total_output = np.zeros(model.n)
                total_labor = 0
                
                activity_levels = np.ones(model.n) / model.n
                
                for i, prod_func in enumerate(model.production_functions):
                    output = prod_func(activity_levels)
                    labor = model.labor_functions[i](activity_levels)
                    
                    total_output += output
                    total_labor += labor
                
                worker_consumption = total_labor * model.b
                reproducible = np.all(total_output >= worker_consumption)
            except Exception as e:
                reproducible = False
                print(f"Warning: Failed to check reproducibility - {e}")
            
            # Store results
            results['subsistence_bundles'].append(model.b.copy())
            results['exploitation_rates'].append(exploitation_rate)
            results['profit_rates'].append(profit_rate)
            results['reproducibility'].append(reproducible)
        
        # Restore original class power
        model.class_power = original_class_power
        model.update_subsistence_bundle()
        
        return results
    
    @staticmethod
    def compare_models(models, descriptions=None):
        """
        Compare different model specifications.
        
        Parameters:
            models (List): List of economic models to compare
            descriptions (List[str]): Descriptions of each model
            
        Returns:
            Dict[str, Any]: Comparison results
        """
        if descriptions is None:
            descriptions = [f"Model {i+1}" for i in range(len(models))]
        
        results = {
            'model_descriptions': descriptions,
            'exploitation_rates': [],
            'profit_rates': [],
            'reproducibility': []
        }
        
        for model in models:
            # Calculate exploitation rate
            try:
                exploitation_rate = model.compute_exploitation_rate()
            except Exception as e:
                exploitation_rate = np.nan
                print(f"Warning: Failed to calculate exploitation rate - {e}")
            
            # Calculate profit rate
            try:
                if hasattr(model, 'find_equal_profit_rate_prices'):
                    _, profit_rate = model.find_equal_profit_rate_prices()
                elif hasattr(model, 'find_equilibrium_prices'):
                    _, profit_rate = model.find_equilibrium_prices()
                else:
                    profit_rate = np.nan
            except Exception as e:
                profit_rate = np.nan
                print(f"Warning: Failed to calculate profit rate - {e}")
            
            # Check reproducibility
            try:
                if hasattr(model, 'is_reproducible'):
                    # For linear model
                    prices, _ = model.find_equal_profit_rate_prices()
                    reproducible = model.is_reproducible(prices, [model.omega])
                else:
                    # For convex models, perform simplified check
                    total_output = np.zeros(model.n)
                    total_labor = 0
                    
                    activity_levels = np.ones(model.n) / model.n
                    
                    for i, prod_func in enumerate(model.production_functions):
                        output = prod_func(activity_levels)
                        labor = model.labor_functions[i](activity_levels)
                        
                        total_output += output
                        total_labor += labor
                    
                    worker_consumption = total_labor * model.b
                    reproducible = np.all(total_output >= worker_consumption)
            except Exception as e:
                reproducible = False
                print(f"Warning: Failed to check reproducibility - {e}")
            
            # Store results
            results['exploitation_rates'].append(exploitation_rate)
            results['profit_rates'].append(profit_rate)
            results['reproducibility'].append(reproducible)
        
        return results
