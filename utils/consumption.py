"""
Consumption determination mechanisms for Marxian models.

This module implements different approaches to determining workers' subsistence
consumption, as discussed in Section 4 of Roemer's paper.
"""

import numpy as np
from typing import Dict, List, Any, Union, Optional


def technological_needs_consumption(total_output, total_labor, class_power, activity_levels):
    """
    Worker consumption determined by technological requirements and class power.
    
    This implements a consumption mechanism where workers' subsistence is determined
    by technological requirements, economic development level, and class struggle.
    
    Parameters:
        total_output (ndarray): Vector of total outputs by commodity
        total_labor (float): Total labor expended
        class_power (float): Parameter representing relative power of workers vs capitalists
        activity_levels (ndarray): Current production activity levels
        
    Returns:
        ndarray: Subsistence bundle per unit of labor
    """
    # Basic physiological subsistence (minimum consumption)
    n = len(total_output)
    basic_subsistence = np.ones(n) * 0.1
    
    # Additional consumption based on technological development
    tech_sophistication = np.sum(activity_levels) / len(activity_levels)
    tech_component = np.array([0.1] + [0.05] * (n-1)) * min(tech_sophistication, 2.0)
    
    # Class struggle component - workers appropriate more output with higher class power
    if total_labor > 0:
        struggle_component = class_power * total_output / total_labor * 0.2
    else:
        struggle_component = np.zeros(n)
    
    # Combined subsistence bundle
    subsistence = basic_subsistence + tech_component + struggle_component
    
    # Ensure subsistence is not greater than per-capita output
    if total_labor > 0:
        max_sustainable = total_output / total_labor * 0.9  # Leave some for accumulation
        return np.minimum(subsistence, max_sustainable)
    else:
        return basic_subsistence


def historical_moral_consumption(total_output, total_labor, class_power, activity_levels, 
                                 prev_subsistence=None, adjustment_rate=0.2):
    """
    Worker consumption with historical and moral elements, as Marx suggested.
    
    This implements a consumption mechanism where workers' subsistence has a 
    historical component (slow adjustment to changing conditions) and a moral
    component (class struggle determines labor's share).
    
    Parameters:
        total_output (ndarray): Vector of total outputs by commodity
        total_labor (float): Total labor expended
        class_power (float): Parameter representing relative power of workers vs capitalists
        activity_levels (ndarray): Current production activity levels
        prev_subsistence (ndarray): Previous period's subsistence bundle
        adjustment_rate (float): Rate at which consumption adjusts to new conditions
        
    Returns:
        ndarray: Subsistence bundle per unit of labor
    """
    n = len(total_output)
    
    # Base case - no previous subsistence
    if prev_subsistence is None:
        prev_subsistence = np.ones(n) * 0.2
    
    # Calculate "target" subsistence based on current conditions
    # This represents what workers would consume if adjustment were instantaneous
    if total_labor > 0:
        per_capita_output = total_output / total_labor
        target = per_capita_output * class_power * 0.5  # Workers get class_power share of half the output
    else:
        target = np.ones(n) * 0.2
    
    # Historical adjustment - consumption changes gradually based on past patterns
    new_subsistence = prev_subsistence + adjustment_rate * (target - prev_subsistence)
    
    # Ensure minimum physiological needs are met
    minimum_needs = np.ones(n) * 0.1
    new_subsistence = np.maximum(new_subsistence, minimum_needs)
    
    # Ensure subsistence is sustainable
    if total_labor > 0:
        max_sustainable = total_output / total_labor * 0.9
        return np.minimum(new_subsistence, max_sustainable)
    else:
        return new_subsistence


def custom_consumption_function(total_output, total_labor, class_power, activity_levels, 
                                parameters=None):
    """
    Custom consumption function that users can modify.
    
    This provides a template for users to implement their own consumption mechanisms.
    
    Parameters:
        total_output (ndarray): Vector of total outputs by commodity
        total_labor (float): Total labor expended
        class_power (float): Parameter representing relative power of workers vs capitalists
        activity_levels (ndarray): Current production activity levels
        parameters (Dict): Additional parameters for customization
        
    Returns:
        ndarray: Subsistence bundle per unit of labor
    """
    if parameters is None:
        parameters = {}
    
    n = len(total_output)
    
    # Default parameters
    basic_needs_weight = parameters.get('basic_needs_weight', 0.5)
    class_power_weight = parameters.get('class_power_weight', 0.5)
    luxury_items_threshold = parameters.get('luxury_threshold', 0.7)
    
    # Basic needs component - essential goods for survival
    basic_needs = np.ones(n) * 0.1 * basic_needs_weight
    
    # Class struggle component - depends on workers' bargaining power
    if total_labor > 0:
        bargaining_component = class_power * total_output / total_labor * class_power_weight
    else:
        bargaining_component = np.zeros(n)
    
    # Luxury items - only accessible when class power exceeds threshold
    luxury_component = np.zeros(n)
    if class_power > luxury_items_threshold:
        # Higher index goods are considered more luxurious
        for i in range(n):
            luxury_weight = i / (n-1) if n > 1 else 0  # 0 for first good, 1 for last
            luxury_component[i] = 0.1 * luxury_weight * (class_power - luxury_items_threshold)
    
    # Combined subsistence bundle
    subsistence = basic_needs + bargaining_component + luxury_component
    
    # Ensure subsistence is not greater than per-capita output
    if total_labor > 0:
        max_sustainable = total_output / total_labor * 0.9
        return np.minimum(subsistence, max_sustainable)
    else:
        return basic_needs
