"""
Visualization tools for Marxian economic models.

This module provides tools for visualizing model results and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Union, Optional
import warnings


class Visualizer:
    """Tools for visualizing model results and analysis."""
    
    @staticmethod
    def plot_fmt_analysis(fmt_results):
        """
        Plot Fundamental Marxian Theorem analysis results.
        
        Parameters:
            fmt_results (Dict): Results from FMT analysis
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(fmt_results['parameter_values'], fmt_results['exploitation_rates'], 'b-', label='Exploitation Rate')
        ax.plot(fmt_results['parameter_values'], fmt_results['profit_rates'], 'r--', label='Profit Rate')
        
        # Format the plot
        ax.set_title('Fundamental Marxian Theorem Analysis')
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Rate')
        ax.grid(True)
        ax.legend()
        
        # Highlight when FMT holds
        for i, holds in enumerate(fmt_results['fmt_holds']):
            color = 'green' if holds else 'red'
            ax.axvline(x=fmt_results['parameter_values'][i], color=color, alpha=0.1)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_general_equilibrium_analysis(ge_results):
        """
        Plot general equilibrium analysis results.
        
        Parameters:
            ge_results (Dict): Results from general equilibrium analysis
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot exploitation and profit rates
        axs[0, 0].plot(ge_results['parameter_values'], ge_results['exploitation_rates'], 'b-', label='Exploitation Rate')
        axs[0, 0].plot(ge_results['parameter_values'], ge_results['profit_rates'], 'r--', label='Profit Rate')
        axs[0, 0].set_title('Exploitation and Profit Rates')
        axs[0, 0].set_xlabel('Parameter Value')
        axs[0, 0].set_ylabel('Rate')
        axs[0, 0].grid(True)
        axs[0, 0].legend()
        
        # Plot reproducibility
        axs[0, 1].plot(ge_results['parameter_values'], ge_results['reproducibility'], 'g-')
        axs[0, 1].set_title('Reproducibility')
        axs[0, 1].set_xlabel('Parameter Value')
        axs[0, 1].set_ylabel('Reproducible')
        axs[0, 1].grid(True)
        
        # Plot prices
        prices_array = np.array(ge_results['prices'])
        for i in range(prices_array.shape[1]):
            axs[1, 0].plot(ge_results['parameter_values'], prices_array[:, i], label=f'Good {i+1}')
        axs[1, 0].set_title('Prices')
        axs[1, 0].set_xlabel('Parameter Value')
        axs[1, 0].set_ylabel('Price')
        axs[1, 0].grid(True)
        axs[1, 0].legend()
        
        # Plot price ratios (for 2-good case) or relative prices
        if prices_array.shape[1] == 2:
            price_ratios = prices_array[:, 0] / prices_array[:, 1]
            axs[1, 1].plot(ge_results['parameter_values'], price_ratios, 'k-')
            axs[1, 1].set_title('Price Ratio (Good 1 / Good 2)')
            axs[1, 1].set_xlabel('Parameter Value')
            axs[1, 1].set_ylabel('Ratio')
            axs[1, 1].grid(True)
        else:
            # Plot relative prices compared to good 1
            for i in range(1, prices_array.shape[1]):
                relative_prices = prices_array[:, i] / prices_array[:, 0]
                axs[1, 1].plot(ge_results['parameter_values'], relative_prices, label=f'Good {i+1} / Good 1')
            axs[1, 1].set_title('Relative Prices (to Good 1)')
            axs[1, 1].set_xlabel('Parameter Value')
            axs[1, 1].set_ylabel('Relative Price')
            axs[1, 1].grid(True)
            axs[1, 1].legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_social_determination(time_series):
        """
        Plot social determination of workers' consumption analysis.
        
        Parameters:
            time_series (Dict): Time series data from social determination simulation
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot exploitation and profit rates
        axs[0].plot(time_series['periods'], time_series['exploitation_rate'], 'b-', label='Exploitation Rate')
        axs[0].plot(time_series['periods'], time_series['profit_rate'], 'r--', label='Profit Rate')
        axs[0].set_title('Exploitation and Profit Rates')
        axs[0].set_xlabel('Period')
        axs[0].set_ylabel('Rate')
        axs[0].grid(True)
        axs[0].legend()
        
        # Plot class power
        axs[1].plot(time_series['periods'], time_series['class_power'], 'g-')
        axs[1].set_title('Class Power Parameter')
        axs[1].set_xlabel('Period')
        axs[1].set_ylabel('Class Power')
        axs[1].grid(True)
        
        # Plot subsistence bundles
        subsistence_array = np.array(time_series['subsistence'])
        for i in range(subsistence_array.shape[1]):
            axs[2].plot(time_series['periods'], subsistence_array[:, i], label=f'Good {i+1}')
        axs[2].set_title('Subsistence Bundle Components')
        axs[2].set_xlabel('Period')
        axs[2].set_ylabel('Quantity')
        axs[2].grid(True)
        axs[2].legend()
        
        # Highlight periods where economy is not reproducible
        for i, reprod in enumerate(time_series['reproducibility']):
            if not reprod:
                for j in range(3):
                    axs[j].axvspan(i-0.1, i+0.1, color='red', alpha=0.2)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_production_set(model, grid_size=20):
        """
        Visualize the production set for a 2-commodity model.
        
        Parameters:
            model: Economic model with production functions
            grid_size (int): Number of grid points
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if not hasattr(model, 'production_functions') or len(model.production_functions) == 0:
            raise ValueError("Model must have production functions")
        
        if model.n != 2:
            raise ValueError("Production set visualization only supports 2 commodities")
        
        # Create grid of input combinations
        max_input = np.max(model.omega) * 0.5
        x = np.linspace(0, max_input, grid_size)
        y = np.linspace(0, max_input, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Calculate outputs and labor for each input combination
        outputs = np.zeros((grid_size, grid_size, model.n))
        labor = np.zeros((grid_size, grid_size))
        
        for i in range(grid_size):
            for j in range(grid_size):
                inputs = np.array([X[i, j], Y[i, j]])
                
                # Sum outputs across all production functions
                for k, prod_func in enumerate(model.production_functions):
                    try:
                        outputs[i, j] += prod_func(inputs)
                    except Exception as e:
                        # Handle potential errors in production functions
                        pass
                
                # Sum labor across all labor functions
                for k, labor_func in enumerate(model.labor_functions):
                    try:
                        labor[i, j] += labor_func(inputs)
                    except Exception as e:
                        # Handle potential errors in labor functions
                        pass
        
        # Calculate net outputs
        net_output_1 = outputs[:, :, 0] - X
        net_output_2 = outputs[:, :, 1] - Y
        
        # Create figure
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot net output 1
        contour1 = axs[0].contourf(X, Y, net_output_1, 20)
        axs[0].set_title('Net Output of Good 1')
        axs[0].set_xlabel('Input of Good 1')
        axs[0].set_ylabel('Input of Good 2')
        fig.colorbar(contour1, ax=axs[0])
        
        # Plot net output 2
        contour2 = axs[1].contourf(X, Y, net_output_2, 20)
        axs[1].set_title('Net Output of Good 2')
        axs[1].set_xlabel('Input of Good 1')
        axs[1].set_ylabel('Input of Good 2')
        fig.colorbar(contour2, ax=axs[1])
        
        # Plot labor input
        contour3 = axs[2].contourf(X, Y, labor, 20)
        axs[2].set_title('Labor Input')
        axs[2].set_xlabel('Input of Good 1')
        axs[2].set_ylabel('Input of Good 2')
        fig.colorbar(contour3, ax=axs[2])
        
        # Add reproducibility boundary
        for ax in axs:
            # Calculate boundary where net output equals subsistence
            # This is a simplification for visualization
            boundary_x = np.linspace(0, max_input, 100)
            boundary_y = np.zeros_like(boundary_x)
            
            for i, bx in enumerate(boundary_x):
                # Find approximate boundary point
                for by in np.linspace(0, max_input, 100):
                    inputs = np.array([bx, by])
                    net_output = np.zeros(model.n)
                    
                    for k, prod_func in enumerate(model.production_functions):
                        try:
                            output = prod_func(inputs)
                            net_output += output - inputs
                        except:
                            pass
                    
                    if np.all(net_output >= model.b):
                        boundary_y[i] = by
                        break
            
            ax.plot(boundary_x, boundary_y, 'k--', label='Reproducibility Boundary')
        
        axs[0].legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_model_comparison(comparison_results):
        """
        Plot comparison between different model specifications.
        
        Parameters:
            comparison_results (Dict): Results from model comparison
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        model_count = len(comparison_results['model_descriptions'])
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # Bar width
        bar_width = 0.35
        x = np.arange(model_count)
        
        # Plot exploitation rates
        axs[0].bar(x - bar_width/2, comparison_results['exploitation_rates'], bar_width, label='Exploitation Rate')
        axs[0].bar(x + bar_width/2, comparison_results['profit_rates'], bar_width, label='Profit Rate')
        axs[0].set_xticks(x)
        axs[0].set_xticklabels(comparison_results['model_descriptions'])
        axs[0].set_title('Exploitation and Profit Rates')
        axs[0].set_ylabel('Rate')
        axs[0].legend()
        
        # Plot reproducibility
        axs[1].bar(x, comparison_results['reproducibility'])
        axs[1].set_xticks(x)
        axs[1].set_xticklabels(comparison_results['model_descriptions'])
        axs[1].set_title('Reproducibility')
        axs[1].set_ylabel('Reproducible')
        
        plt.tight_layout()
        return fig
