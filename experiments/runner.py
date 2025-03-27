"""
Experiment runners for Marxian economic models.

This module provides tools for running experiments and analyses with Marxian models.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Union, Optional
import os

# Set Bayesian Methods for Hackers style
plt.style.use('bmh')


class MarxianExperiments:
   """Tools for running experiments and analyses with Marxian models."""
   
   @staticmethod
   def run_fmt_experiment(model, parameter_name, values):
       """
       Run experiment testing the Fundamental Marxian Theorem.
       
       Parameters:
           model: Economic model
           parameter_name (str): Parameter to vary
           values (List): Parameter values to test
           
       Returns:
           Dict[str, Any]: Experiment results
       """
       from marxge.analysis.fmt import FundamentalMarxianTheorem
       from marxge.analysis.visualization import Visualizer
       
       fmt_results = FundamentalMarxianTheorem.test_theorem(model, parameter_name, values)
       
       # Plot results
       fig = Visualizer.plot_fmt_analysis(fmt_results)
       
       return {
           'fmt_results': fmt_results,
           'figure': fig
       }
   
   @staticmethod
   def run_general_equilibrium_experiment(model, parameter_name, values):
       """
       Run general equilibrium experiment.
       
       Parameters:
           model: Economic model
           parameter_name (str): Parameter to vary
           values (List): Parameter values to test
           
       Returns:
           Dict[str, Any]: Experiment results
       """
       from marxge.analysis.general_equilibrium import GeneralEquilibriumAnalysis
       from marxge.analysis.visualization import Visualizer
       
       ge_results = GeneralEquilibriumAnalysis.analyze_general_equilibrium_effects(model, parameter_name, values)
       
       # Plot results
       fig = Visualizer.plot_general_equilibrium_analysis(ge_results)
       
       return {
           'ge_results': ge_results,
           'figure': fig
       }
   
   @staticmethod
   def run_social_determination_experiment(model, periods=10, class_power_trajectory=None):
       """
       Run experiment with social determination of workers' consumption.
       
       Parameters:
           model: SocialDeterminationModel
           periods (int): Number of periods to simulate
           class_power_trajectory (List[float]): Class power parameters for each period
           
       Returns:
           Dict[str, Any]: Experiment results
       """
       from marxge.analysis.visualization import Visualizer
       
       if not hasattr(model, 'simulate_class_struggle_dynamics'):
           raise ValueError("Model must be a SocialDeterminationModel")
       
       # Run simulation
       time_series = model.simulate_class_struggle_dynamics(periods, class_power_trajectory)
       
       # Plot results
       fig = Visualizer.plot_social_determination(time_series)
       
       return {
           'time_series': time_series,
           'figure': fig
       }
   
   @staticmethod
   def run_all_experiments(model_type='linear_two_sector', output_dir=None):
       """
       Run all experiments with a specified model type.
       
       Parameters:
           model_type (str): Type of model to use
           output_dir (str): Directory to save output files
           
       Returns:
           Dict[str, Any]: All experiment results
       """
       from marxge.utils.examples import ExampleModels
       
       # Create output directory if specified
       if output_dir:
           os.makedirs(output_dir, exist_ok=True)
       
       # Create model
       if model_type == 'linear_corn':
           model = ExampleModels.create_linear_corn_model()
       elif model_type == 'linear_two_sector':
           model = ExampleModels.create_linear_two_sector_model()
       elif model_type == 'linear_three_sector':
           model = ExampleModels.create_linear_three_sector_model()
       elif model_type == 'convex_two_sector':
           model = ExampleModels.create_convex_two_sector_model()
       elif model_type == 'joint_production':
           model = ExampleModels.create_joint_production_model()
       else:
           raise ValueError(f"Unknown model type: {model_type}")
       
       # Run basic analysis
       exploitation_rate = model.compute_exploitation_rate()
       
       if hasattr(model, 'find_equal_profit_rate_prices'):
           prices, profit_rate = model.find_equal_profit_rate_prices()
       elif hasattr(model, 'find_equilibrium_prices'):
           prices, profit_rate = model.find_equilibrium_prices()
       else:
           prices = np.ones(model.n) / np.sum(model.b)
           profit_rate = 0.0
       
       # Test if Independence Assumption (A7) holds
       if hasattr(model, 'production_functions'):
           from marxge.analysis.fmt import FundamentalMarxianTheorem
           a7_test = FundamentalMarxianTheorem.test_independence_assumption(model)
       else:
           a7_test = {'assumption_holds': 'Not applicable for linear model'}
       
       # Run FMT experiment
       subsistence_values = [0.5, 0.75, 1.0, 1.25, 1.5]
       fmt_subsistence = MarxianExperiments.run_fmt_experiment(model, 'subsistence', subsistence_values)
       
       productivity_values = [0.7, 0.85, 1.0, 1.15, 1.3]
       fmt_productivity = MarxianExperiments.run_fmt_experiment(model, 'productivity', productivity_values)
       
       # Run general equilibrium experiment
       ge_subsistence = MarxianExperiments.run_general_equilibrium_experiment(model, 'subsistence', subsistence_values)
       ge_productivity = MarxianExperiments.run_general_equilibrium_experiment(model, 'productivity', productivity_values)
       
       # Create social determination model
       social_model = ExampleModels.create_social_determination_model(model if hasattr(model, 'production_functions') else None)
       
       # Run social determination experiment
       class_power_trajectory = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.55, 0.5, 0.45]
       social_experiment = MarxianExperiments.run_social_determination_experiment(social_model, 10, class_power_trajectory)
       
       # Save figures if output directory is specified
       if output_dir:
           fmt_subsistence['figure'].savefig(os.path.join(output_dir, f"{model_type}_fmt_subsistence.png"))
           fmt_productivity['figure'].savefig(os.path.join(output_dir, f"{model_type}_fmt_productivity.png"))
           ge_subsistence['figure'].savefig(os.path.join(output_dir, f"{model_type}_ge_subsistence.png"))
           ge_productivity['figure'].savefig(os.path.join(output_dir, f"{model_type}_ge_productivity.png"))
           social_experiment['figure'].savefig(os.path.join(output_dir, f"{model_type}_social_determination.png"))
       
       # Return all results
       return {
           'model': model,
           'exploitation_rate': exploitation_rate,
           'profit_rate': profit_rate,
           'prices': prices,
           'a7_test': a7_test,
           'fmt_subsistence': fmt_subsistence,
           'fmt_productivity': fmt_productivity,
           'ge_subsistence': ge_subsistence,
           'ge_productivity': ge_productivity,
           'social_experiment': social_experiment
       }
   
   @staticmethod
   def run_interactive_experiment():
       """
       Run an interactive experiment with user input.
       
       This function prompts the user for experiment parameters and
       runs the corresponding analysis.
       
       Returns:
           Dict[str, Any]: Experiment results
       """
       from marxge.utils.examples import ExampleModels
       
       print("\nMarxian Economic Experiment Wizard")
       print("----------------------------------")
       
       # Choose model type
       print("\nSelect model type:")
       print("1. Linear Corn Model")
       print("2. Linear Two-Sector Model")
       print("3. Linear Three-Sector Model")
       print("4. Convex Two-Sector Model")
       print("5. Joint Production Model")
       print("6. Social Determination Model")
       print("7. Create Custom Model")
       print("8. Load Model from File")
       
       model_choice = input("Enter your choice (1-8): ")
       
       # Create or load model
       if model_choice == "1":
           model = ExampleModels.create_linear_corn_model()
       elif model_choice == "2":
           model = ExampleModels.create_linear_two_sector_model()
       elif model_choice == "3":
           model = ExampleModels.create_linear_three_sector_model()
       elif model_choice == "4":
           model = ExampleModels.create_convex_two_sector_model()
       elif model_choice == "5":
           model = ExampleModels.create_joint_production_model()
       elif model_choice == "6":
           model = ExampleModels.create_social_determination_model()
       elif model_choice == "7":
           model = ExampleModels.create_from_user_input()
       elif model_choice == "8":
           filename = input("Enter path to model parameter file: ")
           model = ExampleModels.load_from_file(filename)
       else:
           print("Invalid choice. Using default two-sector linear model.")
           model = ExampleModels.create_linear_two_sector_model()
       
       # Choose experiment type
       print("\nSelect experiment type:")
       print("1. Basic Analysis")
       print("2. Fundamental Marxian Theorem Test")
       print("3. General Equilibrium Analysis")
       print("4. Social Determination Simulation")
       print("5. Run All Experiments")
       
       experiment_choice = input("Enter your choice (1-5): ")
       
       # Get output directory
       save_output = input("\nSave output figures? (y/n): ")
       output_dir = None
       if save_output.lower() == 'y':
           output_dir = input("Enter output directory path: ")
           os.makedirs(output_dir, exist_ok=True)
       
       # Run selected experiment
       if experiment_choice == "1":
           # Basic analysis
           results = {}
           
           # Compute basic metrics
           exploitation_rate = model.compute_exploitation_rate()
           
           if hasattr(model, 'find_equal_profit_rate_prices'):
               prices, profit_rate = model.find_equal_profit_rate_prices()
           elif hasattr(model, 'find_equilibrium_prices'):
               prices, profit_rate = model.find_equilibrium_prices()
           else:
               prices = np.ones(model.n) / np.sum(model.b)
               profit_rate = 0.0
           
           # Print results
           print("\nBasic Analysis Results:")
           print(f"Exploitation Rate: {exploitation_rate:.4f}")
           print(f"Profit Rate: {profit_rate:.4f}")
           print(f"Prices: {prices}")
           
           # Plot price-labor value relationship if applicable
           if hasattr(model, 'compute_labor_values'):
               labor_values = model.compute_labor_values()
               
               # Create plot
               fig, ax = plt.subplots(figsize=(8, 6))
               ax.scatter(labor_values, prices, s=80, alpha=0.7)
               
               # Add labels for each point
               for i in range(len(prices)):
                   ax.annotate(f"Good {i+1}", (labor_values[i], prices[i]),
                              xytext=(5, 5), textcoords='offset points')
               
               # Add line of best fit
               coef = np.polyfit(labor_values, prices, 1)
               poly1d_fn = np.poly1d(coef)
               ax.plot(labor_values, poly1d_fn(labor_values), 'r--', alpha=0.7)
               
               # Format plot
               ax.set_xlabel('Labor Value')
               ax.set_ylabel('Price')
               ax.set_title('Price-Labor Value Relationship')
               ax.grid(True)
               
               # Add correlation coefficient
               corr = np.corrcoef(labor_values, prices)[0,1]
               ax.text(0.05, 0.95, f'Correlation: {corr:.4f}', transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
               
               plt.tight_layout()
               
               # Save plot if requested
               if output_dir:
                   fig.savefig(os.path.join(output_dir, "price_labor_value.png"))
                   print(f"Plot saved to {os.path.join(output_dir, 'price_labor_value.png')}")
               
               plt.show()
               
               results['labor_values'] = labor_values
               results['price_labor_correlation'] = corr
           
           results['exploitation_rate'] = exploitation_rate
           results['profit_rate'] = profit_rate
           results['prices'] = prices
           
           return results
           
       elif experiment_choice == "2":
           # FMT test
           parameter = input("\nVary parameter (subsistence/productivity): ")
           
           if parameter.lower().startswith('s'):
               parameter_name = 'subsistence'
           else:
               parameter_name = 'productivity'
           
           start_val = float(input("Start value: "))
           end_val = float(input("End value: "))
           num_points = int(input("Number of points: "))
           
           values = np.linspace(start_val, end_val, num_points)
           
           # Run experiment
           fmt_results = MarxianExperiments.run_fmt_experiment(model, parameter_name, values)
           
           # Save figure if requested
           if output_dir:
               fmt_results['figure'].savefig(os.path.join(output_dir, f"fmt_{parameter_name}.png"))
               print(f"Plot saved to {os.path.join(output_dir, f'fmt_{parameter_name}.png')}")
           
           plt.show()
           
           # Print summary
           print("\nFundamental Marxian Theorem Test Results:")
           fmt_data = fmt_results['fmt_results']
           
           print("\nParameter Values | Exploitation Rate | Profit Rate | FMT Holds")
           print("-" * 65)
           
           for i in range(len(fmt_data['parameter_values'])):
               print(f"{fmt_data['parameter_values'][i]:14.4f} | "
                     f"{fmt_data['exploitation_rates'][i]:16.4f} | "
                     f"{fmt_data['profit_rates'][i]:11.4f} | "
                     f"{fmt_data['fmt_holds'][i]}")
           
           return fmt_results
           
       elif experiment_choice == "3":
           # General equilibrium analysis
           parameter = input("\nVary parameter (subsistence/productivity/endowments): ")
           
           if parameter.lower().startswith('s'):
               parameter_name = 'subsistence'
           elif parameter.lower().startswith('p'):
               parameter_name = 'productivity'
           else:
               parameter_name = 'endowments'
           
           start_val = float(input("Start value: "))
           end_val = float(input("End value: "))
           num_points = int(input("Number of points: "))
           
           values = np.linspace(start_val, end_val, num_points)
           
           # Run experiment
           ge_results = MarxianExperiments.run_general_equilibrium_experiment(model, parameter_name, values)
           
           # Save figure if requested
           if output_dir:
               ge_results['figure'].savefig(os.path.join(output_dir, f"ge_{parameter_name}.png"))
               print(f"Plot saved to {os.path.join(output_dir, f'ge_{parameter_name}.png')}")
           
           plt.show()
           
           # Print summary
           print("\nGeneral Equilibrium Analysis Results:")
           ge_data = ge_results['ge_results']
           
           print("\nParameter Values | Exploitation Rate | Profit Rate | Reproducible")
           print("-" * 65)
           
           for i in range(len(ge_data['parameter_values'])):
               print(f"{ge_data['parameter_values'][i]:14.4f} | "
                     f"{ge_data['exploitation_rates'][i]:16.4f} | "
                     f"{ge_data['profit_rates'][i]:11.4f} | "
                     f"{ge_data['reproducibility'][i]}")
           
           return ge_results
           
       elif experiment_choice == "4":
           # Social determination simulation
           if not hasattr(model, 'simulate_class_struggle_dynamics'):
               # Convert model to social determination if needed
               if hasattr(model, 'production_functions'):
                   model = ExampleModels.create_social_determination_model(model)
               else:
                   print("Cannot run social determination simulation with this model type.")
                   print("Creating a default social determination model...")
                   model = ExampleModels.create_social_determination_model()
           
           # Get simulation parameters
           periods = int(input("\nNumber of periods to simulate: "))
           
           # Class power trajectory
           use_custom = input("Use custom class power trajectory? (y/n): ")
           
           if use_custom.lower() == 'y':
               print("Enter class power values separated by spaces:")
               cp_input = input("Class power trajectory: ")
               class_power_trajectory = [float(x) for x in cp_input.split()]
               
               # Ensure we have enough values
               if len(class_power_trajectory) < periods:
                   # Extend with last value
                   class_power_trajectory.extend([class_power_trajectory[-1]] * (periods - len(class_power_trajectory)))
               
               # Truncate if too long
               class_power_trajectory = class_power_trajectory[:periods]
           else:
               # Use default increasing then decreasing pattern
               class_power_trajectory = np.concatenate([
                   np.linspace(0.3, 0.6, periods // 2),
                   np.linspace(0.6, 0.3, periods - periods // 2)
               ])
           
           # Run experiment
           social_results = MarxianExperiments.run_social_determination_experiment(
               model, periods, class_power_trajectory
           )
           
           # Save figure if requested
           if output_dir:
               social_results['figure'].savefig(os.path.join(output_dir, "social_determination.png"))
               print(f"Plot saved to {os.path.join(output_dir, 'social_determination.png')}")
           
           plt.show()
           
           # Print summary
           print("\nSocial Determination Simulation Results:")
           time_series = social_results['time_series']
           
           print("\nPeriod | Class Power | Exploitation Rate | Profit Rate | Reproducible")
           print("-" * 75)
           
           for i in range(periods):
               print(f"{i:6d} | "
                     f"{time_series['class_power'][i]:11.4f} | "
                     f"{time_series['exploitation_rate'][i]:16.4f} | "
                     f"{time_series['profit_rate'][i]:11.4f} | "
                     f"{time_series['reproducible'][i]}")
           
           return social_results
           
       elif experiment_choice == "5":
           # Run all experiments
           # Determine model type name
           if hasattr(model, 'A') and hasattr(model, 'L'):
               if model.n == 1:
                   model_type = 'linear_corn'
               elif model.n == 2:
                   model_type = 'linear_two_sector'
               else:
                   model_type = 'linear_three_sector'
           elif hasattr(model, 'production_functions'):
               if hasattr(model, 'class_power'):
                   model_type = 'social_determination'
               elif len(model.production_functions) > 1 and model.n == 2:
                   model_type = 'joint_production'
               else:
                   model_type = 'convex_two_sector'
           else:
               model_type = 'custom_model'
           
           # Run all experiments
           results = MarxianExperiments.run_all_experiments(model_type, output_dir)
           
           # Print summary
           print("\nAll Experiments Completed!")
           print(f"Exploitation Rate: {results['exploitation_rate']:.4f}")
           print(f"Profit Rate: {results['profit_rate']:.4f}")
           print(f"Independence Assumption (A7) holds: {results['a7_test']['assumption_holds']}")
           
           if output_dir:
               print(f"\nAll figures saved to {output_dir}")
           
           return results
           
       else:
           print("Invalid choice.")
           return None
