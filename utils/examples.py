"""
Example models for the Marxian General Equilibrium framework.

This module provides ready-to-use example models with different configurations
for testing and demonstration purposes.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Union, Optional

# Set Bayesian Methods for Hackers style
plt.style.use('bmh')


class ExampleModels:
    """Collection of example models for testing and demonstration."""
    
    @staticmethod
    def create_linear_corn_model():
        """
        Create a simple one-good (corn) model.
        
        Returns:
            MarxianLinearModel: Corn model
        """
        from marxge.core.linear_model import MarxianLinearModel
        
        # Corn model: A = [[0.25]], L = [1.0], b = [0.5]
        A = np.array([[0.25]])
        L = np.array([1.0])
        b = np.array([0.5])
        omega = np.array([100.0])
        
        return MarxianLinearModel(A, L, b, omega)
    
    @staticmethod
    def create_linear_two_sector_model(custom_params=None):
        """
        Create a two-sector model with capital and consumer goods.
        
        Parameters:
            custom_params (Dict): Optional custom parameters to modify the model
                - A_matrix: Custom input-output matrix
                - L_vector: Custom labor input vector
                - b_vector: Custom subsistence bundle
                - omega_vector: Custom initial endowments
        
        Returns:
            MarxianLinearModel: Two-sector model
        """
        from marxge.core.linear_model import MarxianLinearModel
        
        if custom_params is None:
            custom_params = {}
            
        # Default parameters - Two-sector model:
        # Sector 1 (Capital goods): 0.2 units of good 1 + 0.1 units of good 2 + 0.6 labor -> 1 unit of good 1
        # Sector 2 (Consumer goods): 0.1 units of good 1 + 0.3 units of good 2 + 0.4 labor -> 1 unit of good 2
        A = custom_params.get('A_matrix', np.array([
            [0.2, 0.1],  # Input requirements for sector 1
            [0.1, 0.3]   # Input requirements for sector 2
        ]))
        L = custom_params.get('L_vector', np.array([0.6, 0.4]))  # Labor requirements
        b = custom_params.get('b_vector', np.array([0.5, 0.3]))  # Subsistence bundle
        omega = custom_params.get('omega_vector', np.array([100.0, 80.0]))  # Initial endowments
        
        return MarxianLinearModel(A, L, b, omega)
    
    @staticmethod
    def create_linear_three_sector_model(custom_params=None):
        """
        Create a three-sector model with raw materials, capital goods, and consumer goods.
        
        Parameters:
            custom_params (Dict): Optional custom parameters to modify the model
        
        Returns:
            MarxianLinearModel: Three-sector model
        """
        from marxge.core.linear_model import MarxianLinearModel
        
        if custom_params is None:
            custom_params = {}
            
        # Default parameters - Three-sector model:
        # Sector 1 (Raw materials): 0.1*g1 + 0.0*g2 + 0.0*g3 + 0.8 labor -> 1 unit of good 1
        # Sector 2 (Capital goods): 0.3*g1 + 0.2*g2 + 0.0*g3 + 0.5 labor -> 1 unit of good 2
        # Sector 3 (Consumer goods): 0.2*g1 + 0.1*g2 + 0.2*g3 + 0.6 labor -> 1 unit of good 3
        A = custom_params.get('A_matrix', np.array([
            [0.1, 0.3, 0.2],
            [0.0, 0.2, 0.1],
            [0.0, 0.0, 0.2]
        ]))
        L = custom_params.get('L_vector', np.array([0.8, 0.5, 0.6]))
        b = custom_params.get('b_vector', np.array([0.2, 0.1, 0.4]))
        omega = custom_params.get('omega_vector', np.array([100.0, 80.0, 60.0]))
        
        return MarxianLinearModel(A, L, b, omega)
    
    @staticmethod
    def create_convex_two_sector_model(custom_params=None):
        """
        Create a two-sector model with convex production technology.
        
        Parameters:
            custom_params (Dict): Optional custom parameters to modify the model
        
        Returns:
            ConvexProductionModel: Two-sector model with convex technology
        """
        from marxge.core.convex_model import ConvexProductionModel
        
        if custom_params is None:
            custom_params = {}
            
        # Production functions
        def sector1_production(inputs):
            # Sector 1: Capital goods with Cobb-Douglas technology
            return np.array([0.8 * inputs[0]**0.7 * inputs[1]**0.2, 0.1 * inputs[0]**0.5])
        
        def sector2_production(inputs):
            # Sector 2: Consumer goods with Cobb-Douglas technology
            return np.array([0.1 * inputs[1]**0.5, 0.7 * inputs[1]**0.6 * inputs[0]**0.3])
        
        # Labor functions
        def sector1_labor(inputs):
            return 0.5 * np.sum(inputs)**0.9
        
        def sector2_labor(inputs):
            return 0.4 * np.sum(inputs)**0.8
        
        # Use custom functions if provided
        production_functions = custom_params.get('production_functions', 
                                                [sector1_production, sector2_production])
        labor_functions = custom_params.get('labor_functions', 
                                           [sector1_labor, sector2_labor])
        
        # Initial endowments and subsistence bundle
        initial_endowments = custom_params.get('omega_vector', np.array([100.0, 80.0]))
        subsistence = custom_params.get('b_vector', np.array([0.2, 0.3]))
        
        # Create model
        model = ConvexProductionModel(
            production_functions,
            labor_functions,
            subsistence,
            initial_endowments
        )
        
        return model
    
    @staticmethod
    def create_joint_production_model(custom_params=None):
        """
        Create a model with joint production to test the independence assumption.
        
        Parameters:
            custom_params (Dict): Optional custom parameters to modify the model
        
        Returns:
            ConvexProductionModel: Model with joint production
        """
        from marxge.core.convex_model import ConvexProductionModel
        
        if custom_params is None:
            custom_params = {}
            
        # Production functions with fixed joint production
        def fixed_joint_production(inputs):
            # Fixed ratio joint production - always produces goods in the ratio 1:2
            total_scale = np.sum(inputs)
            return np.array([total_scale * 0.3, total_scale * 0.6])
        
        # Flexible production
        def flexible_production(inputs):
            # Can produce any combination along a production possibilities frontier
            return np.array([inputs[0] * 0.7, inputs[1] * 0.8])
        
        # Labor functions
        def labor1(inputs):
            return 0.4 * np.sum(inputs)
        
        def labor2(inputs):
            return 0.3 * np.sum(inputs)
        
        # Use custom functions if provided
        production_functions = custom_params.get('production_functions', 
                                                [fixed_joint_production, flexible_production])
        labor_functions = custom_params.get('labor_functions', 
                                           [labor1, labor2])
        
        # Initial endowments and subsistence bundle
        initial_endowments = custom_params.get('omega_vector', np.array([50.0, 40.0]))
        subsistence = custom_params.get('b_vector', np.array([0.1, 0.2]))
        
        # Create model
        model = ConvexProductionModel(
            production_functions,
            labor_functions,
            subsistence,
            initial_endowments
        )
        
        return model
    
    @staticmethod
    def create_social_determination_model(base_model=None, custom_params=None):
        """
        Create a model with social determination of workers' consumption.
        
        Parameters:
            base_model: Base model to use (default: create new convex two-sector model)
            custom_params (Dict): Optional custom parameters to modify the model
        
        Returns:
            SocialDeterminationModel: Model with social determination
        """
        from marxge.core.social_model import SocialDeterminationModel
        from marxge.utils.consumption import technological_needs_consumption
        
        if custom_params is None:
            custom_params = {}
            
        if base_model is None:
            base_model = ExampleModels.create_convex_two_sector_model()
            
        # Get the consumption function
        consumption_function = custom_params.get('consumption_function', technological_needs_consumption)
        class_power = custom_params.get('class_power', 0.4)
        
        # Create social determination model
        model = SocialDeterminationModel(
            base_model.production_functions,
            base_model.labor_functions,
            base_model.omega,
            consumption_function,
            class_power
        )
        
        return model
    
    @staticmethod
    def create_from_user_input():
        """
        Create a model based on user input.
        
        This interactive function allows users to create custom models
        by inputting parameters directly.
        
        Returns:
            Union[MarxianLinearModel, ConvexProductionModel, SocialDeterminationModel]: Custom model
        """
        print("\nModel Creation Wizard")
        print("---------------------")
        
        # Choose model type
        print("\nSelect model type:")
        print("1. Linear Model")
        print("2. Convex Production Model")
        print("3. Social Determination Model")
        
        model_type = input("Enter your choice (1-3): ")
        
        # Get number of sectors
        num_sectors = int(input("\nEnter number of sectors/commodities: "))
        
        if model_type == "1":  # Linear Model
            from marxge.core.linear_model import MarxianLinearModel
            
            # Get input-output matrix
            print("\nEnter input-output matrix A (row by row, space-separated values):")
            A = np.zeros((num_sectors, num_sectors))
            for i in range(num_sectors):
                row = input(f"Row {i+1}: ")
                A[i] = np.array([float(x) for x in row.split()])
            
            # Get labor input vector
            print("\nEnter labor input vector L (space-separated values):")
            L_input = input("Labor inputs: ")
            L = np.array([float(x) for x in L_input.split()])
            
            # Get subsistence bundle
            print("\nEnter workers' subsistence bundle b (space-separated values):")
            b_input = input("Subsistence bundle: ")
            b = np.array([float(x) for x in b_input.split()])
            
            # Get initial endowments
            print("\nEnter initial endowments omega (space-separated values):")
            omega_input = input("Endowments: ")
            omega = np.array([float(x) for x in omega_input.split()])
            
            # Create and return model
            return MarxianLinearModel(A, L, b, omega)
            
        elif model_type == "2":  # Convex Production Model
            from marxge.core.convex_model import ConvexProductionModel
            
            # For simplicity, use Cobb-Douglas production functions with user-defined parameters
            print("\nDefining Cobb-Douglas production functions...")
            production_functions = []
            labor_functions = []
            
            for i in range(num_sectors):
                print(f"\nSector {i+1} production function:")
                output_coeffs = []
                for j in range(num_sectors):
                    coeff = float(input(f"Output coefficient for good {j+1}: "))
                    output_coeffs.append(coeff)
                
                exponents = []
                for j in range(num_sectors):
                    exp = float(input(f"Input exponent for good {j+1}: "))
                    exponents.append(exp)
                
                # Create closure to capture parameters
                def make_prod_func(coeffs, exps):
                    def prod_func(inputs):
                        outputs = np.zeros(num_sectors)
                        for j in range(num_sectors):
                            # Simple Cobb-Douglas function
                            term = coeffs[j]
                            for k in range(num_sectors):
                                if inputs[k] > 0:  # Avoid zero inputs causing errors
                                    term *= inputs[k] ** exps[k]
                            outputs[j] = term
                        return outputs
                    return prod_func
                
                production_functions.append(make_prod_func(output_coeffs, exponents))
                
                # Labor function (also Cobb-Douglas)
                labor_coeff = float(input(f"Labor coefficient for sector {i+1}: "))
                labor_exp = float(input(f"Labor exponent for sector {i+1}: "))
                
                def make_labor_func(coeff, exp):
                    def labor_func(inputs):
                        return coeff * (np.sum(inputs) ** exp)
                    return labor_func
                
                labor_functions.append(make_labor_func(labor_coeff, labor_exp))
            
            # Get subsistence bundle
            print("\nEnter workers' subsistence bundle b (space-separated values):")
            b_input = input("Subsistence bundle: ")
            b = np.array([float(x) for x in b_input.split()])
            
            # Get initial endowments
            print("\nEnter initial endowments omega (space-separated values):")
            omega_input = input("Endowments: ")
            omega = np.array([float(x) for x in omega_input.split()])
            
            # Create and return model
            return ConvexProductionModel(production_functions, labor_functions, b, omega)
            
        elif model_type == "3":  # Social Determination Model
            from marxge.core.social_model import SocialDeterminationModel
            from marxge.utils.consumption import technological_needs_consumption, historical_moral_consumption
            
            # First create a base convex model
            base_model = ExampleModels.create_convex_two_sector_model()
            
            # Choose consumption function
            print("\nSelect consumption function:")
            print("1. Technological Needs Consumption")
            print("2. Historical-Moral Consumption")
            
            consumption_choice = input("Enter your choice (1-2): ")
            
            if consumption_choice == "1":
                consumption_function = technological_needs_consumption
            else:
                consumption_function = historical_moral_consumption
            
            # Get class power parameter
            
