#!/usr/bin/env python
"""
MarxGE: Marxian General Equilibrium Model - Main Application

This script provides a command-line interface to the MarxGE package.
"""

import argparse
import sys
import os
import matplotlib.pyplot as plt

# Set Bayesian Methods for Hackers style
plt.style.use('bmh')


def main():
    """Main application function."""
    parser = argparse.ArgumentParser(description='MarxGE: Marxian General Equilibrium Model')
    
    parser.add_argument('--mode', choices=['interactive', 'batch'], default='interactive',
                        help='Mode of operation: interactive or batch')
    
    parser.add_argument('--model', choices=['linear_corn', 'linear_two_sector', 'linear_three_sector',
                                          'convex_two_sector', 'joint_production', 'social_determination'],
                        default='linear_two_sector',
                        help='Type of model to use (for batch mode)')
    
    parser.add_argument('--experiment', choices=['fmt', 'ge', 'social', 'all'],
                        default='all',
                        help='Type of experiment to run (for batch mode)')
    
    parser.add_argument('--output', type=str, default=None,
                        help='Directory to save output files')
    
    args = parser.parse_args()
    
    if args.mode == 'interactive':
        # Import necessary modules for interactive mode
        from marxge.experiments.runner import MarxianExperiments
        
        # Run interactive experiment
        MarxianExperiments.run_interactive_experiment()
        
    else:  # batch mode
        # Import necessary modules for batch mode
        from marxge.experiments.runner import MarxianExperiments
        from marxge.utils.examples import ExampleModels
        
        # Create output directory if specified
        if args.output:
            os.makedirs(args.output, exist_ok=True)
        
        # Run specified experiment
        if args.experiment == 'fmt':
            # Create model
            if args.model == 'linear_corn':
                model = ExampleModels.create_linear_corn_model()
            elif args.model == 'linear_two_sector':
                model = ExampleModels.create_linear_two_sector_model()
            elif args.model == 'linear_three_sector':
                model = ExampleModels.create_linear_three_sector_model()
            elif args.model == 'convex_two_sector':
                model = ExampleModels.create_convex_two_sector_model()
            elif args.model == 'joint_production':
                model = ExampleModels.create_joint_production_model()
            elif args.model == 'social_determination':
                model = ExampleModels.create_social_determination_model()
            
            # Run FMT experiment
            # Run FMT experiment
            subsistence_values = [0.5, 0.75, 1.0, 1.25, 1.5]
            fmt_results = MarxianExperiments.run_fmt_experiment(model, 'subsistence', subsistence_values)
            
            # Save figure if specified
            if args.output:
                fmt_results['figure'].savefig(os.path.join(args.output, f"{args.model}_fmt_subsistence.png"))
                print(f"Results saved to {os.path.join(args.output, f'{args.model}_fmt_subsistence.png')}")
            
            plt.show()
            
        elif args.experiment == 'ge':
            # Create model
            if args.model == 'linear_corn':
                model = ExampleModels.create_linear_corn_model()
            elif args.model == 'linear_two_sector':
                model = ExampleModels.create_linear_two_sector_model()
            elif args.model == 'linear_three_sector':
                model = ExampleModels.create_linear_three_sector_model()
            elif args.model == 'convex_two_sector':
                model = ExampleModels.create_convex_two_sector_model()
            elif args.model == 'joint_production':
                model = ExampleModels.create_joint_production_model()
            elif args.model == 'social_determination':
                model = ExampleModels.create_social_determination_model()
            
            # Run general equilibrium experiment
            subsistence_values = [0.5, 0.75, 1.0, 1.25, 1.5]
            ge_results = MarxianExperiments.run_general_equilibrium_experiment(model, 'subsistence', subsistence_values)
            
            # Save figure if specified
            if args.output:
                ge_results['figure'].savefig(os.path.join(args.output, f"{args.model}_ge_subsistence.png"))
                print(f"Results saved to {os.path.join(args.output, f'{args.model}_ge_subsistence.png')}")
            
            plt.show()
            
        elif args.experiment == 'social':
            # Create social determination model (convert if needed)
            if args.model == 'social_determination':
                model = ExampleModels.create_social_determination_model()
            elif args.model in ['convex_two_sector', 'joint_production']:
                base_model = ExampleModels.create_convex_two_sector_model() if args.model == 'convex_two_sector' else ExampleModels.create_joint_production_model()
                model = ExampleModels.create_social_determination_model(base_model)
            else:
                print("Creating a default social determination model...")
                model = ExampleModels.create_social_determination_model()
            
            # Run social determination experiment
            class_power_trajectory = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.55, 0.5, 0.45]
            social_results = MarxianExperiments.run_social_determination_experiment(model, 10, class_power_trajectory)
            
            # Save figure if specified
            if args.output:
                social_results['figure'].savefig(os.path.join(args.output, f"{args.model}_social_determination.png"))
                print(f"Results saved to {os.path.join(args.output, f'{args.model}_social_determination.png')}")
            
            plt.show()
            
        else:  # 'all'
            # Run all experiments
            results = MarxianExperiments.run_all_experiments(args.model, args.output)
            
            # Print summary
            print("\nAll Experiments Completed!")
            print(f"Exploitation Rate: {results['exploitation_rate']:.4f}")
            print(f"Profit Rate: {results['profit_rate']:.4f}")
            print(f"Independence Assumption (A7) holds: {results['a7_test']['assumption_holds']}")
            
            if args.output:
                print(f"\nAll figures saved to {args.output}")


if __name__ == "__main__":
    main()
