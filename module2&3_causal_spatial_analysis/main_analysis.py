# -*- coding: utf-8 -*-
"""
Main Analysis Execution Script
Example usage of the modular livestock analysis system

"""
import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import modules
from config.livestock_config import list_available_configs, get_configs_by_system, get_configs_by_species
from analysis.livestock_analyzer import LivestockAnalyzer

def main():
    """
    Main execution function demonstrating the modular analysis system
    """
    
    print("="*80)
    print("MODULAR LIVESTOCK CAUSAL ANALYSIS SYSTEM")
    print("Configuration-driven analysis for all species and production systems")
    print("="*80)
    
    # List all available configurations
    print("\nAvailable Configurations:")
    configs = list_available_configs()
    for i, config in enumerate(configs, 1):
        print(f"  {i}. {config}")
    
    # Show configurations by system
    print(f"\nLandless System Configurations:")
    landless_configs = get_configs_by_system('landless')
    for config in landless_configs:
        print(f"  - {config}")
    
    print(f"\nMixed-farming System Configurations:")
    mixed_configs = get_configs_by_system('mixed_farming')
    for config in mixed_configs:
        print(f"  - {config}")
    
    # Example 1: Analyze Sheep in Landless system
    print("\n" + "="*60)
    print("EXAMPLE 1: Sheep and Goats in Landless LPS")
    print("="*60)
    
    try:
        sheep_analyzer = LivestockAnalyzer('sheep_landless')
        sheep_results = sheep_analyzer.run_full_analysis()
        print("Sheep landless analysis completed successfully!")
        
    except Exception as e:
        print(f"Sheep landless analysis failed: {e}")
    
    # Example 2: Analyze Pig in Mixed-farming system
    print("\n" + "="*60)
    print("EXAMPLE 2: Pigs in Mixed-farming LPS")
    print("="*60)
    
    try:
        pig_analyzer = LivestockAnalyzer('pig_mixed')
        pig_results = pig_analyzer.run_full_analysis()
        print(" Pig mixed-farming analysis completed successfully!")
        
    except Exception as e:
        print(f"Pig mixed-farming analysis failed: {e}")
    
    # Example 3: Run analysis for specific configuration
    print("\n" + "="*60)
    print("EXAMPLE 3: Cattle in Landless LPS")
    print("="*60)
    
    try:
        cattle_analyzer = LivestockAnalyzer('cattle_landless')
        
        # You can also run individual stages
        print("Running Stage 1 only...")
        cattle_analyzer.stage_1_causal_inference_analysis()
        
        print("Running Stage 2 only...")
        cattle_analyzer.stage_2_prediction_analysis()
        
        print("Running Stage 3 only...")
        cattle_analyzer.stage_3_spatial_prediction()
        
        print("Cattle landless analysis completed successfully!")
        
    except Exception as e:
        print(f"Cattle landless analysis failed: {e}")
    
    print("\n" + "="*80)
    print("MODULAR ANALYSIS SYSTEM DEMONSTRATION COMPLETED")
    print("Key Benefits:")
    print(" One codebase for all species and systems")
    print(" Configuration-driven approach eliminates code duplication")
    print(" Consistent academic methodology across all analyses")
    print(" Easy to extend to new species and systems")
    print(" Professional English documentation")
    print("="*80)

if __name__ == "__main__":
    main()