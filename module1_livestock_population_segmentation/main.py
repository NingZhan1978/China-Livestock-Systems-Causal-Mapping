"""
Main script to run livestock intensification analysis
Module 1: Population Segmentation for Multiple Livestock Types
"""

import os
import argparse
from settings import BASE_CONFIG, DATA_PATHS, MODEL_CONFIGS
from pig_model import PigIntensificationModel
from cattle_model import CattleIntensificationModel
from sheep_model import SheepIntensificationModel

def main():
    """Main function to run livestock intensification analysis"""
    parser = argparse.ArgumentParser(description='Livestock Intensification Analysis')
    parser.add_argument('--livestock', choices=['pig', 'cattle', 'sheep', 'all'], 
                       default='all', help='Livestock type to analyze')
    parser.add_argument('--output_dir', default='results', help='Output directory')
    
    args = parser.parse_args()
    
    # Update configuration
    config = BASE_CONFIG.copy()
    config['output_dir'] = args.output_dir
    
    livestock_models = {
        'pig': PigIntensificationModel,
        'cattle': CattleIntensificationModel,
        'sheep': SheepIntensificationModel
    }
    
    if args.livestock == 'all':
        livestock_types = ['pig', 'cattle', 'sheep']
    else:
        livestock_types = [args.livestock]
    
    results = {}
    
    for livestock_type in livestock_types:
        print(f"\n{'='*60}")
        print(f"Running {livestock_type.upper()} Analysis")
        print(f"{'='*60}")
        
        # Create livestock-specific output directory
        livestock_config = config.copy()
        livestock_output_dir = os.path.join(
            config['output_dir'], 
            MODEL_CONFIGS[livestock_type]['output_subdir']
        )
        livestock_config['output_dir'] = livestock_output_dir
        
        # Initialize model
        model_class = livestock_models[livestock_type]
        model = model_class(livestock_config)
        
        # Get data paths
        data_paths = DATA_PATHS[livestock_type]
        
        try:
            # Run analysis
            best_model, predictions, calibration_factors = model.run_analysis(
                data_paths['response'], 
                data_paths['predictor'], 
                data_paths['county']
            )
            
            results[livestock_type] = {
                'model': best_model,
                'predictions': predictions,
                'calibration_factors': calibration_factors
            }
            
            print(f"\n{livestock_type.upper()} analysis completed successfully!")
            print(f"Results saved to: {livestock_output_dir}")
            
        except Exception as e:
            print(f"Error in {livestock_type} analysis: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("All analyses completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")
    
    return results

if __name__ == "__main__":
    results = main()