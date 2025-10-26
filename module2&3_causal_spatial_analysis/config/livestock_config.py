# -*- coding: utf-8 -*-
"""
Livestock Analysis Configuration Management
Unified management of differences across species and production systems
"""

import os
from .variable_mappings import VARIABLE_MAPPINGS, CATEGORY_COLORS

# Base variable sets
MIXED_FARMING_VARIABLES = [
    'population', 'GDP', 'snow_cover', 'travel_time', 'dem', 'tmpmean_year', 
    'presum_year', 'NDVI', 'PH', 'SOC', 'N', 'OSM', 'NTL', 'GSDL', 'wind'
]

LANDLESS_VARIABLES = MIXED_FARMING_VARIABLES + [
    'company_type_sum', 'poi_count', 'total_capital'  # POI variables
]

# Interaction terms configuration
INTERACTION_TERMS = {
    'basic': ['GDP_NDVI', 'presum_population_interaction'],
    'poi': ['poi_GDP_interaction']  # Only for Landless systems
}

# Main configuration dictionary
LIVESTOCK_CONFIGS = {
    # ==================== LANDLESS SYSTEMS ====================
    'pig_landless': {
        'species': 'pig',
        'system': 'landless',
        'title': 'Pigs in the Landless LPS',
        
        # Data file configuration
        'data_paths': {
            'response_data': r'E:\livestock carrying capacitiy\livestock_mapping\pig_farm\Y_livestock\pig_data_county_based_density.xlsx',
            'predictor_data': r'E:\livestock carrying capacitiy\livestock_mapping\pig_farm\x_predict_output_pig_farm_wide2.xlsx',
            'spatial_base_path': r"E:/livestock carrying capacitiy/livestock_mapping/pig_farm/X_resampled_cropped/",
            'reference_file': r'E:/livestock carrying capacitiy/livestock_mapping/pig_farm/X_resampled_cropped/population/resample1km_landscan-global-2021_re.tif'
        },
        
        # Variable configuration
        'response_variable': 'landless_pig_density',
        'variables': LANDLESS_VARIABLES,
        'interaction_terms': INTERACTION_TERMS['basic'] + INTERACTION_TERMS['poi'],
        
        # Data processing configuration
        'data_processing': {
            'filter_range': {'max': 300, 'min': 0.001},  # Fixed: 300 not 500
            'unit_conversion': None,  # No unit conversion needed
            'log_transform': True
        },
        
        # Output configuration
        'output_paths': {
            'visualizations': "E:/livestock carrying capacitiy/livestock_mapping/pig_farm/Pig_Landless_visualizations/",
            'models': "E:/livestock carrying capacitiy/livestock_mapping/pig_farm/Stacking_model/trained_models/",
            'spatial_predictions': "E:/livestock carrying capacitiy/livestock_mapping/pig_farm/predictions_Pig_Landless/"
        },
        
        'model_name': 'Landless_Pig_stacking_model_dml.pkl'
    },
    
    'cattle_landless': {
        'species': 'cattle',
        'system': 'landless', 
        'title': 'Cattle in the Landless LPS',
        
        'data_paths': {
            'response_data': r'E:\livestock carrying capacitiy\livestock_mapping\ruminant_farm\Y_livestock\Cattle_data_county_based_density_shapefile.xlsx',
            'predictor_data': r'E:\livestock carrying capacitiy\livestock_mapping\ruminant_farm\x_predict_output_cattle_sheep_wide_all_vars_with_NTL.xlsx',
            'spatial_base_path': r"E:/livestock carrying capacitiy/livestock_mapping/ruminant_farm/X_resampled_cropped/",
            'reference_file': r'E:/livestock carrying capacitiy/livestock_mapping/ruminant_farm/X_resampled_cropped/population/resample1km_landscan-global-2021_re.tif'
        },
        
        'response_variable': 'landless_牛_density',
        'variables': LANDLESS_VARIABLES,
        'interaction_terms': INTERACTION_TERMS['basic'] + INTERACTION_TERMS['poi'],
        
        'data_processing': {
            'filter_range': {'max': 500, 'min': 0.001},
            'unit_conversion': None,
            'log_transform': True
        },
        
        'output_paths': {
            'visualizations': "E:/livestock carrying capacitiy/livestock_mapping/ruminant_farm/Cattle_Landless_visualizations/",
            'models': "E:/livestock carrying capacitiy/livestock_mapping/ruminant_farm/Stacking_model/trained_models/",
            'spatial_predictions': "E:/livestock carrying capacitiy/livestock_mapping/ruminant_farm/predictions_Cattle_Landless/"
        },
        
        'model_name': 'Landless_Cattle_stacking_model_dml.pkl'
    },
    
    'sheep_landless': {
        'species': 'sheep',
        'system': 'landless',
        'title': 'Sheep and Goats in the Landless LPS',
        
        'data_paths': {
            'response_data': r'E:\livestock carrying capacitiy\livestock_mapping\ruminant_farm\Y_livestock\Sheep_data_county_based_density_shapefile.xlsx',
            'predictor_data': r'E:\livestock carrying capacitiy\livestock_mapping\ruminant_farm\x_predict_output_cattle_sheep_wide_all_vars_with_NTL.xlsx',
            'spatial_base_path': r"E:/livestock carrying capacitiy/livestock_mapping/ruminant_farm/X_resampled_cropped/",
            'reference_file': r'E:/livestock carrying capacitiy/livestock_mapping/ruminant_farm/X_resampled_cropped/population/resample1km_landscan-global-2021_re.tif'
        },
        
        'response_variable': 'landless_羊_density',
        'variables': LANDLESS_VARIABLES,
        'interaction_terms': INTERACTION_TERMS['basic'] + INTERACTION_TERMS['poi'],
        
        'data_processing': {
            'filter_range': {'max': 500, 'min': 0.001},
            'unit_conversion': None,
            'log_transform': True
        },
        
        'output_paths': {
            'visualizations': "E:/livestock carrying capacitiy/livestock_mapping/ruminant_farm/Sheep_Landless_visualizations/",
            'models': "E:/livestock carrying capacitiy/livestock_mapping/ruminant_farm/Stacking_model/trained_models/",
            'spatial_predictions': "E:/livestock carrying capacitiy/livestock_mapping/ruminant_farm/predictions_Sheep_Landless/"
        },
        
        'model_name': 'Landless_Sheep_stacking_model_dml.pkl'
    },
    
    # ==================== MIXED-FARMING SYSTEMS ====================
    'pig_mixed': {
        'species': 'pig',
        'system': 'mixed_farming',
        'title': 'Pigs in the Mixed-farming LPS',
        
        'data_paths': {
            'response_data': r'E:\livestock carrying capacitiy\livestock_mapping\pig_farm\Y_livestock\pig_data_county_based_density2.xlsx',
            'predictor_data': r'E:\livestock carrying capacitiy\livestock_mapping\pig_farm\x_predict_output_merged_0414.xlsx',
            'spatial_base_path': r"E:/livestock carrying capacitiy/livestock_mapping/pig_farm/X_resampled_cropped/",
            'reference_file': r'E:/livestock carrying capacitiy/livestock_mapping/pig_farm/X_resampled_cropped/population/resample1km_landscan-global-2021_re.tif'
        },
        
        'response_variable': 'cropland_pig_density',
        'variables': MIXED_FARMING_VARIABLES,  # No POI variables
        'interaction_terms': INTERACTION_TERMS['basic'],  # No POI interactions
        
        'data_processing': {
            'filter_range': {'max': 100000, 'min': 0.1},
            'unit_conversion': 10000,  # 10,000 head → head/km²
            'log_transform': True
        },
        
        'output_paths': {
            'visualizations': "E:/livestock carrying capacitiy/livestock_mapping/pig_farm/Pig_Mixed_visualizations/",
            'models': "E:/livestock carrying capacitiy/livestock_mapping/pig_farm/Stacking_model/trained_models/",
            'spatial_predictions': "E:/livestock carrying capacitiy/livestock_mapping/pig_farm/predictions_Pig_Mixed/"
        },
        
        'model_name': 'Mixed_Pig_stacking_model_dml.pkl'
    },
    
    'cattle_mixed': {
        'species': 'cattle',
        'system': 'mixed_farming',
        'title': 'Cattle in the Mixed-farming LPS',
        
        'data_paths': {
            'response_data': r'E:\livestock carrying capacitiy\livestock_mapping\ruminant_farm\Y_livestock\Cattle_mixed_data_county_based_density.xlsx',
            'predictor_data': r'E:\livestock carrying capacitiy\livestock_mapping\ruminant_farm\x_predict_output_cattle_mixed.xlsx',
            'spatial_base_path': r"E:/livestock carrying capacitiy/livestock_mapping/ruminant_farm/X_resampled_cropped/",
            'reference_file': r'E:/livestock carrying capacitiy/livestock_mapping/ruminant_farm/X_resampled_cropped/population/resample1km_landscan-global-2021_re.tif'
        },
        
        'response_variable': 'cropland_cattle_density',
        'variables': MIXED_FARMING_VARIABLES,  # No POI variables
        'interaction_terms': INTERACTION_TERMS['basic'],  # No POI interactions
        
        'data_processing': {
            'filter_range': {'max': 50000, 'min': 0.1},
            'unit_conversion': 10000,  # 10,000 head → head/km²
            'log_transform': True
        },
        
        'output_paths': {
            'visualizations': "E:/livestock carrying capacitiy/livestock_mapping/ruminant_farm/Cattle_Mixed_visualizations/",
            'models': "E:/livestock carrying capacitiy/livestock_mapping/ruminant_farm/Stacking_model/trained_models/",
            'spatial_predictions': "E:/livestock carrying capacitiy/livestock_mapping/ruminant_farm/predictions_Cattle_Mixed/"
        },
        
        'model_name': 'Mixed_Cattle_stacking_model_dml.pkl'
    },
    
    'sheep_mixed': {
        'species': 'sheep',
        'system': 'mixed_farming',
        'title': 'Sheep and Goats in the Mixed-farming LPS',
        
        'data_paths': {
            'response_data': r'E:\livestock carrying capacitiy\livestock_mapping\ruminant_farm\Y_livestock\Sheep_mixed_data_county_based_density.xlsx',
            'predictor_data': r'E:\livestock carrying capacitiy\livestock_mapping\ruminant_farm\x_predict_output_sheep_mixed.xlsx',
            'spatial_base_path': r"E:/livestock carrying capacitiy/livestock_mapping/ruminant_farm/X_resampled_cropped/",
            'reference_file': r'E:/livestock carrying capacitiy/livestock_mapping/ruminant_farm/X_resampled_cropped/population/resample1km_landscan-global-2021_re.tif'
        },
        
        'response_variable': 'cropland_sheep_density',
        'variables': MIXED_FARMING_VARIABLES,  # No POI variables
        'interaction_terms': INTERACTION_TERMS['basic'],  # No POI interactions
        
        'data_processing': {
            'filter_range': {'max': 80000, 'min': 0.1},
            'unit_conversion': 10000,  # 10,000 head → head/km²
            'log_transform': True
        },
        
        'output_paths': {
            'visualizations': "E:/livestock carrying capacitiy/livestock_mapping/ruminant_farm/Sheep_Mixed_visualizations/",
            'models': "E:/livestock carrying capacitiy/livestock_mapping/ruminant_farm/Stacking_model/trained_models/",
            'spatial_predictions': "E:/livestock carrying capacitiy/livestock_mapping/ruminant_farm/predictions_Sheep_Mixed/"
        },
        
        'model_name': 'Mixed_Sheep_stacking_model_dml.pkl'
    }
}

def get_config(config_key):
    """Get specified configuration"""
    if config_key not in LIVESTOCK_CONFIGS:
        raise ValueError(f"Configuration key '{config_key}' does not exist. Available configs: {list(LIVESTOCK_CONFIGS.keys())}")
    
    config = LIVESTOCK_CONFIGS[config_key].copy()
    
    # Auto-create output directories
    for path_type, path in config['output_paths'].items():
        os.makedirs(path, exist_ok=True)
    
    return config

def list_available_configs():
    """List all available configurations"""
    return list(LIVESTOCK_CONFIGS.keys())

def get_configs_by_system(system_type):
    """Get configuration list by production system type"""
    return [key for key, config in LIVESTOCK_CONFIGS.items() if config['system'] == system_type]

def get_configs_by_species(species):
    """Get configuration list by livestock species"""
    return [key for key, config in LIVESTOCK_CONFIGS.items() if config['species'] == species]