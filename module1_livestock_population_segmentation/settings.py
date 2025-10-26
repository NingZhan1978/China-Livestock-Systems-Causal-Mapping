"""
Configuration settings for livestock intensification models
"""

# Base configuration
BASE_CONFIG = {
    'random_state': 42,
    'n_cv_folds': 5,
    'validation_years': 2,
    'output_dir': 'results'
}

# Data paths configuration
DATA_PATHS = {
    'pig': {
        'response': 'data/pig_response.xlsx',
        'predictor': 'data/predictors.xlsx',
        'county': 'data/county_data.xlsx'
    },
    'cattle': {
        'response': 'data/cattle_response.xlsx',
        'predictor': 'data/predictors.xlsx',
        'county': 'data/county_data.xlsx'
    },
    'sheep': {
        'response': 'data/sheep_response.xlsx',
        'predictor': 'data/predictors.xlsx',
        'county': 'data/county_data.xlsx'
    }
}

# Model-specific configurations
MODEL_CONFIGS = {
    'pig': {
        'output_subdir': 'pig_results',
        'title': 'Pig Farming Intensification'
    },
    'cattle': {
        'output_subdir': 'cattle_results',
        'title': 'Cattle Farming Intensification'
    },
    'sheep': {
        'output_subdir': 'sheep_results',
        'title': 'Sheep and Goats Farming Intensification'
    }
}