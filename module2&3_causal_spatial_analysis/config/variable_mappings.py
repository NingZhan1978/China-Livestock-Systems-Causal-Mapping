# -*- coding: utf-8 -*-
"""
Variable mappings and visualization configurations
"""

# Variable display name mappings
VARIABLE_MAPPINGS = {
    'population': 'POP',
    'travel_time': 'Travel Time',
    'GDP': 'GDP',
    'dem': 'DEM',
    'GSDL': 'GHSL',
    'OSM': 'OSM',
    'NTL': 'NTL',
    'snow_cover': 'Snow Cover',
    'tmpmean_year': 'Annual Temp',
    'presum_year': 'Annual Precip',
    'NDVI': 'NDVI',
    'PH': 'pH',
    'SOC': 'SOC',
    'N': 'N',
    'wind': 'Wind',
    
    # POI variables (Landless systems only)
    'company_type_sum': 'Company Types',
    'poi_count': 'POI Count',
    'total_capital': 'Total Capital',
    
    # Interaction terms
    'GDP_NDVI': 'GDP×NDVI',
    'presum_population_interaction': 'Precip×Pop',
    'poi_GDP_interaction': 'POI×GDP'
}

# Variable category mappings
VARIABLE_CATEGORIES = {
    'POP': 'Social-economy',
    'Travel Time': 'Social-economy',
    'GDP': 'Social-economy',
    'DEM': 'Environment',
    'GHSL': 'Social-economy',
    'OSM': 'Social-economy',
    'NTL': 'Social-economy',
    'Snow Cover': 'Environment',
    'Annual Temp': 'Environment',
    'Annual Precip': 'Environment',
    'NDVI': 'Environment',
    'pH': 'Environment',
    'SOC': 'Environment',
    'N': 'Environment',
    'Wind': 'Environment',
    
    # POI variables
    'Company Types': 'POI-Business',
    'POI Count': 'POI-Business',
    'Total Capital': 'POI-Business',
    
    # Interaction terms
    'GDP×NDVI': 'Causal-Interaction',
    'Precip×Pop': 'Causal-Interaction',
    'POI×GDP': 'Causal-Interaction'
}

# Category color configurations
CATEGORY_COLORS = {
    'Environment': '#2ca25f',           # Green
    'Social-economy': '#fd8d3c',        # Orange
    'POI-Business': '#756bb1',          # Purple (Landless systems only)
    'Causal-Interaction': '#e41a1c',    # Red
    'Target': '#e41a1c',                # Red
    'Other': '#cccccc'                  # Gray
}

def get_system_categories(system_type):
    """Get applicable categories by system type"""
    if system_type == 'landless':
        return CATEGORY_COLORS
    elif system_type == 'mixed_farming':
        # Mixed-farming systems don't include POI-Business category
        return {k: v for k, v in CATEGORY_COLORS.items() if k != 'POI-Business'}
    else:
        raise ValueError(f"Unknown system type: {system_type}")

def get_variable_category(var_name, system_type):
    """Get variable category, system-specific"""
    mapped_name = VARIABLE_MAPPINGS.get(var_name, var_name)
    category = VARIABLE_CATEGORIES.get(mapped_name, 'Other')
    
    # If mixed-farming system and variable is POI category, return Other
    if system_type == 'mixed_farming' and category == 'POI-Business':
        return 'Other'
    
    return category