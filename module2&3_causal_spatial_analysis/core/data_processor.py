# -*- coding: utf-8 -*-
"""
Data Processing Core Module
Unified processing for different configuration data loading, cleaning and preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings

class DataProcessor:
    """Unified data processor"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.processed_data = None
        
    def load_raw_data(self):
        """Load raw data"""
        print("1. Loading and preparing causal analysis data...")
        
        # Load response and predictor data
        response_data = pd.read_excel(self.config['data_paths']['response_data'])
        predictor_data = pd.read_excel(self.config['data_paths']['predictor_data'])
        
        print(f"   Response data shape: {response_data.shape}")
        print(f"   Predictor data shape: {predictor_data.shape}")
        
        return response_data, predictor_data
    
    def prepare_features(self, predictor_data):
        """Prepare feature data"""
        # Create base DataFrame
        X = pd.DataFrame({'year': predictor_data['year'], 'code': predictor_data['code']})
        
        # Add predictor variables
        available_vars = []
        for variable in self.config['variables']:
            if variable in predictor_data.columns:
                X[variable] = predictor_data[variable]
                available_vars.append(variable)
                print(f"  Added variable: {variable}")
            else:
                print(f"  Warning: Variable {variable} does not exist")
        
        # Remove duplicates
        X = X.drop_duplicates(subset=['year', 'code'], keep='first')
        
        return X, available_vars
    
    def merge_and_clean_data(self, X, response_data):
        """Merge and clean data"""
        response_variable = self.config['response_variable']
        
        # Merge data
        merged_data = pd.merge(X, response_data[['year', 'code', response_variable]], 
                              on=['year', 'code'], how='inner')
        print(f"Data after merging: {merged_data.shape}")
        
        # Remove missing values and duplicates
        merged_data = merged_data.dropna()
        merged_data['code'] = merged_data['code'].astype(str)
        merged_data = merged_data.drop_duplicates(subset=['year', 'code'], keep='first')
        print(f"After removing missing values and duplicates: {merged_data.shape}")
        
        return merged_data
    
    def apply_data_processing(self, merged_data):
        """Apply data processing rules"""
        response_variable = self.config['response_variable']
        processing_config = self.config['data_processing']
        
        # Unit conversion
        if processing_config['unit_conversion']:
            print(f"2. Applying unit conversion: × {processing_config['unit_conversion']}")
            merged_data['processed_response'] = merged_data[response_variable] * processing_config['unit_conversion']
            target_var = 'processed_response'
        else:
            target_var = response_variable
        
        # Data filtering
        filter_config = processing_config['filter_range']
        print(f"3. Data filtering: {filter_config['min']} <= {target_var} <= {filter_config['max']}")
        
        merged_data = merged_data[
            (merged_data[target_var] <= filter_config['max']) & 
            (merged_data[target_var] >= filter_config['min'])
        ]
        
        merged_data['original_response'] = merged_data[target_var]
        print(f"After data filtering: {merged_data.shape}")
        
        # Log transformation
        if processing_config['log_transform']:
            merged_data['log_livestock'] = np.log1p(merged_data[target_var])
            print("4. Applied log transformation")
        
        return merged_data
    
    def standardize_features(self, causal_data, available_features):
        """Standardize features"""
        print("5. Standardizing feature data...")
        
        # Check actually existing features
        available_feature_cols = [col for col in available_features if col in causal_data.columns]
        print(f"Available feature count: {len(available_feature_cols)}")
        
        # Standardize
        causal_data[available_feature_cols] = self.scaler.fit_transform(causal_data[available_feature_cols])
        
        return causal_data, available_feature_cols
    
    def create_interaction_terms(self, causal_data):
        """Create interaction terms"""
        print("6. Creating interaction terms...")
        
        interaction_terms = self.config['interaction_terms']
        created_terms = []
        
        for term in interaction_terms:
            if term == 'GDP_NDVI':
                if 'GDP' in causal_data.columns and 'NDVI' in causal_data.columns:
                    causal_data['GDP_NDVI'] = causal_data['GDP'] * causal_data['NDVI']
                    created_terms.append('GDP_NDVI')
                    print("  Created: GDP_NDVI")
                    
            elif term == 'presum_population_interaction':
                if 'presum_year' in causal_data.columns and 'population' in causal_data.columns:
                    causal_data['presum_population_interaction'] = causal_data['presum_year'] * causal_data['population']
                    created_terms.append('presum_population_interaction')
                    print("  Created: presum_population_interaction")
                    
            elif term == 'poi_GDP_interaction':
                if 'poi_count' in causal_data.columns and 'GDP' in causal_data.columns:
                    causal_data['poi_GDP_interaction'] = causal_data['poi_count'] * causal_data['GDP']
                    created_terms.append('poi_GDP_interaction')
                    print("  Created: poi_GDP_interaction")
                else:
                    print("  Skipped POI interaction: Required variables not available")
        
        return causal_data, created_terms
    
    def process_full_pipeline(self):
        """Execute complete data processing pipeline"""
        print(f"\n=== Starting data processing: {self.config['title']} ===")
        
        # 1. Load raw data
        response_data, predictor_data = self.load_raw_data()
        
        # 2. Prepare features
        X, available_vars = self.prepare_features(predictor_data)
        
        # 3. Merge and clean
        merged_data = self.merge_and_clean_data(X, response_data)
        
        # 4. Apply data processing
        processed_data = self.apply_data_processing(merged_data)
        
        # 5. Sort data
        causal_data = processed_data.sort_values(['code', 'year'])
        
        # 6. Standardize features
        causal_data, available_feature_cols = self.standardize_features(causal_data, available_vars)
        
        # 7. Create interaction terms
        causal_data, created_interaction_terms = self.create_interaction_terms(causal_data)
        
        # 8. Final feature list
        final_features = available_feature_cols + created_interaction_terms
        
        print(f"Final causal analysis sample size: {len(causal_data)}")
        print(f"Final feature count: {len(final_features)}")
        
        # Save processing results
        self.processed_data = {
            'causal_data': causal_data,
            'available_features': final_features,
            'available_original_features': available_feature_cols,
            'interaction_terms': created_interaction_terms,
            'scaler': self.scaler
        }
        
        return self.processed_data