"""
Cattle Intensification Prediction Model
Module 1: Population Segmentation for Cattle
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from base_model import BaseLivestockModel

class CattleIntensificationModel(BaseLivestockModel):
    """Cattle farming intensification prediction model"""
    
    def __init__(self, config=None):
        self.livestock_type = 'cattle'
        super().__init__(config)
    
    def _load_official_data(self):
        """Load cattle-specific official data"""
        official_data = {
            2000: 19.1, 2001: 19.0, 2002: 19.0, 2003: 19.3, 2004: 18.9,
            2005: 18.4, 2006: 18.0, 2007: 16.1, 2008: 19.4, 2009: 21.3,
            2010: 25.3, 2011: 28.7, 2012: 31.8, 2013: 34.8, 2014: 37.3,
            2015: 38.5, 2016: 39.3, 2017: 38.6, 2018: 39.2, 2019: 40.0,
            2020: 41.0, 2021: 41.4
        }
        
        return pd.DataFrame({
            'Year': list(official_data.keys()),
            'official_intensification_pct': list(official_data.values())
        })
    
    def engineer_features(self, df):
        """Cattle-specific feature engineering"""
        df = super().engineer_features(df)
        
        # Cattle-specific: grassland importance (cattle need more grassland)
        df['grassland_importance'] = df['CLCD_4'] / df['crop_sum'].replace(0, 0.001)
        df['grassland_importance'] = df['grassland_importance'].clip(0, 10)
        
        return df
    
    def get_feature_columns(self):
        """Define cattle-specific features"""
        basic_features = ['crop_sum', 'population', 'CLCD_4', 'Longitude', 'Latitude',
                         'CLCD_1', 'poi_count', 'travel_time']
        time_features = ['year_scaled', 'year_trend']
        
        return basic_features + time_features
    
    def prepare_target_variable(self, df):
        """Prepare cattle target variable (intensification rate = 1 - ratio_1)"""
        return (1 - df['ratio_1']).values
    
    def get_calibration_periods(self):
        """Define cattle-specific calibration periods"""
        return {
            'early': (2000, 2006),
            'transition': (2007, 2010),
            'growth': (2011, 2021)
        }
    
    def get_model_configs(self):
        """Model configurations for cattle"""
        return {
            'GradientBoosting': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', GradientBoostingRegressor(random_state=self.config['random_state']))
                ]),
                'params': {
                    'regressor__n_estimators': [100, 200],
                    'regressor__learning_rate': [0.05, 0.1],
                    'regressor__max_depth': [4, 6],
                    'regressor__min_samples_leaf': [3, 5]
                }
            },
            'RandomForest': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', RandomForestRegressor(random_state=self.config['random_state']))
                ]),
                'params': {
                    'regressor__n_estimators': [100, 200],
                    'regressor__max_depth': [8, 12],
                    'regressor__min_samples_leaf': [2, 5]
                }
            },
            'XGBoost': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', xgb.XGBRegressor(random_state=self.config['random_state']))
                ]),
                'params': {
                    'regressor__n_estimators': [100, 200],
                    'regressor__learning_rate': [0.05, 0.1],
                    'regressor__max_depth': [4, 6]
                }
            }
        }