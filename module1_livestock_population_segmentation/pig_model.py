"""
Pig Intensification Prediction Model
Module 1: Population Segmentation for Pig
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from base_model import BaseLivestockModel

class PigIntensificationModel(BaseLivestockModel):
    """Pig farming intensification prediction model"""
    
    def __init__(self, config=None):
        self.livestock_type = 'pig'
        super().__init__(config)
    
    def _load_official_data(self):
        """Load pig-specific official data"""
        official_data = {
            2000: 8.7, 2001: 8.2, 2002: 10.0, 2003: 10.7, 2004: 12.1,
            2005: 13.1, 2006: 15.0, 2007: 21.8, 2008: 27.3, 2009: 31.7,
            2010: 34.5, 2011: 36.6, 2012: 38.4, 2013: 40.8, 2014: 41.8,
            2015: 43.3, 2016: 44.9, 2017: 46.9, 2018: 49.1, 2019: 53.0,
            2020: 57.1, 2021: 62.0
        }
        
        return pd.DataFrame({
            'Year': list(official_data.keys()),
            'official_intensification_pct': list(official_data.values())
        })
    
    def get_feature_columns(self):
        """Define pig-specific features"""
        basic_features = ['crop_sum', 'population', 'CLCD_4', 'Longitude', 'Latitude',
                         'CLCD_1', 'poi_count', 'travel_time']
        time_features = ['year_scaled', 'year_trend']
        
        return basic_features + time_features
    
    def prepare_target_variable(self, df):
        """Prepare pig target variable (ratio_1 is already intensification rate)"""
        return df['ratio_1'].values
    
    def get_calibration_periods(self):
        """Define pig-specific calibration periods"""
        return {
            'early': (2000, 2006),
            'transition': (2007, 2010),
            'growth': (2011, 2021)
        }
    
    def get_model_configs(self):
        """Model configurations for pig"""
        return {
            'GradientBoosting': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', GradientBoostingRegressor(random_state=self.config['random_state']))
                ]),
                'params': {
                    'regressor__n_estimators': [100, 200],
                    'regressor__learning_rate': [0.05, 0.1, 0.2],
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