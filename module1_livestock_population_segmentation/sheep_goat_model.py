"""
Sheep and Goats Intensification Prediction Model
Module 1: Population Segmentation for Sheep and Goats
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from base_model import BaseLivestockModel

class SheepIntensificationModel(BaseLivestockModel):
    """Sheep and goats farming intensification prediction model"""
    
    def __init__(self, config=None):
        self.livestock_type = 'sheep'
        super().__init__(config)
    
    def _load_official_data(self):
        """Load sheep-specific official data"""
        official_data = {
            2010: 22.9, 2011: 25.0, 2012: 28.3, 2013: 31.1, 2014: 34.3,
            2015: 36.7, 2016: 37.9, 2017: 38.7, 2018: 38.0, 2019: 40.7
        }
        
        return pd.DataFrame({
            'Year': list(official_data.keys()),
            'official_intensification_pct': list(official_data.values())
        })
    
    def engineer_features(self, df):
        """Sheep-specific feature engineering"""
        df = super().engineer_features(df)
        
        # Sheep-specific: grassland importance
        df['grassland_importance'] = df['CLCD_4'] / df['crop_sum'].replace(0, 0.001)
        df['grassland_importance'] = df['grassland_importance'].clip(0, 10)
        
        return df
    
    def get_feature_columns(self):
        """Define sheep-specific features"""
        basic_features = ['crop_sum', 'population', 'CLCD_4', 'Longitude', 'Latitude',
                         'CLCD_1', 'poi_count', 'travel_time']
        time_features = ['year_scaled', 'year_trend']
        
        return basic_features + time_features
    
    def prepare_target_variable(self, df):
        """Prepare sheep target variable (intensification rate = 1 - ratio_1)"""
        return (1 - df['ratio_1']).values
    
    def get_calibration_periods(self):
        """Define sheep-specific calibration periods"""
        return {
            'early': (2004, 2010),
            'transition': (2011, 2015),
            'growth': (2016, 2021)
        }
    
    def get_model_configs(self):
        """Model configurations for sheep"""
        return {
            'GradientBoosting': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', GradientBoostingRegressor(random_state=self.config['random_state']))
                ]),
                'params': {
                    'regressor__n_estimators': [100, 200],
                    'regressor__learning_rate': [0.05, 0.1, 0.2],
                    'regressor__max_depth': [3, 4, 6],
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
                    'regressor__max_depth': [6, 10, 12],
                    'regressor__min_samples_leaf': [2, 3, 5]
                }
            },
            'XGBoost': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', xgb.XGBRegressor(random_state=self.config['random_state']))
                ]),
                'params': {
                    'regressor__n_estimators': [100, 200],
                    'regressor__learning_rate': [0.01, 0.05, 0.1],
                    'regressor__max_depth': [3, 4, 5]
                }
            }
        }