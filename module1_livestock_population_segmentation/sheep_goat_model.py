"""
Sheep and Goats Intensification Prediction Model
Module 1: Population Segmentation for Sheep and Goats
Updated to match the latest full implementation
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
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
        """Load sheep-specific official data
        
        Returns combined yearbook (2004-2010) and official statistics (2010-2019)
        """
        # Official statistics for 2010-2019
        official_data = {
            2010: 22.9,  # 100 - 77.1
            2011: 25.0,  # 100 - 75.0
            2012: 28.3,  # 100 - 71.7
            2013: 31.1,  # 100 - 68.9
            2014: 34.3,  # 100 - 65.7
            2015: 36.7,  # 100 - 63.3
            2016: 37.9,  # 100 - 62.1
            2017: 38.7,  # 100 - 61.3
            2018: 38.0,  # 100 - 62.0
            2019: 40.7   # 100 - 59.3
        }
        
        return pd.DataFrame({
            'Year': list(official_data.keys()),
            'official_intensification_pct': list(official_data.values())
        })
    
    def engineer_features(self, df):
        """Sheep-specific feature engineering"""
        df = super().engineer_features(df)
        
        # Sheep-specific: grassland importance (key feature for sheep/goats)
        df['grassland_importance'] = df['CLCD_4'] / df['crop_sum'].replace(0, 0.001)
        df['grassland_importance'] = df['grassland_importance'].clip(0, 10)
        
        return df
    
    def get_feature_columns(self):
        """Define sheep-specific features - matching full implementation"""
        basic_features = [
            'crop_sum', 'population', 'CLCD_4', 'Longitude', 'Latitude',
            'CLCD_1', 'poi_count', 'travel_time'
        ]
        time_features = ['year_scaled', 'year_trend']
        
        # Note: grassland_importance is created but not used in final model
        # to match the validated feature set
        return basic_features + time_features
    
    def prepare_target_variable(self, df):
        """Prepare sheep target variable (intensification rate = 1 - ratio_1)"""
        return (1 - df['ratio_1']).values
    
    def get_calibration_periods(self):
        """Define sheep-specific calibration periods
        
        Updated to match full implementation:
        - Before 2010: Validation period (no calibration)
        - 2010-2013: Late transition period
        - 2014-2021: Growth period with year-specific factors
        """
        return {
            'transition_late': (2010, 2013),
            'growth': (2014, 2021)
        }
    
    def get_validation_cutoff_year(self):
        """Return the year that separates validation and calibration
        
        For sheep: Before 2010 for validation, 2010+ for calibration
        """
        return 2010
    
    def get_model_configs(self):
        """Model configurations for sheep - updated to match full implementation"""
        return {
            'GradientBoosting': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', GradientBoostingRegressor(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=4,
                        min_samples_leaf=5,
                        subsample=0.8,
                        random_state=self.config['random_state']
                    ))
                ]),
                'params': {}  # Using fixed params from validated model
            },
            'RandomForest': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', RandomForestRegressor(
                        n_estimators=200,
                        max_depth=10,
                        min_samples_leaf=5,
                        random_state=self.config['random_state'],
                        n_jobs=-1
                    ))
                ]),
                'params': {}
            },
            'ExtraTrees': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', ExtraTreesRegressor(
                        n_estimators=200,
                        max_depth=10,
                        min_samples_leaf=5,
                        random_state=self.config['random_state'],
                        n_jobs=-1
                    ))
                ]),
                'params': {}
            },
            'XGBoost': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', xgb.XGBRegressor(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=4,
                        subsample=0.8,
                        random_state=self.config['random_state']
                    ))
                ]),
                'params': {}
            }
        }
    
    def get_time_weights(self, years):
        """Create time-based sample weights for sheep model
        
        Enhanced weights for recent years with special emphasis on post-2010 data
        """
        min_year = np.min(years)
        max_year = np.max(years)
        
        # Base sigmoid weights
        normalized_years = (years - min_year) / (max_year - min_year)
        weights = 1 / (1 + np.exp(-6 * (normalized_years - 0.5)))
        weights = 0.3 + 0.7 * weights
        
        # Enhanced weights for post-2010 (calibration period)
        weights[years >= 2010] = weights[years >= 2010] * 3
        
        # Further enhanced weights for post-2015
        weights[years >= 2015] = weights[years >= 2015] * 1.5
        
        return weights
    
    def get_feature_display_names(self):
        """Get display names for features in visualizations"""
        return {
            'Longitude': 'Longitude',
            'Latitude': 'Latitude',
            'CLCD_1': 'Cropland Area',
            'CLCD_4': 'Grassland Area',
            'crop_sum': 'Crop Production',
            'population': 'Human Population',
            'poi_count': 'Sheep Farm Count',
            'travel_time': 'Travel Time to Cities',
            'year_trend': 'Time Trend',
            'year_scaled': 'Year (normalized)',
            'grassland_importance': 'Grassland Importance'
        }
    
    def get_scatter_color(self):
        """Return color for scatter plots"""
        return '#3498DB'  # Blue for sheep
    
    def get_title_prefix(self):
        """Return title prefix for plots"""
        return 'Sheep and Goats'


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'random_state': 42,
        'response_file': r'E:\livestock carrying capacitiy\livestock_mapping\ruminant_farm\Y_livestock\羊饲养规模年鉴.xlsx',
        'predictor_file': r'E:\livestock carrying capacitiy\livestock_mapping\pig_farm\x_predict_output_province_total.xlsx',
        'output_dir': 'Sheep_farm_figures'
    }
    
    # Initialize model
    model = SheepIntensificationModel(config)
    
    # Load and prepare data
    print("Loading data...")
    data = model.load_and_prepare_data()
    
    # Train model
    print("Training model...")
    results = model.train()
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(data['X_test'])
    
    # Calibrate predictions
    print("Calibrating predictions...")
    calibrated = model.calibrate_predictions()
    
    # Generate visualizations
    print("Generating visualizations...")
    model.create_visualizations()
    
    print("Done!")