"""
Cattle Intensification Prediction Model
Module 1: Population Segmentation for Cattle
Updated to match the latest full implementation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
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
        """Load cattle-specific official data (2000-2021)"""
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
        """Define cattle-specific features - matching full implementation"""
        basic_features = [
            'crop_sum', 'population', 'CLCD_4', 'Longitude', 'Latitude',
            'CLCD_1', 'poi_count', 'travel_time'
        ]
        time_features = ['year_scaled', 'year_trend']
        
        # Note: grassland_importance is created but not used in final model
        # to match the validated feature set
        return basic_features + time_features
    
    def prepare_target_variable(self, df):
        """Prepare cattle target variable (intensification rate = 1 - ratio_1)"""
        return (1 - df['ratio_1']).values
    
    def get_calibration_periods(self):
        """Define cattle-specific calibration periods
        
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
        
        For cattle: Before 2010 for validation, 2010+ for calibration
        """
        return 2010
    
    def get_model_configs(self):
        """Model configurations for cattle - updated to match full implementation"""
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
        """Create time-based sample weights for cattle model
        
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
            'poi_count': 'Cattle Farm Count',
            'travel_time': 'Travel Time to Cities',
            'year_trend': 'Time Trend',
            'year_scaled': 'Year (normalized)',
            'grassland_importance': 'Grassland Importance'
        }
    
    def get_scatter_color(self):
        """Return color for scatter plots"""
        return '#C0392B'  # Red for cattle
    
    def get_title_prefix(self):
        """Return title prefix for plots"""
        return 'Cattle'
    
    def calibrate_predictions_post_2010(self, comparison_df):
        """Custom calibration logic for cattle using only post-2010 data
        
        This method overrides the base calibration to implement the specific
        strategy: validation before 2010, calibration from 2010 onwards
        """
        # Filter data from 2010 and after for calibration
        calibration_data = comparison_df[
            comparison_df['Year'] >= 2010
        ].dropna(subset=['official_intensification_pct']).copy()
        
        if len(calibration_data) == 0:
            print("No data from 2010 and after for calibration")
            return {}
        
        print(f"\nUsing {len(calibration_data)} data points from 2010 and after for calibration:")
        
        # Define calibration periods
        periods = self.get_calibration_periods()
        period_factors = {}
        
        for period_name, (start_year, end_year) in periods.items():
            period_data = calibration_data[
                (calibration_data['Year'] >= start_year) & 
                (calibration_data['Year'] <= end_year)
            ].copy()
            
            if len(period_data) > 0:
                official_values = period_data['official_intensification_pct'].values
                predicted_values = period_data['predicted_intensification'].values * 100
                
                # For growth period, use year-specific calibration factors
                if period_name == 'growth':
                    year_ratios = {}
                    for i, year in enumerate(period_data['Year'].values):
                        if predicted_values[i] > 0:
                            ratio = official_values[i] / predicted_values[i]
                            year_ratios[int(year)] = ratio
                    period_factors[period_name] = year_ratios
                    
                    avg_factor = np.mean(list(year_ratios.values()))
                    print(f"  {period_name} period: Using year-specific calibration factors, avg {avg_factor:.4f}")
                else:
                    ratios = [o/p if p > 0 else 1.0 for o, p in zip(official_values, predicted_values)]
                    period_factors[period_name] = np.mean(ratios)
                    print(f"  {period_name} period calibration factor: {period_factors[period_name]:.4f}")
            else:
                period_factors[period_name] = 1.0
                print(f"  {period_name} period has no data, using default factor 1.0")
        
        # Assign calibration factors to each year
        calibration_factors = {}
        
        # Before 2010: no calibration
        for year in range(2000, 2010):
            calibration_factors[year] = 1.0
        
        # 2010 and after: assign calibration factors based on periods
        for year in range(2010, 2022):
            assigned_period = None
            for period_name, (start_year, end_year) in periods.items():
                if start_year <= year <= end_year:
                    assigned_period = period_name
                    break
            
            if assigned_period:
                if assigned_period == 'growth' and isinstance(period_factors.get(assigned_period), dict):
                    year_factors = period_factors[assigned_period]
                    if year in year_factors:
                        calibration_factors[year] = year_factors[year]
                    else:
                        # Interpolation or extrapolation
                        available_years = sorted(list(year_factors.keys()))
                        if not available_years:
                            calibration_factors[year] = 1.0
                        elif year < min(available_years):
                            calibration_factors[year] = year_factors[min(available_years)]
                        elif year > max(available_years):
                            calibration_factors[year] = year_factors[max(available_years)]
                        else:
                            year_before = max([y for y in available_years if y < year])
                            year_after = min([y for y in available_years if y > year])
                            w_after = (year - year_before) / (year_after - year_before)
                            w_before = 1 - w_after
                            calibration_factors[year] = (
                                w_before * year_factors[year_before] + 
                                w_after * year_factors[year_after]
                            )
                elif assigned_period in period_factors:
                    calibration_factors[year] = period_factors[assigned_period]
                else:
                    calibration_factors[year] = 1.0
            else:
                calibration_factors[year] = 1.0
        
        return calibration_factors


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'random_state': 42,
        'response_file': r'E:\livestock carrying capacitiy\livestock_mapping\ruminant_farm\Y_livestock\牛饲养规模年鉴.xlsx',
        'predictor_file': r'E:\livestock carrying capacitiy\livestock_mapping\pig_farm\x_predict_output_province_total.xlsx',
        'output_dir': 'ruminant_farm_figures'
    }
    
    # Initialize model
    model = CattleIntensificationModel(config)
    
    # Load and prepare data
    print("Loading data...")
    data = model.load_and_prepare_data()
    
    # Train model
    print("Training model...")
    results = model.train()
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(data['X_test'])
    
    # Calibrate predictions using post-2010 data only
    print("Calibrating predictions...")
    calibrated = model.calibrate_predictions()
    
    # Generate visualizations
    print("Generating visualizations...")
    model.create_visualizations()
    
    print("Done!")