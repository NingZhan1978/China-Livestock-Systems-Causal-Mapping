"""
Base class for livestock intensification models
Module 1: Population Segmentation Base Framework
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import shap
import os
import matplotlib.gridspec as gridspec
from abc import ABC, abstractmethod

class BaseLivestockModel(ABC):
    """Base class for all livestock intensification models"""
    
    def __init__(self, config=None):
        self.config = self._default_config()
        if config:
            self.config.update(config)
        self._setup_environment()
        self.official_data = self._load_official_data()
    
    def _default_config(self):
        """Default configuration"""
        return {
            'output_dir': 'results',
            'random_state': 42,
            'n_cv_folds': 5,
            'validation_years': 2
        }
    
    def _setup_environment(self):
        """Setup directories and plotting"""
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Set matplotlib parameters
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'legend.fontsize': 12,
            'figure.dpi': 300
        })
    
    def load_and_merge_data(self, response_path, predictor_path):
        """Load and merge response and predictor data"""
        # Load data
        response_df = pd.read_excel(response_path, usecols=['code', 'ratio_1', 'Year'])
        predictor_df = pd.read_excel(predictor_path)
        
        # Filter and merge
        province_data = response_df[response_df['code'] != 0].copy()
        merged_df = pd.merge(province_data, predictor_df, on=['code', 'Year'], how='inner')
        
        return merged_df
    
    def engineer_features(self, df):
        """Common feature engineering"""
        # Crop sum
        crop_cols = ['tiff_maize', 'tiff_maize_major', 'tiff_rice',
                    'tiff_rice_major', 'tiff_rice_second', 'tiff_soybean',
                    'tiff_wheat_spring', 'tiff_wheat_winter']
        df['crop_sum'] = df[crop_cols].sum(axis=1)
        
        # Fill missing crop data
        def fill_crop_data(group):
            if 'Year' in group.columns:
                year_col = 'Year'
            else:
                year_col = 'year'
            
            if 2016 in group[year_col].values:
                fill_value = group.loc[group[year_col] == 2016, 'crop_sum'].iloc[0]
                mask = (group[year_col] >= 2017) & (group[year_col] <= 2021)
                group.loc[mask, 'crop_sum'] = fill_value
            return group
        
        df = df.groupby('code', group_keys=False).apply(fill_crop_data)
        
        # Time features
        df['year_scaled'] = (df['Year'] - df['Year'].min()) / (df['Year'].max() - df['Year'].min())
        df['year_trend'] = df['Year'] - 2000
        
        # Handle missing values
        df['ratio_1'] = pd.to_numeric(df['ratio_1'], errors='coerce')
        df = df.fillna(df.mean())
        
        return df
    
    def create_sample_weights(self, years):
        """Create time-based sample weights emphasizing recent years"""
        normalized_years = (years - years.min()) / (years.max() - years.min())
        weights = 1 / (1 + np.exp(-6 * (normalized_years - 0.5)))
        weights = 0.3 + 0.7 * weights
        
        # Emphasize recent years
        weights[years >= 2010] *= 2
        weights[years >= 2015] *= 1.5
        
        return weights
    
    def optimize_models(self, X, y, years):
        """Optimize models with cross-validation"""
        weights = self.create_sample_weights(years)
        model_configs = self.get_model_configs()
        
        results = {}
        cv = KFold(n_splits=self.config['n_cv_folds'], shuffle=True,
                  random_state=self.config['random_state'])
        
        for name, config in model_configs.items():
            print(f"Optimizing {name}...")
            
            grid_search = GridSearchCV(
                config['model'], config['params'],
                cv=cv, scoring='neg_mean_squared_error', n_jobs=-1
            )
            
            grid_search.fit(X, y, regressor__sample_weight=weights)
            
            # Cross-validation predictions
            y_pred = np.zeros_like(y)
            for train_idx, test_idx in cv.split(X):
                grid_search.best_estimator_.fit(
                    X[train_idx], y[train_idx],
                    regressor__sample_weight=weights[train_idx]
                )
                y_pred[test_idx] = grid_search.best_estimator_.predict(X[test_idx])
            
            results[name] = {
                'model': grid_search.best_estimator_,
                'r2': r2_score(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'predictions': y_pred,
                'best_params': grid_search.best_params_
            }
            
            print(f"{name} - R²: {results[name]['r2']:.4f}")
        
        best_model_name = max(results, key=lambda k: results[k]['r2'])
        print(f"\nBest model: {best_model_name}")
        
        return results, best_model_name
    
    def plot_validation_results(self, model, X, y_true, y_pred, feature_names):
        """Create validation and feature importance plot"""
        fig = plt.figure(figsize=(16, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5], wspace=0.3)
        
        # Validation scatter plot
        ax1 = fig.add_subplot(gs[0])
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        ax1.scatter(y_true, y_pred, alpha=0.7, s=60)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', alpha=0.7)
        
        ax1.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}', 
                transform=ax1.transAxes, va='top', fontsize=12)
        ax1.set_xlabel(f'Observed {self.livestock_type.title()} Intensification Rate')
        ax1.set_ylabel(f'Predicted {self.livestock_type.title()} Intensification Rate')
        ax1.set_title('Model Validation')
        
        # Feature importance with SHAP
        ax2 = fig.add_subplot(gs[1])
        
        try:
            # Get SHAP values
            base_model = model.named_steps['regressor']
            X_transformed = model.named_steps['scaler'].transform(X)
            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer.shap_values(X_transformed)
            
            # Create SHAP summary plot
            shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, 
                             ax=ax2, show=False)
            ax2.set_title('Feature Importance (SHAP)')
        except Exception as e:
            print(f"Could not create SHAP plot: {e}")
            ax2.text(0.5, 0.5, 'SHAP analysis unavailable', 
                    transform=ax2.transAxes, ha='center', va='center')
        
        plt.tight_layout()
        save_path = os.path.join(self.config['output_dir'], 
                                f'{self.livestock_type}_validation_results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def conservative_calibration(self, predictions_df, official_df):
        """Apply conservative temporal calibration"""
        periods = self.get_calibration_periods()
        calibration_factors = {}
        
        # Calculate period-specific factors
        for period_name, (start_year, end_year) in periods.items():
            period_official = official_df[
                (official_df['Year'] >= start_year) & 
                (official_df['Year'] <= end_year)
            ]
            
            period_predicted = predictions_df[
                (predictions_df['Year'] >= start_year) & 
                (predictions_df['Year'] <= end_year)
            ]
            
            if len(period_official) > 0 and len(period_predicted) > 0:
                # Merge for comparison
                period_data = pd.merge(period_predicted, period_official, on='Year', how='inner')
                
                if len(period_data) > 0:
                    official_mean = period_data['official_intensification_pct'].mean()
                    predicted_mean = period_data['predicted_intensification'].mean() * 100
                    
                    if predicted_mean > 0:
                        factor = official_mean / predicted_mean
                    else:
                        factor = 1.0
                else:
                    factor = 1.0
            else:
                factor = 1.0
            
            # Conservative blending (70% calibration, 30% original)
            conservative_factor = 0.7 * factor + 0.3 * 1.0
            
            # Apply to all years in period
            for year in range(start_year, end_year + 1):
                calibration_factors[year] = conservative_factor
            
            print(f"{period_name} period: factor = {conservative_factor:.3f}")
        
        return calibration_factors
    
    def plot_calibration_comparison(self, comparison_df, calibration_factors):
        """Plot model predictions vs official data"""
        plt.figure(figsize=(12, 8))
        
        # Plot original predictions
        plt.plot(comparison_df['Year'], comparison_df['predicted_intensification'] * 100,
                'o-', label='Original Predictions', linewidth=2, markersize=6)
        
        # Plot calibrated predictions
        calibrated_preds = []
        for _, row in comparison_df.iterrows():
            factor = calibration_factors.get(int(row['Year']), 1.0)
            calibrated_preds.append(row['predicted_intensification'] * 100 * factor)
        
        plt.plot(comparison_df['Year'], calibrated_preds,
                'd--', label='Calibrated Predictions', linewidth=2, markersize=6)
        
        # Plot official data
        plt.plot(self.official_data['Year'], self.official_data['official_intensification_pct'],
                's-', label='Official Data', linewidth=2.5, markersize=8)
        
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Intensification Rate (%)', fontsize=14)
        plt.title(f'{self.livestock_type.title()} Farming Intensification: Model vs Official Data', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.config['output_dir'], 
                                f'{self.livestock_type}_calibration_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @abstractmethod
    def _load_official_data(self):
        """Load official data for each livestock type"""
        pass
    
    @abstractmethod
    def get_feature_columns(self):
        """Define feature columns for each livestock type"""
        pass
    
    @abstractmethod
    def prepare_target_variable(self, df):
        """Prepare target variable for each livestock type"""
        pass
    
    @abstractmethod
    def get_model_configs(self):
        """Model configurations for each livestock type"""
        pass
    
    @abstractmethod
    def get_calibration_periods(self):
        """Calibration periods for each livestock type"""
        pass
    
    def run_analysis(self, response_path, predictor_path, county_path):
        """Execute complete analysis pipeline"""
        print(f"Running {self.livestock_type} analysis...")
        
        # Load and prepare data
        df = self.load_and_merge_data(response_path, predictor_path)
        df = self.engineer_features(df)
        
        feature_cols = self.get_feature_columns()
        X = df[feature_cols].fillna(df[feature_cols].mean()).values
        y = self.prepare_target_variable(df)
        years = df['Year'].values
        
        # Model optimization
        model_results, best_model_name = self.optimize_models(X, y, years)
        best_model = model_results[best_model_name]['model']
        y_pred = model_results[best_model_name]['predictions']
        
        # Create validation plot
        self.plot_validation_results(best_model, X, y, y_pred, feature_cols)
        
        # Generate county predictions
        print("Generating county-level predictions...")
        county_df = pd.read_excel(county_path)
        county_df = self.engineer_features(county_df)
        
        # Handle year column naming
        year_col = 'Year' if 'Year' in county_df.columns else 'year'
        county_df = county_df[county_df[year_col].between(2000, 2021)]
        
        county_features = county_df[feature_cols].fillna(county_df[feature_cols].mean())
        county_predictions = best_model.predict(county_features)
        
        county_df['predicted_intensification'] = county_predictions
        
        # National aggregation for calibration
        national_pred = county_df.groupby(year_col)['predicted_intensification'].mean().reset_index()
        national_pred.rename(columns={year_col: 'Year'}, inplace=True)
        
        # Conservative calibration
        print("Applying conservative temporal calibration...")
        calibration_factors = self.conservative_calibration(national_pred, self.official_data)
        
        # Apply calibration
        county_df['calibration_factor'] = county_df[year_col].map(calibration_factors)
        county_df['calibrated_intensification'] = (
            county_df['predicted_intensification'] * county_df['calibration_factor']
        ).clip(0, 1)
        
        # Create comparison plot
        comparison_df = pd.merge(national_pred, self.official_data, on='Year', how='outer')
        self.plot_calibration_comparison(comparison_df, calibration_factors)
        
        # Save results
        output_path = os.path.join(self.config['output_dir'], 
                                  f'{self.livestock_type}_county_predictions.csv')
        county_df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
        
        return best_model, county_df, calibration_factors