# -*- coding: utf-8 -*-
"""
Main Livestock Analysis Module
Unified analysis pipeline for all species and production systems
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.base import clone
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from linearmodels.panel import PanelOLS

from ..config.livestock_config import get_config
from ..config.variable_mappings import VARIABLE_MAPPINGS, get_system_categories, get_variable_category
from ..core.data_processor import DataProcessor
from ..core.enhanced_stacking import EnhancedStacking
from ..core.dml_analyzer import CausalDMLAnalyzer
from ..visualization.kde_plots import plot_kde_performance
from ..visualization.shap_plots import generate_complete_shap_analysis
from ..visualization.dml_plots import plot_dml_causal_vs_predictive_analysis
from ..visualization.spatial_plots import create_spatial_visualizations
from ..utils.statistical_utils import calculate_robustness_score

class LivestockAnalyzer:
    """
    Main livestock analysis class supporting all species and production systems
    """
    
    def __init__(self, config_key):
        """
        Initialize analyzer with configuration
        
        Parameters:
        config_key (str): Configuration key (e.g., 'pig_landless', 'sheep_mixed')
        """
        self.config_key = config_key
        self.config = get_config(config_key)
        self.data_processor = DataProcessor(self.config)
        
        # Analysis results storage
        self.processed_data = None
        self.causal_results = None
        self.stacking_model = None
        self.spatial_results = None
        
        print(f"Initialized analyzer for: {self.config['title']}")
        print(f"System: {self.config['system']}, Species: {self.config['species']}")
        
    def stage_1_causal_inference_analysis(self):
        """
        Stage 1: Complete causal inference analysis
        Including OLS, Fixed Effects, Instrumental Variables
        """
        print("="*60)
        print("Stage 1: Causal Inference Analysis (OLS, FE, IV)")
        print("="*60)
        
        # Process data
        self.processed_data = self.data_processor.process_full_pipeline()
        causal_data = self.processed_data['causal_data']
        available_features = self.processed_data['available_features']
        
        # Define variables for different models
        dependent_var = 'log_livestock'
        core_vars = self.processed_data['available_original_features']
        interaction_vars = self.processed_data['interaction_terms']
        
        # OLS uses all variables
        all_vars_ols = core_vars + interaction_vars
        
        # FE model uses time-varying variables (exclude time-invariant variables)
        # These variables are usually time-invariant or barely change at county level: travel_time, dem, PH, SOC, N
        time_invariant_vars = ['travel_time', 'dem', 'PH', 'SOC', 'N']
        time_varying_vars = [var for var in core_vars if var not in time_invariant_vars]
        all_vars_fe = time_varying_vars + interaction_vars
        
        print(f"OLS variable count: {len(all_vars_ols)}")
        print(f"FE variable count: {len(all_vars_fe)} (excluded time-invariant variables)")

        # OLS regression
        print("\n4. Executing OLS regression...")
        X_ols = causal_data[all_vars_ols]
        X_ols = sm.add_constant(X_ols)
        y_ols = causal_data[dependent_var]
        
        ols_model = OLS(y_ols, X_ols).fit(cov_type='HC1')
        print("OLS regression completed")
        print(f"OLS R²: {ols_model.rsquared:.4f}")

        # Fixed Effects regression
        print("\n5. Executing Fixed Effects regression...")
        causal_data_panel = causal_data.set_index(['code', 'year'])
        
        try:
            fe_model = PanelOLS(causal_data_panel[dependent_var], 
                               causal_data_panel[all_vars_fe], 
                               entity_effects=True, 
                               time_effects=True,
                               drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)
            print("Fixed Effects regression completed")
            print(f"FE R²: {fe_model.rsquared:.4f}")
            fe_success = True
        except Exception as e:
            print(f"Fixed Effects regression failed: {e}")
            print("Using simplified FE model...")
            try:
                # Further simplify, use only clearly time-varying variables
                simple_fe_vars = ['population', 'GDP', 'snow_cover', 'tmpmean_year', 'presum_year', 
                                 'NDVI', 'OSM', 'NTL', 'GSDL', 'wind']
                
                # Add POI variables to time-varying variables if available
                if self.config['system'] == 'landless':
                    poi_vars = ['company_type_sum', 'poi_count', 'total_capital']
                    for var in poi_vars:
                        if var in causal_data.columns:
                            simple_fe_vars.append(var)
                
                simple_fe_vars = [var for var in simple_fe_vars if var in causal_data.columns]
                
                fe_model = PanelOLS(causal_data_panel[dependent_var], 
                                   causal_data_panel[simple_fe_vars], 
                                   entity_effects=True, 
                                   drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)
                print("Simplified FE regression completed")
                print(f"Simplified FE R²: {fe_model.rsquared:.4f}")
                all_vars_fe = simple_fe_vars  # Update FE variable list
                fe_success = True
            except Exception as e2:
                print(f"Simplified FE regression also failed: {e2}")
                fe_model = ols_model  # Use OLS as fallback
                fe_success = False

        # Since lag variables are removed, use FE results for IV regression
        print("\n6. IV regression using FE results (no lag variables)...")
        iv_model = fe_model
        iv_success = fe_success

        # Organize results
        print("\n7. Organizing causal inference results...")
        causal_results = []
        
        for var in all_vars_ols:  # Use OLS complete variable list as baseline
            result = {'Variable': var}
            
            # OLS results
            if var in ols_model.params.index:
                result['OLS_Coef'] = ols_model.params[var]
                result['OLS_PValue'] = ols_model.pvalues[var]
                result['OLS_Significant'] = result['OLS_PValue'] < 0.05
            else:
                result['OLS_Coef'] = 0
                result['OLS_PValue'] = 1
                result['OLS_Significant'] = False
            
            # FE results
            if fe_success and var in fe_model.params.index:
                result['FE_Coef'] = fe_model.params[var]
                result['FE_PValue'] = fe_model.pvalues[var]
                result['FE_Significant'] = result['FE_PValue'] < 0.05
            else:
                # For time-invariant variables, FE cannot estimate, mark as N/A
                if var in time_invariant_vars:
                    result['FE_Coef'] = np.nan
                    result['FE_PValue'] = np.nan
                    result['FE_Significant'] = False
                else:
                    result['FE_Coef'] = 0
                    result['FE_PValue'] = 1
                    result['FE_Significant'] = False
            
            # IV results
            if iv_success and var in iv_model.params.index:
                result['IV_Coef'] = iv_model.params[var]
                result['IV_PValue'] = iv_model.pvalues[var]
                result['IV_Significant'] = result['IV_PValue'] < 0.05
            else:
                # If IV doesn't have this variable, use FE results
                result['IV_Coef'] = result['FE_Coef'] if not np.isnan(result['FE_Coef']) else result['OLS_Coef']
                result['IV_PValue'] = result['FE_PValue'] if not np.isnan(result['FE_PValue']) else result['OLS_PValue']
                result['IV_Significant'] = result['FE_Significant']
            
            # Calculate robustness score
            result['Robustness_Score'] = calculate_robustness_score(result)
            
            causal_results.append(result)
        
        causal_results_df = pd.DataFrame(causal_results)
        
        # Save causal inference results
        causal_csv_path = os.path.join(
            self.config['output_paths']['visualizations'], 
            f'causal_inference_results_{datetime.now().strftime("%Y%m%d")}.csv'
        )
        causal_results_df.to_csv(causal_csv_path, index=False)
        
        print(f"\nCausal inference results saved to: {causal_csv_path}")
        
        self.causal_results = {
            'results_df': causal_results_df,
            'causal_data': causal_data,
            'scaler': self.processed_data['scaler']
        }
        
        return self.causal_results
    
    def stage_2_prediction_analysis(self):
        """
        Stage 2: Prediction model analysis - Integrating SHAP and DML causal analysis
        """
        print("\n" + "="*60)
        print("Stage 2: Prediction Model Analysis (Stacking + Complete SHAP + DML Causal Analysis)")
        print("="*60)
        
        start_time = time.time()
        
        # Prepare prediction analysis data
        print("1. Preparing prediction analysis data...")
        
        causal_data = self.causal_results['causal_data']
        available_features = self.processed_data['available_features']
        
        # Select available features
        features = causal_data[available_features]
        target = causal_data['log_livestock']
        
        print(f"Prediction analysis feature count: {len(available_features)}")
        print(f"Prediction analysis sample count: {len(causal_data)}")
        
        # Get system-specific categories and colors
        category_colors = get_system_categories(self.config['system'])
        
        # Split training and test sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Define models
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                max_features='log2', random_state=42
            ),
            'ExtraTrees': ExtraTreesRegressor(
                n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                max_features=None, random_state=42
            ),
            'XGBoost': XGBRegressor(
                n_estimators=500, max_depth=9, learning_rate=0.1, subsample=0.9,
                colsample_bytree=0.7, min_child_weight=3, reg_alpha=0, reg_lambda=0.1, random_state=42
            ),
            'LightGBM': LGBMRegressor(
                n_estimators=400, max_depth=15, learning_rate=0.2, num_leaves=20,
                subsample=0.7, colsample_bytree=0.9, min_child_samples=15, reg_alpha=1,
                reg_lambda=1.5, random_state=42, verbosity=-1, verbose=-1
            ),
            'CatBoost': CatBoostRegressor(
                iterations=500, depth=10, learning_rate=0.05, l2_leaf_reg=1,
                subsample=0.7, random_strength=1, border_count=32, random_state=42,
                bagging_temperature=10, verbose=0
            )
        }

        # Evaluate model performance
        print("2. Evaluating model performance...")
        outer_kf = KFold(n_splits=5, shuffle=True, random_state=42)
        model_performances = {name: {'r2': [], 'mse': []} for name in models.keys()}

        for fold, (train_idx, test_idx) in enumerate(outer_kf.split(X_train)):
            print(f"\nProcessing outer cross-validation fold {fold+1}/5")

            if hasattr(X_train, 'iloc'):
                X_train_fold, X_test_fold = X_train.iloc[train_idx], X_train.iloc[test_idx]
                y_train_fold, y_test_fold = y_train.iloc[train_idx], y_train.iloc[test_idx]
            else:
                X_train_fold, X_test_fold = X_train[train_idx], X_train[test_idx]
                y_train_fold, y_test_fold = y_train[train_idx], y_train[test_idx]

            inner_kf = KFold(n_splits=3, shuffle=True, random_state=fold)

            for name, model in models.items():
                print(f"  Evaluating model: {name}")
                
                cv_results = cross_validate(
                    clone(model), 
                    X_train_fold, 
                    y_train_fold, 
                    cv=inner_kf,
                    scoring=['r2', 'neg_mean_squared_error'],
                    return_train_score=False,
                    n_jobs=-1
                )
                
                trained_model = clone(model)
                
                if name == 'LightGBM':
                    trained_model.set_params(verbosity=-1, verbose=-1)
                elif name == 'CatBoost':
                    trained_model.set_params(verbose=0)
                    
                trained_model.fit(X_train_fold, y_train_fold)
                
                y_pred = trained_model.predict(X_test_fold)
                r2 = r2_score(y_test_fold, y_pred)
                mse = mean_squared_error(y_test_fold, y_pred)
                
                model_performances[name]['r2'].append(r2)
                model_performances[name]['mse'].append(mse)
                
                print(f"    Fold {fold+1}: R²={r2:.4f}, MSE={mse:.4f}")

        # Find best model
        model_avg_performance = {}
        for name, scores in model_performances.items():
            avg_r2 = np.mean(scores['r2'])
            std_r2 = np.std(scores['r2'])
            avg_mse = np.mean(scores['mse'])
            std_mse = np.std(scores['mse'])

            model_avg_performance[name] = {
                'mean_r2': avg_r2,
                'std_r2': std_r2,
                'mean_mse': avg_mse,
                'std_mse': std_mse,
                'mean_rmse': np.sqrt(avg_mse)
            }

            print(f"\n{name} - Average R²: {avg_r2:.4f} (±{std_r2:.4f}), MSE: {avg_mse:.4f} (±{std_mse:.4f})")

        best_model_name = max(model_avg_performance.items(), key=lambda x: x[1]['mean_r2'])[0]
        print(f"\nBased on nested cross-validation, best model is: {best_model_name}")

        # Build Stacking model
        print("\n3. Building Stacking model...")
        base_models = []
        for name, model in models.items():
            model_clone = clone(model)
            
            if name == 'LightGBM':
                model_clone.set_params(verbosity=-1, verbose=-1)
            elif name == 'CatBoost':
                model_clone.set_params(verbose=0)
            
            base_models.append(model_clone)

        meta_model = clone(models[best_model_name])
        if best_model_name == 'LightGBM':
            meta_model.set_params(verbosity=-1, verbose=-1)
        elif best_model_name == 'CatBoost':
            meta_model.set_params(verbose=0)

        stacking_model = EnhancedStacking(base_models, meta_model, n_folds=5)
        stacking_model.fit(X_train, y_train)

        # Save model
        model_path = os.path.join(self.config['output_paths']['models'], self.config['model_name'])
        joblib.dump(stacking_model, model_path)
        print(f"Stacking model saved to {model_path}")

        # Evaluate final model
        print("\n4. Evaluating model performance on independent test set...")
        final_results = {}
        for name, model in models.items():
            trained_model = clone(model)
            
            if name == 'LightGBM':
                trained_model.set_params(verbosity=-1, verbose=-1)
            elif name == 'CatBoost':
                trained_model.set_params(verbose=0)
                
            trained_model.fit(X_train, y_train)

            y_pred = trained_model.predict(X_test)
            test_r2 = r2_score(y_test, y_pred)
            test_mse = mean_squared_error(y_test, y_pred)
            test_rmse = np.sqrt(test_mse)

            final_results[name] = {
                'model': trained_model,
                'test_r2': test_r2,
                'test_rmse': test_rmse
            }

            print(f"{name} - Test R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}")

        # Evaluate Stacking model
        y_pred_stacking = stacking_model.predict(X_test)
        stacking_r2 = r2_score(y_test, y_pred_stacking)
        stacking_rmse = np.sqrt(mean_squared_error(y_test, y_pred_stacking))

        final_results['Stacking'] = {
            'model': stacking_model,
            'test_r2': stacking_r2,
            'test_rmse': stacking_rmse
        }

        print(f"Stacking - Test R²: {stacking_r2:.4f}, RMSE: {stacking_rmse:.4f}")

        # KDE performance plot
        print("\n5. Generating KDE performance comparison plot...")
        plot_kde_performance(
            y_test, y_pred_stacking, stacking_r2, stacking_rmse,
            self.config['title'], self.config['output_paths']['visualizations']
        )

        # Complete SHAP analysis
        print("\n6. Generating complete SHAP analysis...")
        shap_importance_df = generate_complete_shap_analysis(
            stacking_model, X_train, X_test, y_test, 
            available_features, VARIABLE_MAPPINGS, 
            category_colors, best_model_name,
            self.config['title'], self.config['output_paths']['visualizations'],
            self.config['system']
        )

        # DML causal vs predictive importance comparison analysis
        print("\n7. DML causal vs predictive importance comparison analysis...")
        try:
            # Prepare data
            analysis_data = X_train.copy()
            analysis_data['log_livestock'] = y_train.values
            
            # Create DML causal analyzer
            analyzer = CausalDMLAnalyzer(
                data=analysis_data,
                config=self.config,
                feature_mapping=VARIABLE_MAPPINGS,
                category_mapping=category_colors
            )
            
            # Causal structure discovery
            causal_results = analyzer.run_dml_analysis()
            
            # Causal vs predictive importance comparison
            comparison_results = plot_dml_causal_vs_predictive_analysis(
                analyzer, stacking_model, X_train, self.config
            )
            
            print(f"DML causal analysis completed")
            
        except Exception as e:
            print(f"DML causal analysis failed: {e}")
            comparison_results = None

        end_time = time.time()
        print(f"\nStage 2 completed, time taken: {(end_time - start_time)/60:.2f} minutes")
        
        self.stacking_model = stacking_model
        
        return {
            'final_results': final_results,
            'shap_importance_df': shap_importance_df,
            'stacking_model': stacking_model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'available_features': available_features,
            'comparison_results': comparison_results
        }
    
    def stage_3_spatial_prediction(self):
        """
        Stage 3: Spatial prediction analysis
        """
        print("\n" + "="*60)
        print("Stage 3: Spatial Prediction Analysis (DML-Enhanced)")
        print("="*60)
        
        start_time = time.time()
        
        # Spatial prediction configuration
        spatial_config = {
            'variables': self.config['variables'],
            'years': range(2000, 2022),
            'base_path': self.config['data_paths']['spatial_base_path'],
            'output_dir': self.config['output_paths']['spatial_predictions'],
            'reference_file': self.config['data_paths']['reference_file'],
            'chunk_size': 1000000
        }
        
        print("1. Starting spatial prediction...")
        print(f"   Using variables: {len(spatial_config['variables'])}")
        print(f"   Prediction years: {min(spatial_config['years'])}-{max(spatial_config['years'])}")
        
        try:
            # Execute real spatial prediction
            spatial_results = self._perform_spatial_prediction(spatial_config)
            
            # Create visualizations
            if spatial_results and spatial_results['status'] == 'completed':
                print("2. Creating spatial prediction visualizations...")
                create_spatial_visualizations(spatial_results, spatial_config, self.config['title'])
            
        except Exception as e:
            print(f"Spatial prediction failed: {e}")
            spatial_results = {'status': 'failed', 'error': str(e)}
        
        end_time = time.time()
        print(f"\nStage 3 completed, time taken: {(end_time - start_time)/60:.2f} minutes")
        
        self.spatial_results = spatial_results
        
        return spatial_results
    
    def _perform_spatial_prediction(self, config):
        """
        Execute real spatial prediction - handles lag variables and interaction terms
        """
        print("=== Starting Real Spatial Prediction ===")
        
        variables = config['variables']
        years = config['years']
        base_path = config['base_path']
        output_dir = config['output_dir']
        reference_file = config['reference_file']
        CHUNK_SIZE = config['chunk_size']
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Check reference file exists
        if not os.path.exists(reference_file):
            raise FileNotFoundError(f"Reference file does not exist: {reference_file}")
        
        # Read reference file information
        import rasterio
        with rasterio.open(reference_file) as src:
            consistent_size = src.shape
            profile = src.profile
            profile.update(dtype=rasterio.float32, count=1)
        
        print(f"Raster size: {consistent_size}")
        print(f"Processing years: {min(years)}-{max(years)}")
        
        successful_years = []
        failed_years = []
        
        # Get model feature list used during training
        if hasattr(self.stacking_model, 'feature_names'):
            model_features = self.stacking_model.feature_names
        else:
            # If no feature names, get from first base model
            try:
                sample_data = pd.DataFrame(np.random.random((1, len(variables))), columns=variables)
                # Add causal interaction terms
                for term in self.config['interaction_terms']:
                    if term == 'GDP_NDVI' and 'GDP' in sample_data.columns and 'NDVI' in sample_data.columns:
                        sample_data['GDP_NDVI'] = sample_data['GDP'] * sample_data['NDVI']
                    elif term == 'presum_population_interaction' and 'presum_year' in sample_data.columns and 'population' in sample_data.columns:
                        sample_data['presum_population_interaction'] = sample_data['presum_year'] * sample_data['population']
                    elif term == 'poi_GDP_interaction' and 'poi_count' in sample_data.columns and 'GDP' in sample_data.columns:
                        sample_data['poi_GDP_interaction'] = sample_data['poi_count'] * sample_data['GDP']
                
                # Try prediction to get feature requirements
                _ = self.stacking_model.predict(sample_data)
                model_features = sample_data.columns.tolist()
                print(f"Inferred model feature list: {len(model_features)} features")
            except Exception as e:
                print(f"Cannot determine model feature requirements: {e}")
                return {'status': 'failed', 'error': 'Cannot determine model features'}
        
        print(f"Model required features: {model_features}")
        
        for year in years:
            print(f"\nProcessing year {year}...")
            
            try:
                # 1. Load original variable data
                feature_data = []
                loaded_vars = []
                missing_vars = []
                
                for var in variables:
                    var_path = os.path.join(base_path, var)
                    
                    if not os.path.exists(var_path):
                        print(f"Warning: Variable directory does not exist: {var}")
                        missing_vars.append(var)
                        continue
                    
                    # Find files matching the year
                    matching_files = [f for f in os.listdir(var_path) if str(year) in f and f.endswith('.tif')]
                    
                    if not matching_files:
                        print(f"Warning: No {year} data file found for variable {var}")
                        missing_vars.append(var)
                        continue
                    
                    # Use first matching file
                    filename = matching_files[0]
                    filepath = os.path.join(var_path, filename)
                    
                    print(f"  Loading: {var} -> {filename}")
                    
                    with rasterio.open(filepath) as src:
                        if src.shape != consistent_size:
                            print(f"Warning: Size mismatch: {var} {year}, expected {consistent_size}, actual {src.shape}")
                            missing_vars.append(var)
                            continue
                        
                        data = src.read(1)
                        data = np.nan_to_num(data, nan=-9999)
                        feature_data.append(data.ravel())
                        loaded_vars.append(var)
                
                # Check if key variables exist
                critical_vars = ['GDP', 'NDVI', 'presum_year', 'population', 'dem']
                missing_critical = [var for var in critical_vars if var not in loaded_vars]
                
                if missing_critical:
                    print(f"Missing critical variables: {missing_critical}")
                    if len(missing_critical) > 2:  # If too many critical variables are missing, skip
                        print(f"Year {year} missing too many critical variables, skipping")
                        failed_years.append(year)
                        continue
                
                if len(feature_data) < len(variables) * 0.7:  # Need at least 70% of variables
                    print(f"Warning: {year} data incomplete, skipping")
                    failed_years.append(year)
                    continue
                
                # 2. Prepare feature array
                features_array = np.column_stack(feature_data)
                print(f"  Feature array shape: {features_array.shape}")
                
                # 3. Initialize prediction results
                predictions = np.zeros(features_array.shape[0], dtype=np.float32)
                
                # 4. Chunk prediction
                print("  Starting chunk prediction...")
                total_chunks = (features_array.shape[0] + CHUNK_SIZE - 1) // CHUNK_SIZE
                
                for i in range(0, features_array.shape[0], CHUNK_SIZE):
                    chunk_num = i // CHUNK_SIZE + 1
                    if chunk_num % 10 == 1 or chunk_num % 10 == 0:
                        print(f"  Processing chunk {chunk_num} / {total_chunks}")
                    
                    chunk = features_array[i:i+CHUNK_SIZE]
                    
                    # Identify valid pixels
                    valid_mask = ~np.any(chunk == -9999, axis=1)
                    valid_chunk = chunk[valid_mask]
                    
                    if valid_chunk.shape[0] > 0:
                        # Create DataFrame
                        valid_chunk_df = pd.DataFrame(valid_chunk, columns=loaded_vars)
                        
                        # Add causal interaction terms (only add those used during model training)
                        try:
                            for term in self.config['interaction_terms']:
                                if term == 'GDP_NDVI' and term in model_features and 'GDP' in loaded_vars and 'NDVI' in loaded_vars:
                                    valid_chunk_df['GDP_NDVI'] = valid_chunk_df['GDP'] * valid_chunk_df['NDVI']
                                elif term == 'presum_population_interaction' and term in model_features and 'presum_year' in loaded_vars and 'population' in loaded_vars:
                                    valid_chunk_df['presum_population_interaction'] = valid_chunk_df['presum_year'] * valid_chunk_df['population']
                                elif term == 'poi_GDP_interaction' and term in model_features and 'poi_count' in loaded_vars and 'GDP' in loaded_vars:
                                    valid_chunk_df['poi_GDP_interaction'] = valid_chunk_df['poi_count'] * valid_chunk_df['GDP']
                            
                        except Exception as e:
                            print(f"    Interaction term generation failed: {e}")
                        
                        # Ensure all model required features are included
                        missing_features = [f for f in model_features if f not in valid_chunk_df.columns]
                        if missing_features:
                            print(f"    Missing features: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
                            # Set missing features to 0 or mean
                            for feat in missing_features:
                                valid_chunk_df[feat] = 0
                        
                        # Arrange by model feature order
                        try:
                            valid_chunk_df = valid_chunk_df[model_features]
                        except KeyError as e:
                            print(f"    Feature alignment failed: {e}")
                            continue
                        
                        # Prediction
                        try:
                            preds = self.stacking_model.predict(valid_chunk_df)
                            
                            # Convert back to original scale (if log transformation was used)
                            original_preds = np.expm1(preds)
                            
                            # Ensure prediction values are positive
                            original_preds = np.clip(original_preds, 0, None)
                            
                            predictions[i:i+CHUNK_SIZE][valid_mask] = original_preds
                            
                        except Exception as e:
                            print(f"    Prediction failed: {e}")
                            continue
                    
                    # Set invalid pixels to NaN
                    predictions[i:i+CHUNK_SIZE][~valid_mask] = np.nan
                
                # 5. Save results
                pred_grid = predictions.reshape(consistent_size)
                
                # Calculate statistics
                valid_predictions = pred_grid[~np.isnan(pred_grid)]
                stats_info = {
                    'year': year,
                    'total_pixels': pred_grid.size,
                    'valid_pixels': len(valid_predictions),
                    'mean_prediction': np.mean(valid_predictions) if len(valid_predictions) > 0 else 0,
                    'max_prediction': np.max(valid_predictions) if len(valid_predictions) > 0 else 0,
                    'min_prediction': np.min(valid_predictions) if len(valid_predictions) > 0 else 0,
                    'coverage_rate': len(valid_predictions) / pred_grid.size * 100
                }
                
                print(f"  Prediction statistics: Valid pixels {stats_info['valid_pixels']:,}/{stats_info['total_pixels']:,} ({stats_info['coverage_rate']:.1f}%)")
                print(f"  Prediction range: {stats_info['min_prediction']:.2f} - {stats_info['max_prediction']:.2f}")
                
                # Save raster file
                species_name = self.config['species'].title()
                system_name = self.config['system'].replace('_', '_').title()
                output_filename = f'stacking_pred_{system_name}_{species_name}_farm_{year}.tif'
                output_path = os.path.join(output_dir, output_filename)
                
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(pred_grid.astype(rasterio.float32), 1)
                
                print(f"Year {year} prediction results saved to: {output_path}")
                successful_years.append(year)
                
            except Exception as e:
                print(f"Year {year} processing failed: {e}")
                failed_years.append(year)
                continue
        
        # Prediction completion summary
        print(f"\n=== Spatial Prediction Completed ===")
        print(f"Successfully predicted years: {len(successful_years)}")
        if successful_years:
            print(f"Year range: {min(successful_years)}-{max(successful_years)}")
        if failed_years:
            print(f"Failed years: {len(failed_years)}: {failed_years}")
        
        return {
            'status': 'completed' if successful_years else 'failed',
            'successful_years': successful_years,
            'failed_years': failed_years,
            'output_directory': output_dir,
            'total_files': len(successful_years),
            'prediction_method': 'DML-Enhanced Stacking',
            'variables_used': variables,
            'model_features': model_features if 'model_features' in locals() else []
        }
    
    def run_full_analysis(self):
        """
        Run complete three-stage analysis
        """
        print("="*80)
        print(f"Complete Analysis: {self.config['title']} (DML-Enhanced)")
        print("Including: Causal Inference + DML + Prediction Models + Spatial Analysis")
        print("="*80)
        
        total_start_time = time.time()
        
        try:
            # Stage 1: Causal inference analysis
            print("\nStarting Stage 1: Causal Inference Analysis...")
            self.stage_1_causal_inference_analysis()
            print("Stage 1 Completed")
            
            # Stage 2: Prediction model analysis (including DML)
            print("\nStarting Stage 2: Prediction Model & DML Analysis...")
            self.stage_2_prediction_analysis()
            print("Stage 2 Completed")
            
            # Stage 3: Spatial prediction analysis
            print("\nStarting Stage 3: Spatial Prediction Analysis...")
            self.stage_3_spatial_prediction()
            print("Stage 3 Completed")
            
        except Exception as e:
            print(f"Error occurred during analysis: {e}")
            import traceback
            traceback.print_exc()
        
        total_end_time = time.time()
        total_duration = (total_end_time - total_start_time) / 60
        
        print("\n" + "="*80)
        print("Complete analysis pipeline finished!")
        print(f"Total time: {total_duration:.2f} minutes")
        print("="*80)
        
        # Generate final summary report
        self._generate_final_summary_report(total_duration)
        
        return {
            'causal_results': self.causal_results,
            'stacking_model': self.stacking_model,
            'spatial_results': self.spatial_results,
            'total_duration': total_duration
        }
    
    def _generate_final_summary_report(self, total_duration):
        """
        Generate final summary report
        """
        summary_content = f"""
{self.config['title'].upper()} CAUSAL INFERENCE & PREDICTION ANALYSIS
FINAL SUMMARY REPORT
{'='*60}

Analysis Completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Total Duration: {total_duration:.2f} minutes
Configuration: {self.config_key}

ANALYSIS COMPONENTS COMPLETED:
Stage 1: Traditional Causal Inference (OLS, FE, IV)
Stage 2: Advanced Machine Learning & DML Analysis
   - Stacking Ensemble Models
   - Comprehensive SHAP Analysis  
   - Double Machine Learning (DML)
   - E-value Sensitivity Analysis
Stage 3: Spatial Prediction Analysis

SYSTEM CONFIGURATION:
Production System: {self.config['system'].replace('_', ' ').title()}
Livestock Species: {self.config['species'].title()}
Variable Count: {len(self.config['variables'])}
POI Variables: {'Included' if self.config['system'] == 'landless' else 'Not Included'}

KEY METHODOLOGICAL FEATURES:
Advanced Causal Inference:
   - Double Machine Learning for unbiased estimates
   - E-value analysis for sensitivity assessment
   - Multiple method validation

Enhanced Machine Learning:
   - Configuration-driven variable selection
   - System-specific interaction terms
   - One Earth journal visualization standards

Comprehensive Validation:
   - Cross-method consistency checks
   - Robustness analysis with multiple metrics
   - Sensitivity analysis for unmeasured confounding

OUTPUT DIRECTORIES:
- Visualizations: {self.config['output_paths']['visualizations']}
- Models: {self.config['output_paths']['models']}
- Spatial Predictions: {self.config['output_paths']['spatial_predictions']}

MODULAR ARCHITECTURE BENEFITS:
- Configuration-driven design eliminates code duplication
- Core algorithms maintain academic integrity
- Easy extension to new species and systems
- Unified analysis pipeline across all configurations

For detailed results, please refer to individual analysis reports
and generated visualizations.
"""
        
        summary_path = os.path.join(
            self.config['output_paths']['visualizations'], 
            'FINAL_ANALYSIS_SUMMARY.txt'
        )
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        print(f"📋 Final summary report saved to: {summary_path}")