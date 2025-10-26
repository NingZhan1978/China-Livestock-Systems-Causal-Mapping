# -*- coding: utf-8 -*-
"""
DML Causal Analyzer 
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

from ..utils.e_value_utils import calculate_e_value

class CausalDMLAnalyzer:
    """
    Double Machine Learning based causal analyzer 
    
    """
    
    def __init__(self, data, config, feature_mapping=None, category_mapping=None):
        self.data = data.copy()
        self.config = config
        self.feature_mapping = feature_mapping or {}
        self.category_mapping = category_mapping or {}
        
        # Define variable categories based on system type
        if config['system'] == 'landless':
            self.env_vars = ['dem', 'tmpmean_year', 'presum_year', 'NDVI', 'PH', 'SOC', 'N', 'wind', 'snow_cover']
            self.socio_vars = ['population', 'GDP', 'NTL', 'travel_time', 'OSM', 'GSDL']
            self.poi_vars = ['company_type_sum', 'poi_count', 'total_capital']
        else:  # mixed_farming
            self.env_vars = ['dem', 'tmpmean_year', 'presum_year', 'NDVI', 'PH', 'SOC', 'N', 'wind', 'snow_cover']
            self.socio_vars = ['population', 'GDP', 'NTL', 'travel_time', 'OSM', 'GSDL']
            self.poi_vars = []  # Mixed-farming systems have no POI variables
        
        # Filter actually existing variables
        self.env_vars = [v for v in self.env_vars if v in data.columns]
        self.socio_vars = [v for v in self.socio_vars if v in data.columns]
        self.poi_vars = [v for v in self.poi_vars if v in data.columns]
        self.all_vars = self.env_vars + self.socio_vars + self.poi_vars
        
        print(f"Environmental variables: {len(self.env_vars)} - {self.env_vars}")
        print(f"Social-economic variables: {len(self.socio_vars)} - {self.socio_vars}")
        if self.poi_vars:
            print(f"POI variables: {len(self.poi_vars)} - {self.poi_vars}")
        
        # Store analysis results
        self.causal_results = {}
        self.treatment_effects = {}
        
    def prepare_data_for_causal_analysis(self):
        """Prepare data for causal analysis"""
        print("\n=== Data Preparation Stage ===")
        
        # Select analysis variables
        analysis_vars = self.all_vars.copy()
        
        # Add target variable
        target_var = 'log_livestock'
        if target_var in self.data.columns:
            analysis_vars.append(target_var)
        
        print(f"Total analysis variables: {len(analysis_vars)}")
        
        # Prepare clean data
        causal_data = self.data[analysis_vars].copy()
        
        # Handle missing values
        print(f"Original sample size: {len(causal_data)}")
        causal_data = causal_data.dropna()
        print(f"After removing missing values: {len(causal_data)}")
        
        # Remove extreme outliers
        for col in analysis_vars:
            Q1 = causal_data[col].quantile(0.05)
            Q3 = causal_data[col].quantile(0.95)
            causal_data = causal_data[(causal_data[col] >= Q1) & (causal_data[col] <= Q3)]
        
        print(f"After removing outliers: {len(causal_data)}")
        
        # Standardize data
        scaler = StandardScaler()
        causal_data_scaled = pd.DataFrame(
            scaler.fit_transform(causal_data),
            columns=causal_data.columns,
            index=causal_data.index
        )
        
        self.causal_data = causal_data_scaled
        self.analysis_vars = analysis_vars
        self.scaler = scaler
        
        return causal_data_scaled
    
    def double_machine_learning(self, treatment_var, outcome_var, control_vars, 
                               ml_model_y=None, ml_model_t=None, n_folds=5):
        """
        Double Machine Learning implementation 
        """
        
        if ml_model_y is None:
            ml_model_y = RandomForestRegressor(n_estimators=100, random_state=42)
        if ml_model_t is None:
            ml_model_t = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Prepare data
        Y = self.causal_data[outcome_var].values
        T = self.causal_data[treatment_var].values
        X = self.causal_data[control_vars].values
        
        n = len(Y)
        
        # Initialize residuals
        Y_residuals = np.zeros(n)
        T_residuals = np.zeros(n)
        
        # Cross-fitting
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for train_idx, test_idx in kf.split(X):
            # Training set
            X_train, Y_train, T_train = X[train_idx], Y[train_idx], T[train_idx]
            # Test set
            X_test, Y_test, T_test = X[test_idx], Y[test_idx], T[test_idx]
            
            # Step 1: Y on X regression
            model_y = clone(ml_model_y)
            model_y.fit(X_train, Y_train)
            Y_pred = model_y.predict(X_test)
            Y_residuals[test_idx] = Y_test - Y_pred
            
            # Step 2: T on X regression
            model_t = clone(ml_model_t)
            model_t.fit(X_train, T_train)
            T_pred = model_t.predict(X_test)
            T_residuals[test_idx] = T_test - T_pred
        
        # Step 3: Y residuals on T residuals regression to get causal effect
        try:
            # Simple linear regression
            theta = np.sum(Y_residuals * T_residuals) / np.sum(T_residuals ** 2)
            
            # Calculate standard error
            prediction_error = Y_residuals - theta * T_residuals
            variance = np.mean(prediction_error ** 2) / np.mean(T_residuals ** 2)
            std_error = np.sqrt(variance / n)
            
            # t-statistic and p-value
            t_stat = theta / std_error
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n-1))
            
            # Confidence interval
            t_critical = stats.t.ppf(0.975, df=n-1)
            ci_lower = theta - t_critical * std_error
            ci_upper = theta + t_critical * std_error
            
            # Calculate E-value
            e_value_point, e_value_ci = calculate_e_value(theta, std_error)
            
        except Exception as e:
            print(f"DML calculation failed: {e}")
            theta, std_error, t_stat, p_value = 0, np.inf, 0, 1
            ci_lower, ci_upper = 0, 0
            e_value_point, e_value_ci = 1, 1
        
        return {
            'treatment_effect': theta,
            'std_error': std_error,
            't_statistic': t_stat,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': p_value < 0.05,
            'Y_residuals': Y_residuals,
            'T_residuals': T_residuals,
            'e_value_point': e_value_point,
            'e_value_ci': e_value_ci
        }
    
    def run_dml_analysis(self):
        """
        Run DML analysis
        """
        print("\n=== DML Causal Analysis Stage ===")
        
        causal_data = self.prepare_data_for_causal_analysis()
        
        causal_discovery_results = {}
        
        # Prepare analysis
        outcome_var = 'log_livestock'
        
        print("Double Machine Learning analysis...")
        dml_results = []
        
        for treatment_var in self.all_vars:
            if treatment_var == outcome_var:
                continue
                
            # Control variables: all variables except current treatment variable
            control_vars = [v for v in self.all_vars if v != treatment_var]
            
            if len(control_vars) == 0:
                continue
            
            print(f"   Analyzing {treatment_var} -> {outcome_var}")
            
            try:
                dml_result = self.double_machine_learning(
                    treatment_var=treatment_var,
                    outcome_var=outcome_var,
                    control_vars=control_vars,
                    ml_model_y=RandomForestRegressor(n_estimators=50, random_state=42),
                    ml_model_t=RandomForestRegressor(n_estimators=50, random_state=42),
                    n_folds=3
                )
                
                dml_result['treatment_var'] = treatment_var
                dml_result['method'] = 'DML'
                dml_results.append(dml_result)
                
            except Exception as e:
                print(f"   DML failed for {treatment_var}: {e}")
                continue
        
        causal_discovery_results['DML_Results'] = pd.DataFrame(dml_results)
        
        self.causal_discovery_results = causal_discovery_results
        
        return causal_discovery_results
    
    def calculate_causal_importance(self):
        """
        Calculate causal importance based on DML results
        """
        print("\n=== Calculating Causal Importance ===")
        
        causal_importance_scores = {}
        
        # Initialize scores
        for var in self.analysis_vars:
            if var != 'log_livestock':
                causal_importance_scores[var] = {
                    'dml_score': 0,
                    'total_score': 0
                }
        
        # DML scores
        if 'DML_Results' in self.causal_discovery_results:
            dml_df = self.causal_discovery_results['DML_Results']
            
            if not dml_df.empty:
                # Normalize treatment effects
                max_abs_effect = dml_df['treatment_effect'].abs().max()
                if max_abs_effect > 0:
                    for _, row in dml_df.iterrows():
                        var = row['treatment_var']
                        if var in causal_importance_scores:
                            # Combine effect size and significance
                            effect_strength = abs(row['treatment_effect']) / max_abs_effect
                            significance_bonus = 0.5 if row['significant'] else 0
                            causal_importance_scores[var]['dml_score'] = effect_strength + significance_bonus
        
        # Calculate total scores
        for var in causal_importance_scores:
            scores = causal_importance_scores[var]
            scores['total_score'] = scores['dml_score']  # Only DML score
        
        # Convert to DataFrame
        importance_df = pd.DataFrame.from_dict(causal_importance_scores, orient='index')
        importance_df['Variable'] = importance_df.index
        importance_df = importance_df.reset_index(drop=True)
        
        # Normalize
        if importance_df['total_score'].max() > 0:
            importance_df['Causal_Importance_Norm'] = (
                importance_df['total_score'] / importance_df['total_score'].max()
            )
        else:
            importance_df['Causal_Importance_Norm'] = 0
        
        return importance_df.sort_values('total_score', ascending=False)
    
    def get_variable_category(self, var_name):
        """Get variable category"""
        if var_name in self.env_vars:
            return 'Environment'
        elif var_name in self.socio_vars:
            return 'Social-economy'
        elif var_name in self.poi_vars:
            return 'POI-Business'
        elif var_name == 'log_livestock':
            return 'Target'
        else:
            return 'Other'