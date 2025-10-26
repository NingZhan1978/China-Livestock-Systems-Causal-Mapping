# -*- coding: utf-8 -*-
"""
Enhanced Stacking Model Implementation

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin, clone

class EnhancedStacking(BaseEstimator, RegressorMixin):
    """
    Improved Stacking model implementation ensuring no data leakage
    during model selection and meta-feature generation
    """
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.base_models_ = []
        self.feature_names = None

    def fit(self, X, y):
        """
        Train stacking model using cross-validation to ensure no data leakage
        """
        self.feature_names = X.columns if hasattr(X, 'columns') else None
        
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        self.base_models_ = [[] for _ in range(len(self.base_models))]
        
        for i, model in enumerate(self.base_models):
            for train_idx, val_idx in kf.split(X):
                if hasattr(X, 'iloc'):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train = y.iloc[train_idx]
                else:
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train = y[train_idx]
                
                fold_model = clone(model)
                fold_model.fit(X_train, y_train)
                
                self.base_models_[i].append(fold_model)
                
                meta_features[val_idx, i] = fold_model.predict(X_val)
        
        self.meta_model.fit(meta_features, y)
        
        for i, model in enumerate(self.base_models):
            final_model = clone(model)
            final_model.fit(X, y)
            self.base_models_[i].append(final_model)
            
        return self
        
    def predict(self, X):
        """
        Generate predictions using trained models
        """
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, fold_models in enumerate(self.base_models_):
            meta_features[:, i] = fold_models[-1].predict(X)
            
        return self.meta_model.predict(meta_features)

    def get_meta_features(self, X):
        """
        Get meta-model features for visualization and analysis
        """
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, fold_models in enumerate(self.base_models_):
            meta_features[:, i] = fold_models[-1].predict(X)
            
        return meta_features