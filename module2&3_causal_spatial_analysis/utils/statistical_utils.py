# -*- coding: utf-8 -*-
"""
Statistical Analysis Utilities
Robustness assessment for causal inference results
"""

import numpy as np
import pandas as pd

def calculate_robustness_score(result):
    """
    Calculate robustness score based on consistency across OLS, FE, IV methods
    
    Parameters:
    - result: Dictionary containing OLS, FE, IV results
    
    Returns:
    - score: Robustness score (0-1)
    """
    score = 0
    
    # Significance consistency (30%)
    significant_flags = [result['OLS_Significant'], result['FE_Significant'], result['IV_Significant']]
    valid_significant_flags = [flag for flag, pval in zip(significant_flags, [result['OLS_PValue'], result['FE_PValue'], result['IV_PValue']]) if not pd.isna(pval)]
    
    if len(valid_significant_flags) > 0:
        significant_count = sum(valid_significant_flags)
        score += (significant_count / len(valid_significant_flags)) * 0.3
    
    # Effect direction consistency (40%)
    coeffs = [result['OLS_Coef'], result['FE_Coef'], result['IV_Coef']]
    valid_coeffs = [c for c in coeffs if not pd.isna(c) and abs(c) > 1e-6]
    
    if len(valid_coeffs) > 0:
        signs = [np.sign(c) for c in valid_coeffs]
        direction_consistency = len([s for s in signs if s == signs[0]]) / len(signs)
        score += direction_consistency * 0.4
    
    # P-value strength (30%)
    p_values = [result['OLS_PValue'], result['FE_PValue'], result['IV_PValue']]
    valid_p_values = [p for p in p_values if not pd.isna(p)]
    
    if len(valid_p_values) > 0:
        avg_p_value = np.mean(valid_p_values)
        p_score = max(0, 1 - avg_p_value)
        score += p_score * 0.3
    
    return min(score, 1.0)