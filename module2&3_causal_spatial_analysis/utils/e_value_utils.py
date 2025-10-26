# -*- coding: utf-8 -*-
"""
E-value Calculation Utilities
For sensitivity analysis in causal inference
"""

import numpy as np
from scipy import stats

def calculate_e_value(estimate, se, confidence_level=0.95):
    """
    Calculate E-value for sensitivity analysis
    
    Parameters:
    - estimate: Effect estimate value
    - se: Standard error
    - confidence_level: Confidence level
    
    Returns:
    - e_value_point: E-value for point estimate
    - e_value_ci: E-value for confidence interval
    """
    
    # Calculate confidence interval
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    ci_lower = estimate - z_score * se
    ci_upper = estimate + z_score * se
    
    # Convert to risk ratio (assuming effect is on log scale)
    rr_point = np.exp(abs(estimate))
    rr_ci_bound = np.exp(abs(ci_lower if abs(ci_lower) < abs(ci_upper) else ci_upper))
    
    # Calculate E-value
    def e_value_from_rr(rr):
        if rr <= 1:
            return 1
        return rr + np.sqrt(rr * (rr - 1))
    
    e_value_point = e_value_from_rr(rr_point)
    e_value_ci = e_value_from_rr(rr_ci_bound)
    
    return e_value_point, e_value_ci

def interpret_e_value(e_value):
    """
    Interpret E-value strength
    
    Parameters:
    - e_value: E-value to interpret
    
    Returns:
    - interpretation: String interpretation
    """
    if e_value >= 2.0:
        return "Strong robustness"
    elif e_value >= 1.25:
        return "Moderate robustness"
    else:
        return "Limited robustness"

def calculate_e_value_threshold(desired_robustness="moderate"):
    """
    Calculate E-value threshold for desired robustness level
    
    Parameters:
    - desired_robustness: "strong" or "moderate"
    
    Returns:
    - threshold: E-value threshold
    """
    if desired_robustness.lower() == "strong":
        return 2.0
    elif desired_robustness.lower() == "moderate":
        return 1.25
    else:
        return 1.0