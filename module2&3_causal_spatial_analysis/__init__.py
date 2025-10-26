# -*- coding: utf-8 -*-
"""
Module 2&3: Causal Spatial Analysis
Modular livestock analysis system with configuration-driven approach
"""

__version__ = "1.0.0"
__author__ = "Ning"
__description__ = "Unified causal inference and spatial prediction analysis for livestock systems"

from .analysis.livestock_analyzer import LivestockAnalyzer
from .config.livestock_config import get_config, list_available_configs

__all__ = ['LivestockAnalyzer', 'get_config', 'list_available_configs']