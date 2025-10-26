

"""
Livestock Intensification Prediction Models
A comprehensive framework for predicting livestock farming intensification rates
"""

__version__ = "1.0.0"
__author__ = "Research Team"

from .pig_model import PigIntensificationModel
from .cattle_model import CattleIntensificationModel
from .sheep_model import SheepIntensificationModel

__all__ = [
    'PigIntensificationModel',
    'CattleIntensificationModel', 
    'SheepIntensificationModel'
]

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'xgboost', 
        'matplotlib', 'shap', 'scipy', 'openpyxl'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    return True