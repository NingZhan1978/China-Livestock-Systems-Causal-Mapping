# China Long-term Dataset for Livestock Production Systems (CLD-MLPS & CLD-LLPS)

## Project Overview

This repository presents two high-resolution datasets for China—the China Long-term Dataset for Mixed-farming Livestock Production Systems (CLD-MLPS) and Landless Livestock Production Systems (CLD-LLPS)—which map pigs, cattle, sheep and goats at 1-km resolution from 2000 to 2021. By integrating interpretable ensemble models with causal machine learning, we not only produced high-resolution predictions of livestock distributions but also disentangled their underlying drivers. The research covers three major livestock species (pigs, cattle, sheep/goats) across two production systems, providing a robust evidence base for livestock management, infrastructure planning, and environmental governance.

## Core Features

### Multi-dimensional Analysis Methods
- **Traditional Causal Inference**: OLS regression, Fixed Effects models, Instrumental Variables
- **Modern Machine Learning**: Ensemble learning (RF, ET, XGBoost, LightGBM, CatBoost), Stacking models, SHAP interpretability analysis
- **Causal Discovery**: Double Machine Learning (DML), E-value sensitivity analysis
- **Spatial Prediction**: Nationwide spatial prediction based on raster data

### Modular Architecture
- **Configuration-driven Design**: Unified codebase supporting all species and production systems
- **Extensibility**: Easy addition of new livestock species and production systems
- **Academic Standards**: Compliant with international journal publication requirements

### Comprehensive Visualization
- **Causal Analysis Charts**: DML causal effects, E-value sensitivity analysis
- **Model Performance Assessment**: KDE performance comparison, SHAP importance analysis
- **Spatial Prediction Results**: Nationwide raster prediction visualization

## Project Structure

```
code/
├── module1_livestock_population_segmentation/    # Module 1: Livestock Population Segmentation
│   ├── base_model.py                            # Base model framework
│   ├── pig_model.py                             # Pig intensification model
│   ├── cattle_model.py                          # Cattle intensification model
│   ├── sheep_goat_model.py                      # Sheep/goat intensification model
│   ├── main.py                                  # Main execution script
│   └── settings.py                              # Configuration settings
│
└── module2&3_causal_spatial_analysis/           # Module 2&3: Causal Spatial Analysis
    ├── analysis/
    │   └── livestock_analyzer.py                # Main analyzer
    ├── config/
    │   ├── livestock_config.py                  # Livestock configuration management
    │   └── variable_mappings.py                 # Variable mapping configuration
    ├── core/
    │   ├── data_processor.py                    # Data processor
    │   ├── dml_analyzer.py                      # DML causal analyzer
    │   └── enhanced_stacking.py                 # Enhanced stacking model
    ├── utils/
    │   ├── e_value_utils.py                     # E-value calculation utilities
    │   └── statistical_utils.py                 # Statistical utilities
    ├── visualization/
    │   ├── dml_plots.py                         # DML visualization
    │   ├── kde_plots.py                         # KDE performance plots
    │   ├── shap_plots.py                        # SHAP analysis plots
    │   └── spatial_plots.py                     # Spatial prediction plots
    └── main_analysis.py                         # Main analysis script
```

## Quick Start

### Requirements

```bash
Python >= 3.8
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
xgboost >= 1.5.0
lightgbm >= 3.2.0
catboost >= 1.0.0
matplotlib >= 3.5.0
seaborn >= 0.11.0
shap >= 0.40.0
rasterio >= 1.2.0
statsmodels >= 0.13.0
linearmodels >= 4.27
```

### Installation

```bash
pip install -r requirements.txt
```

### Usage Examples

#### 1. Livestock Intensification Analysis (Module 1)

```python
# Run intensification analysis for all livestock types
python module1_livestock_population_segmentation/main.py --livestock all

# Run analysis for specific livestock type
python module1_livestock_population_segmentation/main.py --livestock pig
```

#### 2. Causal Inference and Spatial Prediction Analysis (Module 2&3)

```python
# Run complete analysis pipeline
python module2&3_causal_spatial_analysis/main_analysis.py
```

#### 3. Specific Configuration Analysis

```python
from module2&3_causal_spatial_analysis.analysis.livestock_analyzer import LivestockAnalyzer

# Landless pig production system analysis
analyzer = LivestockAnalyzer('pig_landless')
results = analyzer.run_full_analysis()

# Mixed-farming cattle production system analysis
analyzer = LivestockAnalyzer('cattle_mixed')
results = analyzer.run_full_analysis()
```

## Supported Configurations

### Production Systems
- **Landless Production Systems (LPS)**: Includes POI variables (company types, POI count, total capital)
- **Mixed-farming Production Systems (MPS)**: Traditional agricultural variables, no POI variables

### Livestock Species
- **Pigs**: Pig intensification analysis
- **Cattle**: Beef/dairy cattle intensification analysis  
- **Sheep/Goats**: Sheep and goat intensification analysis

### Variable Categories
- **Environmental Variables**: DEM, temperature, precipitation, NDVI, soil pH, SOC, nitrogen content, wind speed, snow cover
- **Socio-economic Variables**: Population, GDP, nighttime lights, travel time, OSM, GHSL
- **POI Variables** (LPS only): Company types, POI count, total capital

## Analysis Methods

### Stage 1: Causal Inference Analysis
1. **OLS Regression**: Full variable linear regression analysis
2. **Fixed Effects Models**: Control for time-invariant variables
3. **Instrumental Variables**: Address endogeneity issues

### Stage 2: Prediction Model Analysis
1. **Ensemble Learning**: Random Forest (RF), Extra Trees (ET), XGBoost, LightGBM, CatBoost
2. **Stacking Models**: Multi-model ensemble prediction
3. **SHAP Analysis**: Model interpretability analysis
4. **DML Causal Analysis**: Double Machine Learning causal discovery
5. **E-value Analysis**: Sensitivity analysis

### Stage 3: Spatial Prediction Analysis
1. **Raster Data Processing**: Multi-source spatial data integration
2. **Spatial Prediction**: Nationwide scale prediction
3. **Result Visualization**: Spatial distribution map generation

## Output Results

### Analysis Reports
- **Causal Inference Results**: CSV format regression analysis results
- **Model Performance Assessment**: R², RMSE and other metrics
- **SHAP Importance Analysis**: Feature importance ranking
- **DML Causal Effects**: Causal effect estimates and confidence intervals

### Visualization Charts
- **Model Validation Plots**: Predicted vs observed scatter plots
- **Feature Importance Plots**: SHAP importance analysis
- **Causal Effect Plots**: DML causal effect visualization
- **Spatial Prediction Maps**: Nationwide scale prediction results

### Spatial Data
- **Prediction Rasters**: GeoTIFF format spatial prediction results
- **Statistical Summary**: Prediction statistics

## Configuration

### Data Path Configuration
Configure data paths in `livestock_config.py`:

```python
'data_paths': {
    'response_data': 'path/to/response_data.xlsx',
    'predictor_data': 'path/to/predictor_data.xlsx',
    'spatial_base_path': 'path/to/spatial_data/',
    'reference_file': 'path/to/reference.tif'
}
```

### Output Path Configuration
```python
'output_paths': {
    'visualizations': 'path/to/visualizations/',
    'models': 'path/to/models/',
    'spatial_predictions': 'path/to/predictions/'
}
```

## Technical Features

### Academic Rigor
- **Multiple Validation**: Cross-validation, nested cross-validation
- **Robustness Analysis**: Multi-method consistency testing
- **Sensitivity Analysis**: E-value analysis for unmeasured confounding
- **Reproducibility**: Fixed random seeds, detailed logging

### Engineering Optimization
- **Memory Optimization**: Chunked processing for large-scale spatial data
- **Parallel Computing**: Multi-core parallel training
- **Error Handling**: Comprehensive exception handling mechanisms
- **Progress Monitoring**: Detailed progress display

### Extensibility
- **Configuration-driven**: Support for new species/systems through configuration files
- **Modular Design**: Independent functional modules
- **Standardized Interface**: Unified API design

## Acknowledgments

This code accompanies a manuscript currently under review.

## Citation

Full citation information will be updated upon publication.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions, please contact:

- Email: ningzhan98@163.com
- GitHub Issues: https://github.com/NingZhan1978/China-Livestock-Systems-Causal-Mapping/issues

---

**Note**: This code is developed for academic research purposes. Please ensure you carefully read the related papers and documentation before use to understand the applicability and limitations of the methods.

