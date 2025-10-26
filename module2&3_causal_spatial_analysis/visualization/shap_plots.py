# -*- coding: utf-8 -*-
"""
SHAP Analysis Plots
Comprehensive SHAP visualization functions
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import shap

def generate_complete_shap_analysis(stacking_model, X_train, X_test, y_test, 
                                   available_features, variable_mapping, 
                                   category_colors, best_model_name,
                                   title, output_dir, system_type):
    """
    Generate complete SHAP analysis - English version with enlarged fonts
    
    Parameters:
    - stacking_model: Trained stacking model
    - X_train, X_test, y_test: Training and test data
    - available_features: List of available features
    - variable_mapping: Variable name mapping
    - category_colors: Category color mapping
    - best_model_name: Name of best performing model
    - title: Analysis title
    - output_dir: Output directory
    - system_type: System type ('landless' or 'mixed_farming')
    """
    
    # SHAP analysis settings
    sample_size = min(500, X_train.shape[0])
    X_sample = X_train.sample(sample_size, random_state=42)

    model_abbreviations = {
        'RandomForestRegressor': 'RF', 'ExtraTreesRegressor': 'ET', 'XGBRegressor': 'XGB',
        'LGBMRegressor': 'LGBM', 'CatBoostRegressor': 'CB'
    }

    print("Generating comprehensive SHAP visualization...")

    fig = plt.figure(figsize=(26, 20), dpi=300)  # Enlarged figure size

    all_importance_data = []
    category_importance_by_model = {}

    # Get trained models
    trained_models = [model_list[-1] for model_list in stacking_model.base_models_]
    models_to_plot = {
        'RF': trained_models[0], 'ET': trained_models[1], 'XGB': trained_models[2],
        'LGBM': trained_models[3], 'CB': trained_models[4]
    }

    # Create subplot for each model
    for model_idx, (model_name, model) in enumerate(models_to_plot.items()):
        print(f"Processing model: {model_name}")
        
        ax = plt.subplot(2, 3, model_idx + 1)
        
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            feature_importance = np.abs(shap_values).mean(0)
            
            # Get variable categories based on system type
            def get_category_for_system(var_name):
                mapped_name = variable_mapping.get(var_name, var_name)
                if mapped_name in ['Company Types', 'POI Count', 'Total Capital'] and system_type == 'mixed_farming':
                    return 'Other'  # Mixed-farming doesn't have POI variables
                elif mapped_name in ['Company Types', 'POI Count', 'Total Capital']:
                    return 'POI-Business'
                elif mapped_name in ['POP', 'Travel Time', 'GDP', 'GHSL', 'OSM', 'NTL']:
                    return 'Social-economy'
                elif mapped_name in ['DEM', 'Snow Cover', 'Annual Temp', 'Annual Precip', 'NDVI', 'pH', 'SOC', 'N', 'Wind']:
                    return 'Environment'
                elif 'Interaction' in mapped_name or '×' in mapped_name:
                    return 'Causal-Interaction'
                else:
                    return 'Other'
            
            model_importance = pd.DataFrame({
                'Feature': X_train.columns,
                'FeatureName': [variable_mapping.get(col, col) for col in X_train.columns],
                'Importance': feature_importance,
                'Category': [get_category_for_system(col) for col in X_train.columns],
                'Model': model_name
            })
            all_importance_data.append(model_importance)
            
            category_importance = model_importance.groupby('Category')['Importance'].sum()
            total_importance = category_importance.sum()
            category_percentages = (category_importance / total_importance * 100).to_dict()
            category_importance_by_model[model_name] = category_percentages
            
            model_importance = model_importance.sort_values('Importance', ascending=False)
            top_features = model_importance.head(15)
            
            top_feature_names = top_features['FeatureName'].values
            top_feature_indices = [list(X_train.columns).index(f) for f in top_features['Feature'].values]
            
            left_margin = 0.0
            max_imp = top_features['Importance'].max()
            
            y_positions = range(len(top_feature_names)-1, -1, -1)
            ax.set_ylim(-0.5, len(top_feature_names) - 0.5)
            
            for i, (_, row) in enumerate(top_features.iterrows()):
                category = row['Category']
                color = category_colors.get(category, '#f0f0f0')
                width = row['Importance']
                
                position = len(top_feature_names) - 1 - i
                
                rect = plt.Rectangle(
                    (left_margin, position - 0.4),
                    width,
                    0.8,
                    color=color,
                    alpha=0.6,
                    zorder=1
                )
                ax.add_patch(rect)
            
            for i, feat_idx in enumerate(top_feature_indices):
                feat_shap_values = shap_values[:, feat_idx]
                feat_values = X_sample.iloc[:, feat_idx].values
                
                sorted_indices = np.argsort(feat_values)
                percentiles = np.linspace(0, 1, len(sorted_indices))
                normalized_values = np.zeros_like(feat_values, dtype=float)
                normalized_values[sorted_indices] = percentiles
                
                dot_size = 40  # Enlarged dot size
                dot_alpha = 0.8
                
                position = len(top_feature_names) - 1 - i
                ys = np.random.normal(position, 0.05, size=len(feat_shap_values))
                
                cmap = LinearSegmentedColormap.from_list(
                    'custom_diverging',
                    [(0, '#0571b0'), (0.5, '#f7f7f7'), (1, '#ca0020')]
                )
                
                sc = ax.scatter(
                    feat_shap_values, ys,
                    c=normalized_values,
                    cmap=cmap,
                    s=dot_size,
                    alpha=dot_alpha,
                    edgecolor='none',
                    zorder=2
                )
            
            ax.axvline(x=0, color='#666666', linestyle='--', alpha=0.7, zorder=0, linewidth=2)
            
            reversed_labels = top_feature_names[::-1]
            ax.set_yticks(range(len(reversed_labels)))
            ax.set_yticklabels(reversed_labels, fontsize=20)  # Enlarged y-axis label font
            
            ax.set_xlabel("SHAP Value (impact on model output)", fontsize=22, fontweight='bold')  # Enlarged x-axis label font
            ax.set_title(f"{model_name}", fontsize=26, fontweight='bold')  # Enlarged title font
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', labelsize=20) # Keep this for font size setting
            for label in ax.get_xticklabels():
                label.set_fontweight('bold') # Iterate and bold X-axis tick labels
        except Exception as e:
            print(f"Model {model_name} SHAP analysis failed: {str(e)}")
            ax.text(0.5, 0.5, f"SHAP analysis failed\n{str(e)[:50]}...", 
                    ha='center', va='center', transform=ax.transAxes, fontsize=18)

    # Meta Model subplot
    if len(models_to_plot) <= 5:
        meta_ax = plt.subplot(2, 3, 6)
        
        try:
            meta_features = stacking_model.get_meta_features(X_train)
            meta_X = pd.DataFrame(
                meta_features,
                columns=[model_abbr for model_abbr in models_to_plot.keys()]
            )
            meta_explainer = shap.TreeExplainer(stacking_model.meta_model)
            meta_shap_values = meta_explainer.shap_values(meta_X.iloc[:sample_size])
            
            meta_importance = np.abs(meta_shap_values).mean(0)
            meta_importance_df = pd.DataFrame({
                'Feature': meta_X.columns,
                'Importance': meta_importance
            })
            
            meta_importance_df = meta_importance_df.sort_values('Importance', ascending=True)
            
            y_pos = np.arange(len(meta_importance_df))
            meta_ax.barh(y_pos, meta_importance_df['Importance'], height=0.6, color='#4682B4', alpha=0.8)
            
            meta_ax.axvline(x=0, color='#666666', linestyle='--', alpha=0.7, zorder=0, linewidth=2)
            
            meta_ax.set_yticks(y_pos)
            meta_ax.set_yticklabels(meta_importance_df['Feature'], fontsize=20)  # Enlarged y-axis label font
            meta_ax.set_xlabel("SHAP Value (impact on model output)", fontsize=22, fontweight='bold')  # Enlarged x-axis label font
            meta_ax.set_title("Meta Model", fontsize=26, fontweight='bold')  # Enlarged title font
            
            meta_ax.spines['top'].set_visible(False)
            meta_ax.spines['right'].set_visible(False)
            meta_ax.grid(axis='x', linestyle='--', alpha=0.3)
            meta_ax.tick_params(axis='x', labelsize=20)  # Enlarged x-axis tick font
            
        except Exception as e:
            print(f"Meta model SHAP analysis failed: {str(e)}")

    # Add colorbar
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = plt.colorbar(sc, cax=cax)
    cbar.set_label('Feature Value (Low to High)', fontsize=20, fontweight='bold')  # Enlarged colorbar label font
    cbar.ax.tick_params(labelsize=20)  # Enlarged colorbar tick font

    # Create category legend
    category_handles = []
    for category, color in category_colors.items():
        if system_type == 'mixed_farming' and category == 'POI-Business':
            continue  # Skip POI-Business for mixed-farming systems
        patch = mpatches.Patch(color=color, alpha=0.8, label=category)
        category_handles.append(patch)

    fig.legend(
        handles=category_handles,
        loc='lower center',
        ncol=len(category_handles),  # Adjust number of columns based on system type
        bbox_to_anchor=(0.45, 0.01),
        title="Feature Categories",
        title_fontsize=24,  # Enlarged legend title font
        fontsize=22,  # Enlarged legend font
        columnspacing=1.5,
        handletextpad=0.5
    )

    fig.suptitle(title, fontsize=30, fontweight='bold', y=0.95)  # Enlarged main title font

    plt.tight_layout(rect=[0, 0.08, 0.9, 0.95])

    shap_path = os.path.join(output_dir, f'{title.replace(" ", "_").replace(":", "").lower()}_combined_shap_dml_{time.strftime("%Y%m%d")}.png')
    plt.savefig(shap_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    print(f"Comprehensive SHAP visualization saved to: {shap_path}")

    # Generate best model SHAP analysis with pie chart
    generate_best_model_shap_with_pie(X_train, X_sample, best_model_name, stacking_model,
                                     variable_mapping, category_colors, title, output_dir, system_type)

    # Extract SHAP importance data
    if all_importance_data:
        best_model_importance = all_importance_data[0]
        
        shap_importance_df = pd.DataFrame({
            'Variable': best_model_importance['Feature'],
            'SHAP_Importance': best_model_importance['Importance'],
            'Mapped_Name': best_model_importance['FeatureName'],
            'Category': best_model_importance['Category']
        })
        
        # Normalize importance
        shap_importance_df['SHAP_Importance_Normalized'] = (
            shap_importance_df['SHAP_Importance'] / shap_importance_df['SHAP_Importance'].max()
        )
        
        return shap_importance_df
    else:
        return None

def generate_best_model_shap_with_pie(X_train, X_sample, best_model_name, stacking_model,
                                     variable_mapping, category_colors, title, output_dir, system_type):
    """
    Generate best model SHAP analysis with pie chart - English version with enlarged fonts
    """
    print("\nCreating detailed SHAP analysis for best model (with pie chart)...")

    try:
        trained_models = [model_list[-1] for model_list in stacking_model.base_models_]
        best_model_for_shap = trained_models[0]
        
        fig_best = plt.figure(figsize=(18, 20), dpi=300)  # Enlarged figure size
        
        explainer = shap.TreeExplainer(best_model_for_shap)
        shap_values = explainer.shap_values(X_sample)
        
        feature_importance = np.abs(shap_values).mean(0)
        
        # Get variable categories based on system type
        def get_category_for_system(var_name):
            mapped_name = variable_mapping.get(var_name, var_name)
            if mapped_name in ['Company Types', 'POI Count', 'Total Capital'] and system_type == 'mixed_farming':
                return 'Other'  # Mixed-farming doesn't have POI variables
            elif mapped_name in ['Company Types', 'POI Count', 'Total Capital']:
                return 'POI-Business'
            elif mapped_name in ['POP', 'Travel Time', 'GDP', 'GHSL', 'OSM', 'NTL']:
                return 'Social-economy'
            elif mapped_name in ['DEM', 'Snow Cover', 'Annual Temp', 'Annual Precip', 'NDVI', 'pH', 'SOC', 'N', 'Wind']:
                return 'Environment'
            elif 'Interaction' in mapped_name or '×' in mapped_name:
                return 'Causal-Interaction'
            else:
                return 'Other'
        
        importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'FeatureName': [variable_mapping.get(col, col) for col in X_train.columns],
            'Importance': feature_importance,
            'Category': [get_category_for_system(col) for col in X_train.columns]
        })
        
        category_importance = importance_df.groupby('Category')['Importance'].sum()
        total_importance = category_importance.sum()
        category_percentages = (category_importance / total_importance * 100).to_dict()
        
        importance_df = importance_df.sort_values('Importance', ascending=False)
        top_features = importance_df.head(20)
        
        top_feature_names = top_features['FeatureName'].values
        top_feature_indices = [list(X_train.columns).index(f) for f in top_features['Feature'].values]
        
        # Main plot position
        ax = plt.axes([0.15, 0.08, 0.62, 0.80])
        
        y_positions = range(len(top_feature_names)-1, -1, -1)
        ax.set_ylim(-0.5, len(top_feature_names) - 0.5)
        
        for i, (_, row) in enumerate(top_features.iterrows()):
            category = row['Category']
            color = category_colors.get(category, '#f0f0f0')
            width = row['Importance']
            
            position = len(top_feature_names) - 1 - i
            
            rect = plt.Rectangle(
                (0, position - 0.4),
                width,
                0.8,
                color=color,
                alpha=0.6,
                zorder=1
            )
            ax.add_patch(rect)
        
        for i, feat_idx in enumerate(top_feature_indices):
            feat_shap_values = shap_values[:, feat_idx]
            feat_values = X_sample.iloc[:, feat_idx].values
            
            sorted_indices = np.argsort(feat_values)
            percentiles = np.linspace(0, 1, len(sorted_indices))
            normalized_values = np.zeros_like(feat_values, dtype=float)
            normalized_values[sorted_indices] = percentiles
            
            dot_size = 70  # Enlarged dot size
            dot_alpha = 0.85
            
            position = len(top_feature_names) - 1 - i
            ys = np.random.normal(position, 0.07, size=len(feat_shap_values))
            
            cmap = LinearSegmentedColormap.from_list(
                'custom_diverging',
                [(0, '#0571b0'), (0.5, '#f7f7f7'), (1, '#ca0020')]
            )
            
            sc_best = ax.scatter(
                feat_shap_values, ys,
                c=normalized_values,
                cmap=cmap,
                s=dot_size,
                alpha=dot_alpha,
                edgecolor='none',
                zorder=2
            )
        
        ax.axvline(x=0, color='#666666', linestyle='--', alpha=0.7, zorder=0, linewidth=2)
        
        reversed_labels = top_feature_names[::-1]
        ax.set_yticks(range(len(reversed_labels)))
        ax.set_yticklabels(reversed_labels, fontsize=26) # Keep this line for font size setting
        for label in ax.get_yticklabels():
            label.set_fontweight('bold') # Iterate and bold all Y-axis tick labels
        ax.tick_params(axis='x', labelsize=22)  # Enlarged x-axis tick font
        
        ax.set_xlabel("SHAP Value (impact on model output)", fontsize=24, fontweight='bold')  # Enlarged x-axis label font
        
        # Enlarged pie chart
        pie_ax = fig_best.add_axes([0.78, 0.68, 0.20, 0.20])
        
        categories = []
        values = []
        colors = []
        
        for cat, percent in sorted(category_percentages.items(), key=lambda x: x[1], reverse=True):
            if percent > 0:
                categories.append(cat)
                values.append(percent)
                colors.append(category_colors.get(cat, '#f0f0f0'))
        
        wedges, texts, autotexts = pie_ax.pie(
            values, 
            labels=None,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            wedgeprops={'alpha': 0.8, 'linewidth': 1.5, 'edgecolor': 'white'}
        )
        
        for autotext in autotexts:
            autotext.set_fontsize(20)  # Enlarged pie chart percentage font
            autotext.set_weight('bold')
            autotext.set_color('black')
        
        pie_ax.set_title("Feature Category\nDistribution", fontsize=26, fontweight='bold')  # Enlarged pie chart title font
        
        # Colorbar
        cbar_ax = fig_best.add_axes([0.82, 0.25, 0.02, 0.35])
        cbar = plt.colorbar(sc_best, cax=cbar_ax)
        cbar.set_label('Feature Value\n(Low to High)', fontsize=26, fontweight='bold')  # Enlarged colorbar label font
        cbar.ax.tick_params(labelsize=22)  # Enlarged colorbar tick font
        
        fig_best.suptitle(f"{title}: {best_model_name} Model", 
                         fontsize=30, fontweight='bold', y=0.93)  # Enlarged main title font
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Legend
        category_handles = []
        for category, color in category_colors.items():
            if system_type == 'mixed_farming' and category == 'POI-Business':
                continue  # Skip POI-Business for mixed-farming systems
            patch = mpatches.Patch(color=color, alpha=0.7, label=category)
            category_handles.append(patch)
        
        legend = fig_best.legend(
            handles=category_handles,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.02),
            ncol=len(category_handles),  # Adjust number of columns based on system type
            title="Feature Categories",
            title_fontsize=24,  # Enlarged legend title font
            fontsize=22,  # Enlarged legend font
            frameon=True,
            fancybox=True,
            shadow=True,
            columnspacing=1.5,
            handletextpad=0.5
        )
        
        legend.get_title().set_fontweight('bold')
        
        plt.subplots_adjust(left=0.15, right=0.77, top=0.9, bottom=0.06)
        
        best_model_path = os.path.join(output_dir, f'{title.replace(" ", "_").replace(":", "").lower()}_best_model_shap_dml_{best_model_name}_{time.strftime("%Y%m%d")}.png')
        plt.savefig(best_model_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.show()
        plt.close(fig_best)
        
        print(f"Best model SHAP analysis (with pie chart) saved to: {best_model_path}")
        
    except Exception as e:
        print(f"Creating best model SHAP analysis failed: {str(e)}")