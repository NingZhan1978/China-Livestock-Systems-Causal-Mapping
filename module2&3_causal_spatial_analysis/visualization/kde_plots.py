# -*- coding: utf-8 -*-
"""
KDE Performance Plots
Kernel Density Estimation plots for model performance visualization
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

def plot_kde_performance(y_test, y_pred_stacking, stacking_r2, stacking_rmse, title, output_dir):
    """
    Generate KDE performance plot
    
    Parameters:
    - y_test: True values
    - y_pred_stacking: Predicted values
    - stacking_r2: R-squared value
    - stacking_rmse: RMSE value
    - title: Plot title
    - output_dir: Output directory
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Clean data, remove invalid values
    mask = ~np.isnan(y_test) & ~np.isnan(y_pred_stacking)
    y_clean = y_test[mask]
    y_pred_clean = y_pred_stacking[mask]

    # Auto-adjust axis range
    min_val = min(y_clean.min(), y_pred_clean.min())
    max_val = max(y_clean.max(), y_pred_clean.max())

    # Add some margin
    range_margin = (max_val - min_val) * 0.05
    min_val = min_val - range_margin
    max_val = max_val + range_margin

    # Create KDE plot
    sns.kdeplot(
        data=pd.DataFrame({'Observed': y_clean, 'Predicted': y_pred_clean}),
        x='Observed',
        y='Predicted',
        cmap="Blues",
        fill=True,
        alpha=0.5,
        levels=20,
        bw_adjust=0.8,
        ax=ax
    )

    # Add diagonal line
    ax.plot([min_val, max_val], [min_val, max_val], "--", color="gray", alpha=0.8, linewidth=2)

    ax.set_xlabel("ln(Observed Livestock Number)", fontsize=18, fontweight='bold')
    ax.set_ylabel("ln(Predicted Livestock Number)", fontsize=18, fontweight='bold')
    ax.set_title(title, fontsize=20, fontweight='bold', pad=15)

    # Set axis range
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # Add performance metrics
    ax.text(0.05, 0.95,
           f"R$^2$: {r2_score(y_clean, y_pred_clean):.3f}",
           transform=ax.transAxes,
           verticalalignment='top',
           fontsize=16,
           color='DimGrey',
           fontweight='bold')

    ax.text(0.05, 0.89,
           f"RMSE: {np.sqrt(mean_squared_error(y_clean, y_pred_clean)):.3f}",
           transform=ax.transAxes,
           verticalalignment='top',
           fontsize=16,
           color='DimGrey',
           fontweight='bold')

    ax.text(0.70, 0.06,
           "DML-Stacking",
           transform=ax.transAxes,
           verticalalignment='top',
           fontsize=16,
           color='DimGrey',
           fontweight='bold')

    # Grid and styling
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=16)

    plt.tight_layout()

    # Save plot
    kde_output_path = os.path.join(
        output_dir,
        f'{title.replace(" ", "_").replace(":", "").lower()}_kde_performance_dml.png'
    )
    plt.savefig(kde_output_path, dpi=300, bbox_inches='tight')
    print(f"KDE performance plot saved to: {kde_output_path}")
    plt.show()
    plt.close()