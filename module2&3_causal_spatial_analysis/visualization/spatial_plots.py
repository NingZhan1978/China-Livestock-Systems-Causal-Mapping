# -*- coding: utf-8 -*-
"""
Spatial Prediction Visualization
Functions for visualizing spatial prediction results
"""

import os
import matplotlib.pyplot as plt

def create_spatial_visualizations(spatial_results, config, title):
    """
    Create spatial prediction visualizations - only for real results
    
    Parameters:
    - spatial_results: Dictionary containing spatial prediction results
    - config: Spatial configuration dictionary
    - title: Analysis title
    """
    if spatial_results['status'] != 'completed':
        print("Prediction not successfully completed, skipping visualization")
        return
    
    try:
        print("Creating spatial prediction visualizations...")
        
        # Create spatial prediction summary plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{title} Spatial Prediction Results\n(DML-Enhanced Stacking Model)', 
                    fontsize=18, fontweight='bold')
        
        # Subplot 1: Prediction success rate
        ax1 = axes[0, 0]
        total_years = len(config['years'])
        success_years = len(spatial_results['successful_years'])
        fail_years = len(spatial_results['failed_years'])
        
        labels = ['Successful', 'Failed']
        sizes = [success_years, fail_years]
        colors = ['#2ca25f', '#e74c3c']
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                          startangle=90)
        ax1.set_title('Prediction Success Rate', fontsize=16, fontweight='bold')
        
        # Subplot 2: Year distribution
        ax2 = axes[0, 1]
        if spatial_results['successful_years']:
            years_success = spatial_results['successful_years']
            ax2.hist(years_success, bins=20, alpha=0.7, color='#2ca25f', edgecolor='black')
            ax2.set_xlabel('Year', fontweight='bold')
            ax2.set_ylabel('Frequency', fontweight='bold')
            ax2.set_title('Distribution of Successful Years', fontsize=16, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Processing statistics
        ax3 = axes[1, 0]
        stats_text = f"""SPATIAL PREDICTION STATISTICS
{'='*30}

Total Years Processed: {total_years}
Successful Predictions: {success_years}
Failed Predictions: {fail_years}
Success Rate: {success_years/total_years*100:.1f}%

Variables Used: {len(config['variables'])}
Output Directory: 
{config['output_dir']}

Model Features: {len(spatial_results.get('model_features', []))}
Prediction Method: {spatial_results['prediction_method']}
"""
        
        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=11, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8))
        ax3.set_title('Processing Summary', fontsize=16, fontweight='bold')
        ax3.axis('off')
        
        # Subplot 4: Output file information
        ax4 = axes[1, 1]
        if spatial_results['successful_years']:
            first_year = min(spatial_results['successful_years'])
            last_year = max(spatial_results['successful_years'])
            
            files_text = f"""OUTPUT FILES INFORMATION
{'='*25}

File Pattern: 
stacking_pred_*_farm_YYYY.tif

Year Range: {first_year} - {last_year}
Total Files: {len(spatial_results['successful_years'])}

Sample Files:
• stacking_pred_*_farm_{first_year}.tif
• stacking_pred_*_farm_{last_year}.tif

File Format: GeoTIFF
Data Type: Float32
Coordinate System: {spatial_results.get('crs', 'EPSG:4326')}
"""
        else:
            files_text = "No output files generated"
        
        ax4.text(0.05, 0.95, files_text, transform=ax4.transAxes, fontsize=11, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))
        ax4.set_title('Output Files', fontsize=16, fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        
        viz_path = os.path.join(config['output_dir'], 'spatial_prediction_summary.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"Spatial prediction visualization saved to: {viz_path}")
        
    except Exception as e:
        print(f"Spatial visualization creation failed: {e}")