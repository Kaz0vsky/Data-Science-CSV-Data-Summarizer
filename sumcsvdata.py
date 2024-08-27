import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau
import json
import sys
import os

def summarize_csv(file_path):
    df = pd.read_csv(file_path)

    # Selecting only numerical columns for summary statistics
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Basic statistics
    summary = numeric_df.describe().transpose()
    summary['mode'] = numeric_df.mode().iloc[0]
    summary['median'] = numeric_df.median()

    # Handling missing data
    missing_data = df.isnull().sum()
    summary['missing_data'] = missing_data[numeric_df.columns]

    # Outlier detection using IQR
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).sum()
    summary['outliers'] = outliers

    # Correlation analysis
    correlations = {}
    for col1 in numeric_df.columns:
        for col2 in numeric_df.columns:
            if col1 != col2:
                # Drop rows with missing data in either column
                valid_data = numeric_df[[col1, col2]].dropna()
                correlations[(col1, col2)] = {
                    'pearson': pearsonr(valid_data[col1], valid_data[col2])[0],
                    'spearman': spearmanr(valid_data[col1], valid_data[col2])[0],
                    'kendall': kendalltau(valid_data[col1], valid_data[col2])[0],
                }
    correlation_df = pd.DataFrame(correlations).transpose()

    return summary, correlation_df

def visualize_data(df, output_dir):
    sns.set(style="whitegrid")
    for column in df.select_dtypes(include=[np.number]).columns:
        plt.figure(figsize=(10, 6))

        # Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(df[column].dropna(), kde=True)
        plt.title(f'Histogram of {column}')

        # Box Plot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[column])
        plt.title(f'Box Plot of {column}')

        plt.savefig(os.path.join(output_dir, f'{column}_summary.png'))
        plt.close()

    # Pair Plot
    sns.pairplot(df.dropna(), diag_kind='kde')
    plt.savefig(os.path.join(output_dir, 'pairplot.png'))
    plt.close()

def export_report(summary, correlation_df, output_dir):
    summary_file = os.path.join(output_dir, 'summary.json')
    correlation_file = os.path.join(output_dir, 'correlations.json')

    # Export summary to JSON
    summary.to_json(summary_file, indent=4)
    
    # Export correlations to JSON
    correlation_df.to_json(correlation_file, indent=4)

    # Export HTML summary
    html_summary = summary.to_html()
    with open(os.path.join(output_dir, 'summary.html'), 'w') as f:
        f.write(html_summary)

def main(file_path):
    # Create output directory
    output_dir = "csv_summary_report"
    os.makedirs(output_dir, exist_ok=True)

    # Generate summary
    df = pd.read_csv(file_path)
    summary, correlation_df = summarize_csv(file_path)

    # Visualize data
    visualize_data(df, output_dir)

    # Export summary and correlations
    export_report(summary, correlation_df, output_dir)

    print(f"Summary and visualizations saved in {output_dir} directory.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 sumcsvdata.py <path_to_csv_file>")
    else:
        file_path = sys.argv[1]
        main(file_path)
