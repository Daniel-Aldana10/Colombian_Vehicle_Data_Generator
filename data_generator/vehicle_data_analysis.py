#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


class VehicleDataAnalyzer:
    """
    Analyzer for Colombian vehicle synthetic data.
    Creates visualizations to understand data distributions and correlations.
    """

    def __init__(self, csv_path, output_dir="analysis_results"):
        """
        Initialize the analyzer with data file path and output directory.

        Parameters:
        -----------
        csv_path : str
            Path to the CSV file containing vehicle data
        output_dir : str
            Directory to save visualization outputs
        """
        self.df = pd.read_csv(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Convert date columns to datetime
        date_columns = ['soat_validity', 'techinsp_date', 'techinsp_validity', 'registration_date']
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col])

        print(f"Loaded {len(self.df)} records from {csv_path}")
        print(f"Output directory: {self.output_dir}")

    def save_fig(self, fig, filename):
        """Save figure to output directory with specified filename"""
        filepath = self.output_dir / filename
        fig.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Saved: {filepath}")

    def analyze_categorical_distributions(self):
        """Create pie charts and bar plots for categorical variables"""
        categorical_cols = [
            'brand', 'model', 'color', 'body_type', 'vehicle_type',
            'fuel_type', 'department', 'municipality', 'emission_standard'
        ]

        for col in categorical_cols:
            if col not in self.df.columns:
                continue

            # Get value counts
            counts = self.df[col].value_counts()

            # Create pie chart
            fig, ax = plt.subplots(figsize=(10, 7))
            if len(counts) > 10:  # Too many categories for a pie chart
                # Take top 10 and group others
                others = pd.Series({'Others': counts[10:].sum()})
                counts = pd.concat([counts[:10], others])

            counts.plot.pie(
                autopct='%1.1f%%',
                startangle=90,
                shadow=False,
                ax=ax,
                explode=[0.05] + [0] * (len(counts) - 1)  # Explode first slice
            )
            ax.set_title(f'Distribution of {col.replace("_", " ").title()}')
            ax.set_ylabel('')
            self.save_fig(fig, f'pie_{col}.png')

            # Create bar plot
            fig, ax = plt.subplots(figsize=(12, 6))
            if col == 'model':  # Group by brand for models
                top_models = self.df.groupby('brand')['model'].value_counts().groupby(level=0).head(3)
                top_models.unstack().plot.bar(ax=ax, stacked=False)
                ax.set_title('Top 3 Models by Brand')
                ax.set_ylabel('Count')
                ax.legend(title='Model')
            else:
                counts[:15].plot.bar(ax=ax)
                ax.set_title(f'Top {min(15, len(counts))} {col.replace("_", " ").title()} Categories')

            ax.set_xlabel(col.replace("_", " ").title())
            plt.xticks(rotation=45, ha='right')
            self.save_fig(fig, f'bar_{col}.png')

    def analyze_numeric_distributions(self):
        """Create histograms for numeric variables"""
        numeric_cols = [
            'year', 'engine_displacement', 'doors', 'seats', 'wheels',
            'weight_kg', 'mileage_km', 'price_cop', 'airbags'
        ]

        # Distribution histograms
        for col in numeric_cols:
            if col not in self.df.columns:
                continue

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(self.df[col], kde=True, ax=ax)
            ax.set_title(f'Distribution of {col.replace("_", " ").title()}')
            ax.set_xlabel(col.replace("_", " ").title())
            ax.set_ylabel('Count')

            # Add mean and median lines
            mean_val = self.df[col].mean()
            median_val = self.df[col].median()
            ax.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='g', linestyle='-.', label=f'Median: {median_val:.2f}')
            ax.legend()

            self.save_fig(fig, f'hist_{col}.png')

    def analyze_correlations(self):
        """Create correlation heatmap for numeric variables"""
        numeric_cols = [
            'year', 'engine_displacement', 'doors', 'seats', 'wheels',
            'weight_kg', 'mileage_km', 'price_cop', 'airbags'
        ]

        # Only include columns that exist in the dataframe
        numeric_cols = [col for col in numeric_cols if col in self.df.columns]

        # Create correlation matrix
        corr_matrix = self.df[numeric_cols].corr()

        # Heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            fmt='.2f',
            linewidths=0.5,
            ax=ax
        )
        ax.set_title('Correlation Matrix of Numeric Variables')
        self.save_fig(fig, 'correlation_matrix.png')

    def analyze_relationships(self):
        """Create plots showing relationships between different variables"""

        # 1. Price vs Year scatter plot with trend line
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(
            x='year',
            y='price_cop',
            data=self.df,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red'},
            ax=ax
        )
        ax.set_title('Price vs. Year')
        ax.set_xlabel('Year')
        ax.set_ylabel('Price (COP)')
        self.save_fig(fig, 'scatter_price_year.png')

        # 2. Price vs Mileage scatter plot with trend line
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(
            x='mileage_km',
            y='price_cop',
            data=self.df,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red'},
            ax=ax
        )
        ax.set_title('Price vs. Mileage')
        ax.set_xlabel('Mileage (km)')
        ax.set_ylabel('Price (COP)')
        self.save_fig(fig, 'scatter_price_mileage.png')

        # 3. Vehicle type vs price boxplot
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.boxplot(
            x='vehicle_type',
            y='price_cop',
            data=self.df,
            ax=ax
        )
        ax.set_title('Price Distribution by Vehicle Type')
        ax.set_xlabel('Vehicle Type')
        ax.set_ylabel('Price (COP)')
        plt.xticks(rotation=45, ha='right')
        self.save_fig(fig, 'boxplot_price_vehicle_type.png')

        # 4. Mileage by vehicle type boxplot
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.boxplot(
            x='vehicle_type',
            y='mileage_km',
            data=self.df,
            ax=ax
        )
        ax.set_title('Mileage Distribution by Vehicle Type')
        ax.set_xlabel('Vehicle Type')
        ax.set_ylabel('Mileage (km)')
        plt.xticks(rotation=45, ha='right')
        self.save_fig(fig, 'boxplot_mileage_vehicle_type.png')

        # 5. Brand vs Price boxplot
        fig, ax = plt.subplots(figsize=(15, 8))
        sns.boxplot(
            x='brand',
            y='price_cop',
            data=self.df,
            ax=ax
        )
        ax.set_title('Price Distribution by Brand')
        ax.set_xlabel('Brand')
        ax.set_ylabel('Price (COP)')
        plt.xticks(rotation=45, ha='right')
        self.save_fig(fig, 'boxplot_price_brand.png')

        # 6. Year distribution by brand
        fig, ax = plt.subplots(figsize=(15, 8))
        sns.violinplot(
            x='brand',
            y='year',
            data=self.df,
            inner='quartile',
            ax=ax
        )
        ax.set_title('Year Distribution by Brand')
        ax.set_xlabel('Brand')
        ax.set_ylabel('Year')
        plt.xticks(rotation=45, ha='right')
        self.save_fig(fig, 'violin_year_brand.png')

    def analyze_depreciation(self):
        """Analyze vehicle depreciation patterns"""

        # Group by brand and year to see depreciation patterns
        if all(col in self.df.columns for col in ['brand', 'year', 'price_cop']):
            # Calculate average price by brand and year
            avg_price = self.df.groupby(['brand', 'year'])['price_cop'].mean().reset_index()

            # Plot depreciation curves for each brand
            fig, ax = plt.subplots(figsize=(12, 8))

            for brand in self.df['brand'].unique():
                brand_data = avg_price[avg_price['brand'] == brand]
                if len(brand_data) > 3:  # Only plot brands with sufficient data points
                    ax.plot(
                        brand_data['year'],
                        brand_data['price_cop'],
                        marker='o',
                        linewidth=2,
                        label=brand
                    )

            ax.set_title('Depreciation Curves by Brand')
            ax.set_xlabel('Year')
            ax.set_ylabel('Average Price (COP)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            self.save_fig(fig, 'depreciation_curves.png')

    def analyze_geographical_distribution(self):
        """Analyze geographical distribution of vehicles"""

        if 'department' in self.df.columns:
            # Department distribution
            dept_counts = self.df['department'].value_counts()

            fig, ax = plt.subplots(figsize=(12, 8))
            dept_counts.plot(
                kind='bar',
                color=sns.color_palette("viridis", len(dept_counts)),
                ax=ax
            )
            ax.set_title('Vehicle Distribution by Department')
            ax.set_xlabel('Department')
            ax.set_ylabel('Number of Vehicles')
            plt.xticks(rotation=45, ha='right')
            self.save_fig(fig, 'geo_departments.png')

            # Stacked bar of vehicle types by department
            if 'vehicle_type' in self.df.columns:
                pivot_data = pd.crosstab(
                    self.df['department'],
                    self.df['vehicle_type'],
                    normalize='index'
                )

                fig, ax = plt.subplots(figsize=(14, 8))
                pivot_data.plot(
                    kind='bar',
                    stacked=True,
                    ax=ax,
                    colormap='tab10'
                )
                ax.set_title('Vehicle Type Distribution by Department')
                ax.set_xlabel('Department')
                ax.set_ylabel('Proportion')
                ax.legend(title='Vehicle Type')
                plt.xticks(rotation=45, ha='right')
                self.save_fig(fig, 'geo_vehicle_types.png')

    def create_dashboard_summary(self):
        """Create a dashboard summary of key statistics"""

        # Prepare a 2x3 grid for summary plots
        fig = plt.figure(figsize=(18, 15))
        fig.suptitle('Colombian Vehicle Data Analysis Dashboard', fontsize=16)

        # 1. Brand distribution
        ax1 = fig.add_subplot(2, 3, 1)
        self.df['brand'].value_counts().plot.pie(
            autopct='%1.1f%%',
            startangle=90,
            ax=ax1
        )
        ax1.set_title('Brand Distribution')
        ax1.set_ylabel('')

        # 2. Vehicle type distribution
        ax2 = fig.add_subplot(2, 3, 2)
        self.df['vehicle_type'].value_counts().plot.bar(ax=ax2)
        ax2.set_title('Vehicle Type Distribution')
        ax2.set_xlabel('')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

        # 3. Year distribution
        ax3 = fig.add_subplot(2, 3, 3)
        sns.histplot(self.df['year'], kde=True, ax=ax3)
        ax3.set_title('Year Distribution')

        # 4. Fuel type distribution
        ax4 = fig.add_subplot(2, 3, 4)
        self.df['fuel_type'].value_counts().plot.bar(ax=ax4)
        ax4.set_title('Fuel Type Distribution')
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')

        # 5. Price distribution
        ax5 = fig.add_subplot(2, 3, 5)
        sns.histplot(self.df['price_cop'], kde=True, ax=ax5)
        ax5.set_title('Price Distribution')

        # 6. Mileage vs Price
        ax6 = fig.add_subplot(2, 3, 6)
        sns.scatterplot(
            x='mileage_km',
            y='price_cop',
            hue='vehicle_type',
            size='year',
            sizes=(20, 200),
            alpha=0.5,
            data=self.df.sample(min(1000, len(self.df))),  # Sample to avoid overcrowding
            ax=ax6
        )
        ax6.set_title('Mileage vs Price')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.save_fig(fig, 'dashboard_summary.png')

    def run_all_analyses(self):
        """Run all analysis methods"""
        print("Starting analysis of vehicle data...")
        self.analyze_categorical_distributions()
        self.analyze_numeric_distributions()
        self.analyze_correlations()
        self.analyze_relationships()
        self.analyze_depreciation()
        self.analyze_geographical_distribution()
        self.create_dashboard_summary()
        print("Analysis complete! All visualizations saved to", self.output_dir)


def main():
    """Main function to parse arguments and run analysis"""
    parser = argparse.ArgumentParser(description='Analyze Colombian Vehicle Dataset')
    parser.add_argument(
        '-i', '--input',
        type=str,
        default='vehicle_data.csv',
        help='Input CSV file path'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='analysis_results',
        help='Output directory for visualizations'
    )
    args = parser.parse_args()

    analyzer = VehicleDataAnalyzer(args.input, args.output_dir)
    analyzer.run_all_analyses()


if __name__ == "__main__":
    main()