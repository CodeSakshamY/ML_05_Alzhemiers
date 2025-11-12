#!/usr/bin/env python3
"""
Sample Data Generator for Elastic Net Alzheimer's Classifier Testing

This script generates synthetic biomarker data for testing the classifier.
The generated data mimics realistic biomarker patterns for AD, MCI, and CN groups.

Usage:
    python generate_sample_data.py --output sample_data.xlsx --n-samples 400
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Define realistic biomarker ranges for each diagnosis
# Based on literature values (approximations)
BIOMARKER_PARAMS = {
    'AD': {
        'Abeta40': (120, 140, 15),      # (mean, median, std)
        'Abeta42': (35, 40, 8),
        'Abeta ratio': (0.28, 0.32, 0.08),
        'Ptau-': (60, 70, 15),
        'tau': (300, 320, 50),
        'DHA': (1.0, 1.2, 0.3),
        'Formic acid': (0.40, 0.45, 0.10),
        'lactoferrin': (22, 24, 4),
        'AD7C-NTP': (14, 16, 3)
    },
    'MCI': {
        'Abeta40': (128, 135, 12),
        'Abeta42': (48, 52, 10),
        'Abeta ratio': (0.37, 0.40, 0.07),
        'Ptau-': (40, 45, 12),
        'tau': (260, 280, 45),
        'DHA': (1.3, 1.5, 0.3),
        'Formic acid': (0.35, 0.38, 0.09),
        'lactoferrin': (19, 21, 3.5),
        'AD7C-NTP': (11, 13, 2.5)
    },
    'CN': {
        'Abeta40': (132, 138, 10),
        'Abeta42': (60, 65, 12),
        'Abeta ratio': (0.45, 0.48, 0.08),
        'Ptau-': (22, 25, 8),
        'tau': (200, 220, 40),
        'DHA': (1.6, 1.8, 0.35),
        'Formic acid': (0.28, 0.30, 0.08),
        'lactoferrin': (17, 19, 3),
        'AD7C-NTP': (7, 9, 2)
    }
}

FEATURE_NAMES = [
    'Abeta40', 'Abeta42', 'Abeta ratio', 'Ptau-', 'tau',
    'DHA', 'Formic acid', 'lactoferrin', 'AD7C-NTP'
]


def generate_biomarker_value(mean: float, median: float, std: float, n: int = 1) -> np.ndarray:
    """
    Generate biomarker values using a skewed normal distribution.

    Args:
        mean: Target mean value
        median: Target median value
        std: Standard deviation
        n: Number of samples to generate

    Returns:
        Array of generated values
    """
    # Use log-normal to create realistic skewed distribution
    # Then scale to match desired parameters
    values = np.random.normal(median, std, n)

    # Add some noise and ensure non-negative values
    values = np.abs(values)

    # Add occasional outliers (5% of samples)
    outlier_mask = np.random.random(n) < 0.05
    values[outlier_mask] += np.random.uniform(-std*2, std*2, outlier_mask.sum())

    return np.maximum(values, 0.01)  # Ensure positive values


def generate_sample_data(
    n_samples: int = 400,
    class_distribution: dict = None
) -> pd.DataFrame:
    """
    Generate synthetic biomarker data for AD, MCI, and CN classes.

    Args:
        n_samples: Total number of samples to generate
        class_distribution: Dictionary with class proportions (default: balanced)

    Returns:
        DataFrame with generated data
    """
    if class_distribution is None:
        # Default: relatively balanced with slight imbalance (realistic)
        class_distribution = {'AD': 0.35, 'MCI': 0.32, 'CN': 0.33}

    # Calculate samples per class
    n_ad = int(n_samples * class_distribution['AD'])
    n_mci = int(n_samples * class_distribution['MCI'])
    n_cn = n_samples - n_ad - n_mci  # Ensure exact total

    print(f"Generating {n_samples} samples:")
    print(f"  AD: {n_ad} samples ({n_ad/n_samples*100:.1f}%)")
    print(f"  MCI: {n_mci} samples ({n_mci/n_samples*100:.1f}%)")
    print(f"  CN: {n_cn} samples ({n_cn/n_samples*100:.1f}%)")

    data_list = []

    # Generate data for each class
    for diagnosis, n_class in [('AD', n_ad), ('MCI', n_mci), ('CN', n_cn)]:
        params = BIOMARKER_PARAMS[diagnosis]

        class_data = {
            'diagnosis': [diagnosis] * n_class
        }

        for feature in FEATURE_NAMES:
            mean, median, std = params[feature]
            values = generate_biomarker_value(mean, median, std, n_class)
            class_data[feature] = values

        class_df = pd.DataFrame(class_data)
        data_list.append(class_df)

    # Combine all classes
    df = pd.concat(data_list, ignore_index=True)

    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Add a patient ID column
    df.insert(0, 'patient_id', [f'P{i:04d}' for i in range(len(df))])

    # Optionally add some missing values (realistic scenario)
    # Randomly remove 2% of feature values (but keep all diagnoses)
    for feature in FEATURE_NAMES:
        missing_mask = np.random.random(len(df)) < 0.02
        df.loc[missing_mask, feature] = np.nan

    return df


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics of the generated data."""
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    print(f"Total samples: {len(df)}")
    print(f"Total features: {len(FEATURE_NAMES)}")

    print("\nClass distribution:")
    print(df['diagnosis'].value_counts().to_string())

    print("\nMissing values per feature:")
    missing = df[FEATURE_NAMES].isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0].to_string())
    else:
        print("No missing values")

    print("\nFeature statistics by diagnosis:")
    print("-" * 70)
    for diagnosis in ['AD', 'MCI', 'CN']:
        print(f"\n{diagnosis}:")
        subset = df[df['diagnosis'] == diagnosis][FEATURE_NAMES]
        print(subset.describe().loc[['mean', 'std']].round(2).to_string())

    print("\n" + "="*70)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic biomarker data for Alzheimer\'s classification'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='sample_alzheimer_data.xlsx',
        help='Output Excel file path (default: sample_alzheimer_data.xlsx)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=400,
        help='Number of samples to generate (default: 400)'
    )
    parser.add_argument(
        '--ad-ratio',
        type=float,
        default=0.35,
        help='Proportion of AD samples (default: 0.35)'
    )
    parser.add_argument(
        '--mci-ratio',
        type=float,
        default=0.32,
        help='Proportion of MCI samples (default: 0.32)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Validate ratios
    cn_ratio = 1.0 - args.ad_ratio - args.mci_ratio
    if cn_ratio < 0:
        print("Error: AD and MCI ratios sum to more than 1.0")
        return

    class_distribution = {
        'AD': args.ad_ratio,
        'MCI': args.mci_ratio,
        'CN': cn_ratio
    }

    print("="*70)
    print("SYNTHETIC BIOMARKER DATA GENERATOR")
    print("="*70)
    print(f"Random seed: {args.seed}")
    print(f"Output file: {args.output}")
    print()

    # Generate data
    df = generate_sample_data(args.n_samples, class_distribution)

    # Print statistics
    print_summary_statistics(df)

    # Save to Excel
    output_path = Path(args.output)
    df.to_excel(output_path, index=False, engine='openpyxl')

    print(f"\n✓ Data saved to: {output_path}")
    print(f"✓ File size: {output_path.stat().st_size / 1024:.1f} KB")

    print("\nSample data (first 5 rows):")
    print(df.head().to_string())

    print("\n" + "="*70)
    print("READY TO USE!")
    print("="*70)
    print(f"Run the classifier with:")
    print(f"  python elasticnet_biomarker_classifier.py --input {args.output}")
    print("="*70)


if __name__ == '__main__':
    main()
