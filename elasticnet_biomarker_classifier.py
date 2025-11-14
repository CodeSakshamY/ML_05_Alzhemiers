#!/usr/bin/env python3
"""
Elastic Net Logistic Regression Classifier for Alzheimer's Disease Biomarker Prediction
========================================================================================

This script predicts Alzheimer's disease stage (AD, MCI, CN) from biomarker data using
Elastic Net Logistic Regression with comprehensive cross-validation and explainability.

Features:
- Data loading and preprocessing
- Elastic Net Logistic Regression with GridSearchCV
- Cross-validation metric plots (heatmap)
- Comprehensive test set evaluation
- Feature importance and SHAP explainability
- Production-ready with Google Colab integration

Author: Claude Code
Date: 2025-11-12
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import joblib
import shap

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Define feature columns
FEATURE_COLUMNS = [
    'Abeta40', 'Abeta42', 'Abeta ratio', 'Ptau-', 'tau',
    'DHA', 'Formic acid', 'lactoferrin', 'AD7C-NTP'
]

TARGET_COLUMN = 'diagnosis'


class AlzheimerClassifier:
    """
    Elastic Net Logistic Regression classifier for Alzheimer's disease prediction.
    """

    def __init__(self, results_dir: str = 'results'):
        """
        Initialize the classifier.

        Args:
            results_dir: Directory to save results and models
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.grid_search = None
        self.feature_names = FEATURE_COLUMNS
        self.class_names = None

        # For storing results
        self.cv_results = None
        self.test_results = {}

    def log(self, message: str):
        """Print formatted log message."""
        print(f"[INFO] {message}")

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from Excel file.

        Args:
            file_path: Path to the Excel file

        Returns:
            Loaded DataFrame
        """
        self.log(f"Loading data from {file_path}...")

        try:
            df = pd.read_excel(file_path, engine='openpyxl')
            self.log(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            self.log(f"Error loading data: {str(e)}")
            raise

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess the data: drop missing diagnosis, encode labels, split and scale.

        Args:
            df: Input DataFrame

        Returns:
            X_train, X_test, y_train, y_test (scaled features and encoded labels)
        """
        self.log("Preprocessing data...")

        # Drop rows with missing diagnosis
        df_clean = df.dropna(subset=[TARGET_COLUMN])
        self.log(f"After dropping missing diagnosis: {df_clean.shape[0]} samples")

        # Extract features and target
        X = df_clean[self.feature_names].values
        y = df_clean[TARGET_COLUMN].values

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_names = self.label_encoder.classes_
        self.log(f"Classes: {self.class_names}")

        # Stratified train-test split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=0.2,
            stratify=y_encoded,
            random_state=RANDOM_STATE
        )

        self.log(f"Train set: {X_train.shape[0]} samples")
        self.log(f"Test set: {X_test.shape[0]} samples")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.log("Data preprocessing completed")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train Logistic Regression with L1, L2, and Elastic Net regularization using GridSearchCV.

        Args:
            X_train: Training features
            y_train: Training labels
        """
        self.log("Setting up Logistic Regression with L1, L2, and Elastic Net regularization...")

        # Define base model
        base_model = LogisticRegression(
            multi_class='multinomial',
            solver='saga',
            max_iter=10000,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        # Define hyperparameter grid for different regularization types
        # GridSearchCV supports a list of dictionaries for different parameter spaces
        param_grid = [
            # L1 Regularization (Lasso)
            {
                'penalty': ['l1'],
                'C': [0.001, 0.01, 0.1, 1, 10, 100]
            },
            # L2 Regularization (Ridge)
            {
                'penalty': ['l2'],
                'C': [0.001, 0.01, 0.1, 1, 10, 100]
            },
            # Elastic Net Regularization (L1 + L2 combination)
            {
                'penalty': ['elasticnet'],
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
        ]

        # Setup GridSearchCV with stratified K-fold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        self.grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )

        self.log("Running 5-fold stratified cross-validation grid search...")
        self.log("Testing L1 (Lasso), L2 (Ridge), and Elastic Net regularization")
        self.log(f"L1/L2 regularization strength (C): {param_grid[0]['C']}")
        self.log(f"Elastic Net l1_ratio: {param_grid[2]['l1_ratio']}")
        self.log("This may take a few minutes...")

        # Fit grid search
        self.grid_search.fit(X_train, y_train)

        # Store best model
        self.model = self.grid_search.best_estimator_

        # Store CV results
        self.cv_results = pd.DataFrame(self.grid_search.cv_results_)

        self.log(f"Cross-validation completed!")
        self.log(f"Best regularization type: {self.grid_search.best_params_['penalty']}")
        self.log(f"Best C (inverse regularization strength): {self.grid_search.best_params_['C']}")
        if 'l1_ratio' in self.grid_search.best_params_:
            self.log(f"Best l1_ratio: {self.grid_search.best_params_['l1_ratio']}")
        self.log(f"Best CV accuracy: {self.grid_search.best_score_:.4f}")

    def plot_cv_heatmap(self):
        """
        Generate heatmap of cross-validation accuracy across C vs l1_ratio.
        """
        self.log("Generating cross-validation heatmap...")

        # Prepare data for heatmap
        pivot_data = self.cv_results.pivot_table(
            values='mean_test_score',
            index='param_l1_ratio',
            columns='param_C'
        )

        # Create heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.4f',
            cmap='YlGnBu',
            cbar_kws={'label': 'Mean CV Accuracy'},
            linewidths=0.5
        )
        plt.title('Cross-Validation Accuracy Heatmap\n(Elastic Net Logistic Regression)',
                  fontsize=14, fontweight='bold')
        plt.xlabel('C (Inverse Regularization Strength)', fontsize=12)
        plt.ylabel('l1_ratio (L1 vs L2 Balance)', fontsize=12)
        plt.tight_layout()

        # Save
        save_path = self.results_dir / 'cv_heatmap_accuracy.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.log(f"Heatmap saved to {save_path}")

        # Optional: Line plots of accuracy vs C for each l1_ratio
        self._plot_cv_line_plots()

    def _plot_cv_line_plots(self):
        """Generate line plots of accuracy vs C for each l1_ratio."""
        plt.figure(figsize=(10, 6))

        for l1_ratio in self.cv_results['param_l1_ratio'].unique():
            subset = self.cv_results[self.cv_results['param_l1_ratio'] == l1_ratio]
            plt.plot(
                subset['param_C'],
                subset['mean_test_score'],
                marker='o',
                label=f'l1_ratio={l1_ratio}',
                linewidth=2
            )

        plt.xscale('log')
        plt.xlabel('C (Inverse Regularization Strength)', fontsize=12)
        plt.ylabel('Mean CV Accuracy', fontsize=12)
        plt.title('Cross-Validation Accuracy vs Regularization Parameter C',
                  fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = self.results_dir / 'cv_line_plot_accuracy.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.log(f"Line plot saved to {save_path}")

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Evaluate model on test set with comprehensive metrics.

        Args:
            X_test: Test features
            y_test: Test labels
        """
        self.log("Evaluating model on test set...")

        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        # Calculate metrics - overall
        accuracy = accuracy_score(y_test, y_pred)

        # Macro averages
        precision_macro = precision_score(y_test, y_pred, average='macro')
        recall_macro = recall_score(y_test, y_pred, average='macro')
        f1_macro = f1_score(y_test, y_pred, average='macro')

        # Weighted averages
        precision_weighted = precision_score(y_test, y_pred, average='weighted')
        recall_weighted = recall_score(y_test, y_pred, average='weighted')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')

        # Per-class metrics
        precision_per_class = precision_score(y_test, y_pred, average=None)
        recall_per_class = recall_score(y_test, y_pred, average=None)
        f1_per_class = f1_score(y_test, y_pred, average=None)

        # Store results
        self.test_results = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

        # Generate visualizations and get ROC-AUC metrics
        cm = self._plot_confusion_matrix(y_test, y_pred)
        roc_auc_dict = self._plot_roc_curves(X_test, y_test)

        # Store ROC-AUC results
        self.test_results['roc_auc'] = roc_auc_dict
        self.test_results['confusion_matrix'] = cm

        # Print and save comprehensive performance metrics
        self._print_and_save_performance_metrics()

        # Generate detailed classification report
        self._save_classification_report(y_test, y_pred)

    def _print_and_save_performance_metrics(self):
        """Print and save comprehensive performance metrics."""
        # Get cross-validation stats
        cv_scores = self.cv_results['mean_test_score']
        cv_mean = cv_scores.max()  # Best CV score
        cv_std = self.cv_results.loc[self.cv_results['mean_test_score'].idxmax(), 'std_test_score']

        # Prepare output text
        output = []
        output.append("\n" + "="*70)
        output.append("### Performance Metrics")
        output.append("="*70)

        # Overall metrics
        output.append(f"\nAccuracy: {self.test_results['accuracy']:.2f}")
        output.append(f"Macro Avg → Precision: {self.test_results['precision_macro']:.2f} | "
                     f"Recall: {self.test_results['recall_macro']:.2f} | "
                     f"F1: {self.test_results['f1_macro']:.2f}")
        output.append(f"Weighted Avg → Precision: {self.test_results['precision_weighted']:.2f} | "
                     f"Recall: {self.test_results['recall_weighted']:.2f} | "
                     f"F1: {self.test_results['f1_weighted']:.2f}")

        # Per-class metrics
        output.append("\n--- Per-Class ---")
        for i, class_name in enumerate(self.class_names):
            output.append(f"{class_name}: Precision {self.test_results['precision_per_class'][i]:.2f} | "
                         f"Recall {self.test_results['recall_per_class'][i]:.2f} | "
                         f"F1 {self.test_results['f1_per_class'][i]:.2f}")

        # Confusion matrix
        output.append("\n--- Confusion Matrix ---")
        cm = self.test_results['confusion_matrix']
        output.append(f"{'':>6} " + " ".join(f"{cls:>6}" for cls in self.class_names))
        for i, class_name in enumerate(self.class_names):
            output.append(f"{class_name:>6} " + " ".join(f"{cm[i][j]:>6}" for j in range(len(self.class_names))))

        # ROC-AUC metrics
        output.append("\n--- ROC-AUC Metrics ---")
        roc_auc = self.test_results['roc_auc']
        for i, class_name in enumerate(self.class_names):
            output.append(f"{class_name} AUC: {roc_auc[i]:.2f}")
        output.append(f"Micro-average AUC: {roc_auc['micro']:.2f}")
        output.append(f"Macro-average AUC: {roc_auc['macro']:.2f}")
        output.append(f"Weighted-average AUC: {roc_auc['weighted']:.2f}")

        # Cross-validation summary
        output.append("\n--- Cross-Validation Summary ---")
        output.append(f"Mean CV Accuracy: {cv_mean:.4f} (±{cv_std:.4f})")
        output.append(f"Best Parameters: C={self.grid_search.best_params_['C']}, "
                     f"l1_ratio={self.grid_search.best_params_['l1_ratio']}")

        output.append("="*70 + "\n")

        # Print to console
        for line in output:
            print(line)

        # Save to file
        metrics_path = self.results_dir / 'performance_metrics.txt'
        with open(metrics_path, 'w') as f:
            f.write('\n'.join(output))

        self.log(f"Performance metrics saved to {metrics_path}")

    def _save_classification_report(self, y_test: np.ndarray, y_pred: np.ndarray):
        """Save detailed classification report."""
        report = classification_report(
            y_test, y_pred,
            target_names=self.class_names,
            digits=4
        )

        report_path = self.results_dir / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write("CLASSIFICATION REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Model: Elastic Net Logistic Regression\n")
            f.write(f"Best Parameters: C={self.grid_search.best_params_['C']}, "
                    f"l1_ratio={self.grid_search.best_params_['l1_ratio']}\n")
            f.write(f"Test Accuracy: {self.test_results['accuracy']:.4f}\n\n")
            f.write(report)

        self.log(f"Classification report saved to {report_path}")

        # Also print to console
        print("\nFull Classification Report:")
        print(report)

    def _plot_confusion_matrix(self, y_test: np.ndarray, y_pred: np.ndarray):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix\n(Elastic Net Logistic Regression)',
                  fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        save_path = self.results_dir / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.log(f"Confusion matrix saved to {save_path}")

        return cm

    def _plot_roc_curves(self, X_test: np.ndarray, y_test: np.ndarray):
        """Plot ROC curves (one-vs-rest) with micro, macro, and weighted averages."""
        # Binarize labels for multi-class ROC
        y_test_bin = label_binarize(y_test, classes=range(len(self.class_names)))
        n_classes = len(self.class_names)

        # Get prediction probabilities
        y_score = self.model.predict_proba(X_test)

        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and AUC
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and AUC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Compute weighted-average AUC
        roc_auc["weighted"] = roc_auc_score(y_test_bin, y_score, average='weighted', multi_class='ovr')

        # Plot ROC curves
        plt.figure(figsize=(12, 8))

        # Plot micro and macro averages
        plt.plot(
            fpr["micro"], tpr["micro"],
            label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
            color='deeppink', linestyle=':', linewidth=3
        )
        plt.plot(
            fpr["macro"], tpr["macro"],
            label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})',
            color='navy', linestyle=':', linewidth=3
        )

        # Plot per-class ROC curves
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, color in zip(range(n_classes), colors[:n_classes]):
            plt.plot(
                fpr[i], tpr[i],
                color=color,
                lw=2,
                label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.3f})'
            )

        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - One-vs-Rest\n(Elastic Net Logistic Regression)',
                  fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = self.results_dir / 'roc_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.log(f"ROC curves saved to {save_path}")

        return roc_auc

    def show_predictions(self, X_test: np.ndarray, y_test: np.ndarray, n_samples: int = 5):
        """
        Show random prediction examples.

        Args:
            X_test: Test features
            y_test: Test labels
            n_samples: Number of samples to show
        """
        self.log(f"Showing {n_samples} random prediction examples...")

        # Get predictions
        y_pred_proba = self.model.predict_proba(X_test)
        y_pred = self.model.predict(X_test)

        # Select random samples
        indices = np.random.choice(len(X_test), size=min(n_samples, len(X_test)), replace=False)

        print("\n" + "="*70)
        print("PREDICTION EXAMPLES")
        print("="*70)

        for idx in indices:
            true_label = self.class_names[y_test[idx]]
            pred_label = self.class_names[y_pred[idx]]
            probas = y_pred_proba[idx]

            print(f"\nSample {idx}:")
            for i, class_name in enumerate(self.class_names):
                print(f"  {class_name}: {probas[i]:.4f}", end="")
                if i < len(self.class_names) - 1:
                    print(" |", end="")
                else:
                    print()
            print(f"  → Predicted: {pred_label} | True: {true_label}")
            if pred_label == true_label:
                print("  ✓ Correct")
            else:
                print("  ✗ Incorrect")

        print("="*70 + "\n")

    def analyze_feature_importance(self):
        """
        Analyze and plot feature importance using model coefficients.
        """
        self.log("Analyzing feature importance...")

        # Get model coefficients (shape: n_classes x n_features)
        coefficients = self.model.coef_

        # Compute mean absolute coefficient across classes
        feature_importance = np.mean(np.abs(coefficients), axis=0)

        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)

        # Save to CSV
        csv_path = self.results_dir / 'feature_importance.csv'
        importance_df.to_csv(csv_path, index=False)
        self.log(f"Feature importance saved to {csv_path}")

        # Print top features
        print("\n" + "="*70)
        print("FEATURE IMPORTANCE (Mean Absolute Coefficients)")
        print("="*70)
        print(importance_df.to_string(index=False))
        print("="*70 + "\n")

        # Plot feature importance
        self._plot_feature_importance(importance_df)

    def _plot_feature_importance(self, importance_df: pd.DataFrame):
        """Plot feature importance bar chart."""
        # Take top 10 features
        top_features = importance_df.head(10)

        plt.figure(figsize=(12, 6))
        bars = plt.barh(
            range(len(top_features)),
            top_features['Importance'],
            color='steelblue'
        )

        # Color the bars with gradient
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(top_features)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Mean Absolute Coefficient', fontsize=12)
        plt.title('Top 10 Most Important Features\n(Elastic Net Logistic Regression)',
                  fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        save_path = self.results_dir / 'feature_importance_plot.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.log(f"Feature importance plot saved to {save_path}")

    def explain_with_shap(self, X_train: np.ndarray, X_test: np.ndarray):
        """
        Generate SHAP explanations for model predictions.

        Args:
            X_train: Training features (for background)
            X_test: Test features (for explanation)
        """
        self.log("Generating SHAP explanations (this may take a few minutes)...")

        # Use a sample of training data as background (for efficiency)
        background_size = min(100, len(X_train))
        background = X_train[np.random.choice(X_train.shape[0], background_size, replace=False)]

        # Create SHAP explainer
        explainer = shap.KernelExplainer(self.model.predict_proba, background)

        # Calculate SHAP values for a sample of test data
        test_sample_size = min(50, len(X_test))
        test_sample_indices = np.random.choice(X_test.shape[0], test_sample_size, replace=False)
        X_test_sample = X_test[test_sample_indices]

        self.log(f"Computing SHAP values for {test_sample_size} test samples...")
        shap_values = explainer.shap_values(X_test_sample)

        # Global explanation: Summary plot (beeswarm)
        self._plot_shap_summary(shap_values, X_test_sample)

        # Local explanation: Force plot for first sample
        self._plot_shap_force(explainer, shap_values, X_test_sample)

    def _plot_shap_summary(self, shap_values, X_test_sample: np.ndarray):
        """Plot SHAP summary plot (beeswarm)."""
        # For multi-class, we'll plot for each class
        for i, class_name in enumerate(self.class_names):
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values[i] if isinstance(shap_values, list) else shap_values[:, :, i],
                X_test_sample,
                feature_names=self.feature_names,
                show=False,
                plot_size=(12, 8)
            )
            plt.title(f'SHAP Summary Plot - {class_name}\n(Feature Impact on Model Output)',
                      fontsize=14, fontweight='bold')
            plt.tight_layout()

            save_path = self.results_dir / f'shap_summary_plot_{class_name}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.log(f"SHAP summary plot for {class_name} saved to {save_path}")

        # Combined summary plot
        plt.figure(figsize=(12, 8))
        if isinstance(shap_values, list):
            # For multi-class, use the first class
            shap.summary_plot(
                shap_values[0],
                X_test_sample,
                feature_names=self.feature_names,
                show=False,
                plot_size=(12, 8)
            )
        else:
            shap.summary_plot(
                shap_values[:, :, 0],
                X_test_sample,
                feature_names=self.feature_names,
                show=False,
                plot_size=(12, 8)
            )
        plt.title('SHAP Summary Plot - Global Feature Impact\n(All Classes)',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = self.results_dir / 'shap_summary_plot.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.log(f"Combined SHAP summary plot saved to {save_path}")

    def _plot_shap_force(self, explainer, shap_values, X_test_sample: np.ndarray):
        """Generate SHAP force plot for a single prediction."""
        # Force plots work best with single samples
        sample_idx = 0

        for i, class_name in enumerate(self.class_names):
            plt.figure(figsize=(20, 3))

            if isinstance(shap_values, list):
                force_plot = shap.force_plot(
                    explainer.expected_value[i],
                    shap_values[i][sample_idx, :],
                    X_test_sample[sample_idx, :],
                    feature_names=self.feature_names,
                    matplotlib=True,
                    show=False
                )
            else:
                force_plot = shap.force_plot(
                    explainer.expected_value[i],
                    shap_values[sample_idx, :, i],
                    X_test_sample[sample_idx, :],
                    feature_names=self.feature_names,
                    matplotlib=True,
                    show=False
                )

            save_path = self.results_dir / f'shap_force_plot_{class_name}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.log(f"SHAP force plot for {class_name} saved to {save_path}")

    def save_model(self):
        """Save trained model and preprocessing objects."""
        self.log("Saving model and preprocessing objects...")

        # Save model
        model_path = self.results_dir / 'elasticnet_model.pkl'
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'best_params': self.grid_search.best_params_,
            'best_score': self.grid_search.best_score_
        }, model_path)

        self.log(f"Model saved to {model_path}")

        # Print reload example
        print("\n" + "="*70)
        print("MODEL PERSISTENCE")
        print("="*70)
        print(f"Model saved to: {model_path}")
        print("\nTo reload the model, use:")
        print("-" * 70)
        print("import joblib")
        print(f"model_data = joblib.load('{model_path}')")
        print("model = model_data['model']")
        print("scaler = model_data['scaler']")
        print("label_encoder = model_data['label_encoder']")
        print("\n# Make predictions on new data:")
        print("X_new_scaled = scaler.transform(X_new)")
        print("predictions = model.predict(X_new_scaled)")
        print("predicted_labels = label_encoder.inverse_transform(predictions)")
        print("="*70 + "\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Elastic Net Logistic Regression for Alzheimer\'s Disease Classification'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Path to input Excel file',
        default=None
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory to save results (default: results)'
    )
    parser.add_argument(
        '--skip-shap',
        action='store_true',
        help='Skip SHAP analysis (saves time)'
    )

    args = parser.parse_args()

    # Check if running in Colab
    try:
        import google.colab
        in_colab = True
    except ImportError:
        in_colab = False

    # Handle file upload in Colab
    if in_colab and args.input is None:
        print("Running in Google Colab. Please upload your Excel file.")
        from google.colab import files
        uploaded = files.upload()

        if not uploaded:
            print("No file uploaded. Exiting.")
            sys.exit(1)

        input_file = list(uploaded.keys())[0]
    elif args.input:
        input_file = args.input
    else:
        print("Error: Please provide an input file using --input or run in Google Colab")
        sys.exit(1)

    # Verify file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    # Start pipeline
    print("\n" + "="*70)
    print("ELASTIC NET ALZHEIMER'S DISEASE CLASSIFIER")
    print("="*70 + "\n")

    # Initialize classifier
    classifier = AlzheimerClassifier(results_dir=args.results_dir)

    # [1/7] Load data
    print("[1/7] Loading data...")
    df = classifier.load_data(input_file)

    # [2/7] Preprocess
    print("\n[2/7] Preprocessing...")
    X_train, X_test, y_train, y_test = classifier.preprocess_data(df)

    # [3/7] Cross-validation grid search
    print("\n[3/7] Running cross-validation grid search...")
    classifier.train_model(X_train, y_train)

    # [3.5/7] Generate CV plots
    print("\n[3.5/7] Generating cross-validation plots...")
    classifier.plot_cv_heatmap()

    # [4/7] Evaluate on test set
    print("\n[4/7] Evaluating on test set...")
    classifier.evaluate_model(X_test, y_test)

    # [5/7] Show predictions
    print("\n[5/7] Showing prediction examples...")
    classifier.show_predictions(X_test, y_test, n_samples=5)

    # [6/7] Feature importance and explainability
    print("\n[6/7] Generating explainability plots...")
    classifier.analyze_feature_importance()

    if not args.skip_shap:
        classifier.explain_with_shap(X_train, X_test)
    else:
        print("Skipping SHAP analysis (--skip-shap flag set)")

    # [7/7] Save model
    print("\n[7/7] Saving model and results...")
    classifier.save_model()

    # Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"✓ Model accuracy: {classifier.test_results['accuracy']:.4f}")
    print(f"✓ Best parameters: C={classifier.grid_search.best_params_['C']}, "
          f"l1_ratio={classifier.grid_search.best_params_['l1_ratio']}")
    print(f"✓ All results saved to: {classifier.results_dir}/")
    print("\nGenerated files:")
    for file in sorted(classifier.results_dir.glob('*')):
        print(f"  - {file.name}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
