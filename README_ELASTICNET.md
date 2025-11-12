# Elastic Net Logistic Regression Classifier for Alzheimer's Disease

A production-ready Python script for predicting Alzheimer's disease stage (AD, MCI, CN) from biomarker data using Elastic Net Logistic Regression with comprehensive cross-validation and explainability features.

## Features

- **Data Loading & Preprocessing**: Automatic handling of Excel files with missing value detection
- **Elastic Net Logistic Regression**: Multi-class classification with L1/L2 penalty balance
- **Hyperparameter Optimization**: 5-fold stratified cross-validation with GridSearchCV
- **Cross-Validation Visualization**: Heatmaps and line plots of CV accuracy
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1, confusion matrix, ROC curves
- **Feature Importance**: Coefficient-based importance analysis
- **SHAP Explainability**: Global and local model interpretability
- **Google Colab Compatible**: Seamless file upload integration
- **Result Persistence**: All outputs saved in organized directory structure

## Requirements

### Python Version
- Python 3.8 or higher

### Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap joblib openpyxl
```

## Dataset Format

Your Excel file should contain:

- **Target column**: `diagnosis` (values: AD, MCI, CN)
- **Feature columns**:
  - Abeta40
  - Abeta42
  - Abeta ratio
  - Ptau-
  - tau
  - DHA
  - Formic acid
  - lactoferrin
  - AD7C-NTP

### Example Dataset Structure

| diagnosis | Abeta40 | Abeta42 | Abeta ratio | Ptau- | tau | DHA | Formic acid | lactoferrin | AD7C-NTP |
|-----------|---------|---------|-------------|-------|-----|-----|-------------|-------------|----------|
| AD        | 125.4   | 45.2    | 0.36        | 65.3  | 320 | 1.2 | 0.45        | 23.1        | 15.6     |
| MCI       | 130.2   | 52.1    | 0.40        | 45.2  | 280 | 1.5 | 0.38        | 20.3        | 12.4     |
| CN        | 135.8   | 65.4    | 0.48        | 25.1  | 210 | 1.8 | 0.30        | 18.2        | 8.3      |

## Usage

### Google Colab (Recommended)

1. Upload the script to your Colab environment:

```python
# Upload the script file
from google.colab import files
files.upload()  # Select elasticnet_biomarker_classifier.py
```

2. Install dependencies:

```python
!pip install pandas numpy scikit-learn matplotlib seaborn shap joblib openpyxl
```

3. Run the script:

```python
!python elasticnet_biomarker_classifier.py
```

The script will prompt you to upload your Excel file.

### Local Environment

```bash
# Basic usage - script will prompt for file in Colab
python elasticnet_biomarker_classifier.py

# Specify input file
python elasticnet_biomarker_classifier.py --input your_data.xlsx

# Specify custom results directory
python elasticnet_biomarker_classifier.py --input your_data.xlsx --results-dir my_results

# Skip SHAP analysis (faster execution)
python elasticnet_biomarker_classifier.py --input your_data.xlsx --skip-shap
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input` | str | None | Path to input Excel file |
| `--results-dir` | str | results | Directory to save results |
| `--skip-shap` | flag | False | Skip SHAP analysis to save time |

## Model Configuration

### Elastic Net Parameters

- **Solver**: SAGA (supports Elastic Net penalty)
- **Multi-class**: Multinomial (softmax regression)
- **Penalty**: Elastic Net (L1 + L2 combination)
- **Max iterations**: 10,000
- **Random state**: 42 (for reproducibility)

### Hyperparameter Grid

The script performs grid search over:

- **C** (Inverse regularization strength): [0.01, 0.1, 1, 10]
- **l1_ratio** (L1 vs L2 balance): [0.1, 0.5, 0.9]
  - l1_ratio = 0: Pure L2 (Ridge)
  - l1_ratio = 1: Pure L1 (Lasso)
  - 0 < l1_ratio < 1: Elastic Net

### Cross-Validation

- **Strategy**: 5-fold Stratified K-Fold
- **Metric**: Accuracy
- **Parallelization**: Enabled (`n_jobs=-1`)

## Output Files

All results are saved in the `results/` directory (or custom directory specified):

### Model Files
- **`elasticnet_model.pkl`**: Trained model, scaler, and metadata

### Evaluation Reports
- **`classification_report.txt`**: Detailed per-class metrics
- **`feature_importance.csv`**: Feature importance scores

### Visualizations

#### Cross-Validation
- **`cv_heatmap_accuracy.png`**: CV accuracy across C vs l1_ratio
- **`cv_line_plot_accuracy.png`**: CV accuracy vs C for each l1_ratio

#### Model Evaluation
- **`confusion_matrix.png`**: Confusion matrix heatmap
- **`roc_curve.png`**: ROC curves (one-vs-rest) with AUC scores

#### Feature Analysis
- **`feature_importance_plot.png`**: Top 10 most important features

#### Explainability (SHAP)
- **`shap_summary_plot.png`**: Global feature impact (beeswarm plot)
- **`shap_summary_plot_[CLASS].png`**: Per-class SHAP summary
- **`shap_force_plot_[CLASS].png`**: Local explanation for sample prediction

## Sample Output

### Console Output

```
======================================================================
ELASTIC NET ALZHEIMER'S DISEASE CLASSIFIER
======================================================================

[1/7] Loading data...
[INFO] Loading data from your_data.xlsx...
[INFO] Data loaded successfully. Shape: (412, 10)

[2/7] Preprocessing...
[INFO] Preprocessing data...
[INFO] After dropping missing diagnosis: 408 samples
[INFO] Classes: ['AD' 'CN' 'MCI']
[INFO] Train set: 326 samples
[INFO] Test set: 82 samples

[3/7] Running cross-validation grid search...
[INFO] Setting up Elastic Net Logistic Regression with GridSearchCV...
[INFO] Running 5-fold stratified cross-validation grid search...
[INFO] Cross-validation completed!
[INFO] Best parameters: C=1.0, l1_ratio=0.5
[INFO] Best CV accuracy: 0.8313

[4/7] Evaluating on test set...
======================================================================
MODEL EVALUATION RESULTS
======================================================================
Model Accuracy: 0.8415
Best Params: C=1.0, l1_ratio=0.5
Macro Precision: 0.8356 | Macro Recall: 0.8298 | Macro F1: 0.8315

Per-Class Metrics:
  AD: Precision=0.8824, Recall=0.8824, F1=0.8824
  CN: Precision=0.8500, Recall=0.8095, F1=0.8293
  MCI: Precision=0.7744, Retention=0.7975, F1=0.7857
======================================================================

[5/7] Showing prediction examples...
======================================================================
PREDICTION EXAMPLES
======================================================================

Sample 45:
  AD: 0.7823 | CN: 0.1456 | MCI: 0.0721
  → Predicted: AD | True: AD
  ✓ Correct

Sample 12:
  AD: 0.1234 | CN: 0.7654 | MCI: 0.1112
  → Predicted: CN | True: CN
  ✓ Correct

[6/7] Generating explainability plots...
[7/7] Saving model and results...

======================================================================
PIPELINE COMPLETED SUCCESSFULLY!
======================================================================
✓ Model accuracy: 0.8415
✓ Best parameters: C=1.0, l1_ratio=0.5
✓ All results saved to: results/

Generated files:
  - classification_report.txt
  - confusion_matrix.png
  - cv_heatmap_accuracy.png
  - cv_line_plot_accuracy.png
  - elasticnet_model.pkl
  - feature_importance.csv
  - feature_importance_plot.png
  - roc_curve.png
  - shap_force_plot_AD.png
  - shap_force_plot_CN.png
  - shap_force_plot_MCI.png
  - shap_summary_plot.png
  - shap_summary_plot_AD.png
  - shap_summary_plot_CN.png
  - shap_summary_plot_MCI.png
======================================================================
```

## Model Reloading

To reload and use the trained model:

```python
import joblib
import numpy as np

# Load model
model_data = joblib.load('results/elasticnet_model.pkl')
model = model_data['model']
scaler = model_data['scaler']
label_encoder = model_data['label_encoder']
feature_names = model_data['feature_names']
class_names = model_data['class_names']

print(f"Best parameters: {model_data['best_params']}")
print(f"CV accuracy: {model_data['best_score']:.4f}")

# Make predictions on new data
# X_new should be a numpy array with shape (n_samples, 9)
X_new = np.array([[125.4, 45.2, 0.36, 65.3, 320, 1.2, 0.45, 23.1, 15.6]])
X_new_scaled = scaler.transform(X_new)

# Get predictions
predictions = model.predict(X_new_scaled)
probabilities = model.predict_proba(X_new_scaled)

# Convert to labels
predicted_labels = label_encoder.inverse_transform(predictions)

print(f"Predicted class: {predicted_labels[0]}")
print(f"Probabilities:")
for i, class_name in enumerate(class_names):
    print(f"  {class_name}: {probabilities[0][i]:.4f}")
```

## Understanding the Results

### Cross-Validation Heatmap
- **Darker colors** = Higher accuracy
- **Best combination** is marked with highest value
- Shows how different C and l1_ratio values affect model performance

### Confusion Matrix
- **Diagonal elements** = Correct predictions
- **Off-diagonal** = Misclassifications
- Helps identify which classes are confused

### ROC Curves
- **Area Under Curve (AUC)** closer to 1.0 = Better performance
- **Micro-average** = Overall performance across all classes
- **Macro-average** = Unweighted mean of per-class performance

### Feature Importance
- Based on **mean absolute coefficients** across all classes
- Higher values = More influential features
- Helps understand which biomarkers drive predictions

### SHAP Plots
- **Summary plot**: Shows global feature impact (all samples)
  - Red = High feature value
  - Blue = Low feature value
  - Position on x-axis = Impact on prediction
- **Force plot**: Shows how features contribute to a single prediction
  - Red = Pushes prediction higher
  - Blue = Pushes prediction lower

## Advantages of Elastic Net

1. **Feature Selection**: L1 penalty drives some coefficients to zero
2. **Handles Multicollinearity**: L2 penalty handles correlated features
3. **Interpretability**: Coefficient-based importance is straightforward
4. **Stability**: Elastic Net is more stable than pure Lasso
5. **Efficiency**: Faster training than Random Forest for this dataset size

## Comparison with Random Forest

| Aspect | Elastic Net | Random Forest |
|--------|-------------|---------------|
| **Interpretability** | ✅ High (linear coefficients) | ⚠️ Medium (tree-based) |
| **Training Speed** | ✅ Fast | ⚠️ Slower |
| **Feature Selection** | ✅ Built-in (L1 penalty) | ❌ Requires extra steps |
| **Handling Non-linearity** | ⚠️ Limited | ✅ Excellent |
| **Hyperparameter Tuning** | ✅ Simple (2 params) | ⚠️ Complex (many params) |
| **Probability Calibration** | ✅ Well-calibrated | ⚠️ May need calibration |

## Troubleshooting

### Issue: "ConvergenceWarning: The max_iter was reached"
**Solution**: The model is already set to `max_iter=10000`. If this warning appears, the SAGA solver is still iterating. The results are usually still valid, but you can increase `max_iter` further if needed.

### Issue: SHAP analysis taking too long
**Solution**: Use the `--skip-shap` flag to skip SHAP analysis, or reduce the sample sizes in the `explain_with_shap()` method.

### Issue: "KeyError: 'diagnosis'"
**Solution**: Ensure your Excel file has a column named exactly `diagnosis` (case-sensitive).

### Issue: Missing feature columns
**Solution**: Verify all 9 feature columns exist in your Excel file with exact names (case-sensitive).

### Issue: Low accuracy
**Possible causes**:
- Insufficient data quality or quantity
- Imbalanced classes (consider SMOTE or class weights)
- Need feature engineering
- Try different hyperparameter ranges

## Performance Tips

1. **Speed up SHAP**: Reduce `background_size` and `test_sample_size` in `explain_with_shap()`
2. **Faster grid search**: Reduce the hyperparameter grid or use `n_jobs=-1` (already enabled)
3. **Memory optimization**: Process data in smaller batches if memory is limited
4. **Skip plots**: Comment out plotting functions if you only need the model

## Customization

### Adding More Hyperparameters

Edit the `param_grid` in the `train_model()` method:

```python
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Expand range
    'l1_ratio': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],  # Add more values
    'class_weight': [None, 'balanced']  # Handle class imbalance
}
```

### Changing Train/Test Split

Modify the `test_size` parameter in `preprocess_data()`:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.3,  # Change to 70/30 split
    stratify=y_encoded,
    random_state=RANDOM_STATE
)
```

### Adding More Features

Update the `FEATURE_COLUMNS` list at the top of the script:

```python
FEATURE_COLUMNS = [
    'Abeta40', 'Abeta42', 'Abeta ratio', 'Ptau-', 'tau',
    'DHA', 'Formic acid', 'lactoferrin', 'AD7C-NTP',
    'new_biomarker_1', 'new_biomarker_2'  # Add your features
]
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{elasticnet_alzheimer_classifier,
  title={Elastic Net Logistic Regression Classifier for Alzheimer's Disease},
  author={Claude Code},
  year={2025},
  url={https://github.com/yourusername/ML_05_Alzhemiers}
}
```

## License

This project is provided as-is for educational and research purposes.

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the console output for error messages
3. Verify your data format matches the requirements
4. Check that all dependencies are installed correctly

## Acknowledgments

- scikit-learn for machine learning algorithms
- SHAP for model explainability
- The Alzheimer's research community for biomarker research

---

**Version**: 1.0.0
**Last Updated**: 2025-11-12
**Python**: 3.8+
**Status**: Production Ready ✅
