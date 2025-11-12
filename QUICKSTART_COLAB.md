# Quick Start Guide for Google Colab

This guide will help you run the Elastic Net Alzheimer's classifier in Google Colab in under 5 minutes.

## Step-by-Step Instructions

### Step 1: Open Google Colab

Go to [Google Colab](https://colab.research.google.com/) and create a new notebook.

### Step 2: Upload the Script

Run this cell to upload the Python script:

```python
# Upload elasticnet_biomarker_classifier.py
from google.colab import files
uploaded = files.upload()
```

Select `elasticnet_biomarker_classifier.py` from your computer.

### Step 3: Install Dependencies

```python
!pip install pandas numpy scikit-learn matplotlib seaborn shap joblib openpyxl
```

Wait for installation to complete (~1-2 minutes).

### Step 4: Run the Classifier

```python
!python elasticnet_biomarker_classifier.py
```

When prompted, upload your Excel file with biomarker data.

### Step 5: View Results

```python
# List all generated files
!ls -lh results/
```

```python
# View classification report
!cat results/classification_report.txt
```

```python
# Display confusion matrix
from IPython.display import Image
Image('results/confusion_matrix.png')
```

```python
# Display CV heatmap
Image('results/cv_heatmap_accuracy.png')
```

```python
# Display ROC curves
Image('results/roc_curve.png')
```

```python
# Display feature importance
Image('results/feature_importance_plot.png')
```

```python
# Display SHAP summary
Image('results/shap_summary_plot.png')
```

### Step 6: Download Results

```python
# Download all results as a zip file
!zip -r results.zip results/
files.download('results.zip')
```

## Complete Colab Notebook Template

Copy and paste these cells into a new Google Colab notebook:

---

### Cell 1: Install Dependencies

```python
# Install required packages
!pip install -q pandas numpy scikit-learn matplotlib seaborn shap joblib openpyxl
print("✓ Dependencies installed successfully!")
```

---

### Cell 2: Upload Script and Data

```python
# Upload the Python script and your data file
from google.colab import files
import os

print("Step 1: Upload elasticnet_biomarker_classifier.py")
uploaded_script = files.upload()

print("\n✓ Script uploaded!")
print("\nFiles in current directory:")
!ls -lh
```

---

### Cell 3: Run the Classifier

```python
# Run the classifier
!python elasticnet_biomarker_classifier.py
```

*Note: When prompted, upload your Excel file with biomarker data*

---

### Cell 4: View Summary Results

```python
# Display the classification report
print("="*70)
print("CLASSIFICATION REPORT")
print("="*70)
!cat results/classification_report.txt

print("\n" + "="*70)
print("GENERATED FILES")
print("="*70)
!ls -lh results/
```

---

### Cell 5: Visualize Confusion Matrix

```python
from IPython.display import Image, display

display(Image('results/confusion_matrix.png'))
```

---

### Cell 6: Visualize Cross-Validation Heatmap

```python
display(Image('results/cv_heatmap_accuracy.png'))
```

---

### Cell 7: Visualize ROC Curves

```python
display(Image('results/roc_curve.png'))
```

---

### Cell 8: Visualize Feature Importance

```python
display(Image('results/feature_importance_plot.png'))
```

---

### Cell 9: Visualize SHAP Summary

```python
display(Image('results/shap_summary_plot.png'))
```

---

### Cell 10: View Feature Importance Data

```python
import pandas as pd

# Load and display feature importance
feat_imp = pd.read_csv('results/feature_importance.csv')
print("Feature Importance Rankings:")
print(feat_imp.to_string(index=False))
```

---

### Cell 11: Load and Test the Model

```python
import joblib
import numpy as np

# Load the trained model
model_data = joblib.load('results/elasticnet_model.pkl')
model = model_data['model']
scaler = model_data['scaler']
label_encoder = model_data['label_encoder']
feature_names = model_data['feature_names']
class_names = model_data['class_names']

print("✓ Model loaded successfully!")
print(f"Best parameters: {model_data['best_params']}")
print(f"Best CV score: {model_data['best_score']:.4f}")
print(f"Features: {', '.join(feature_names)}")
print(f"Classes: {', '.join(class_names)}")
```

---

### Cell 12: Make Predictions on New Data

```python
# Example: Make a prediction on new data
# Replace these values with your actual biomarker measurements

new_sample = np.array([[
    125.4,  # Abeta40
    45.2,   # Abeta42
    0.36,   # Abeta ratio
    65.3,   # Ptau-
    320,    # tau
    1.2,    # DHA
    0.45,   # Formic acid
    23.1,   # lactoferrin
    15.6    # AD7C-NTP
]])

# Scale the data
new_sample_scaled = scaler.transform(new_sample)

# Make prediction
prediction = model.predict(new_sample_scaled)
probabilities = model.predict_proba(new_sample_scaled)

# Display results
predicted_label = label_encoder.inverse_transform(prediction)[0]

print("="*70)
print("PREDICTION FOR NEW SAMPLE")
print("="*70)
print("\nInput features:")
for i, feat in enumerate(feature_names):
    print(f"  {feat}: {new_sample[0][i]}")

print("\nPrediction probabilities:")
for i, class_name in enumerate(class_names):
    print(f"  {class_name}: {probabilities[0][i]:.4f} ({probabilities[0][i]*100:.2f}%)")

print(f"\n✓ Predicted diagnosis: {predicted_label}")
print("="*70)
```

---

### Cell 13: Download All Results

```python
# Zip and download all results
!zip -r results.zip results/
files.download('results.zip')
print("✓ Results downloaded!")
```

---

## Advanced Options

### Run with Custom Parameters

```python
# Run with custom results directory
!python elasticnet_biomarker_classifier.py --input your_data.xlsx --results-dir my_custom_results

# Run without SHAP analysis (faster)
!python elasticnet_biomarker_classifier.py --input your_data.xlsx --skip-shap
```

### Batch Processing Multiple Files

```python
import os

# Upload multiple Excel files
uploaded_files = files.upload()

# Process each file
for filename in uploaded_files.keys():
    if filename.endswith('.xlsx'):
        print(f"\nProcessing {filename}...")
        results_dir = f"results_{filename.replace('.xlsx', '')}"
        !python elasticnet_biomarker_classifier.py --input {filename} --results-dir {results_dir}
        print(f"✓ Results saved to {results_dir}/")
```

## Troubleshooting

### Issue: "No module named 'pandas'"
**Solution**: Run the dependency installation cell again.

### Issue: "File not found"
**Solution**: Make sure you uploaded both the script and your data file.

### Issue: "KeyError: 'diagnosis'"
**Solution**: Check that your Excel file has a column named `diagnosis` (case-sensitive).

### Issue: Script runs but no plots appear
**Solution**: Use `display(Image('path/to/image.png'))` to view images in Colab.

## Tips for Best Results

1. **Data Quality**: Ensure your Excel file has no formatting issues
2. **Column Names**: Match the exact feature names (case-sensitive)
3. **Missing Values**: The script handles missing diagnoses, but features should be complete
4. **Sample Size**: Aim for at least 100+ samples for reliable results
5. **Class Balance**: Check that you have reasonable samples of each class (AD, MCI, CN)

## Expected Runtime

On Google Colab with default settings:

- Data loading: ~5 seconds
- Preprocessing: ~2 seconds
- Grid search CV: ~30-60 seconds
- Evaluation: ~5 seconds
- Feature importance: ~3 seconds
- SHAP analysis: ~2-5 minutes
- **Total**: ~4-7 minutes for 400 samples

## Next Steps

After running the classifier:

1. Review the classification report for overall performance
2. Examine the confusion matrix to understand misclassifications
3. Check feature importance to identify key biomarkers
4. Use SHAP plots to understand model decisions
5. Save and reload the model for future predictions

## Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Elastic Net Explanation](https://scikit-learn.org/stable/modules/linear_model.html#elastic-net)

---

**Happy Classifying! 🧠🔬**
