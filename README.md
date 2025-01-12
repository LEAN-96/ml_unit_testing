# Machine Learning Unit Testing Project

## Project Overview
This project demonstrates unit testing for machine learning models, focusing on a logistic regression classifier. It includes automated tests for model performance and runtime efficiency, with comprehensive logging and metric tracking.

## Project Structure

```
ml_unit_testing/
├── data/                # Data directory
│   └── Advertising.csv  # Dataset
│   └── test_data        # Test dataset
├──notebooks/
│   └──ML_Unit_Testing.ipynb # Notebook for Colab or Mybinder
├── src/                 # Source code
│   ├── __init__.py
│   ├── decorators.py    # Timer and logger decorators
│   └── model.py         # Custom Logistic Regression model
├── tests/               # Test directory
│   ├── __init__.py
│   └── test_model.py    # Unit tests implementation
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

## Requirements
- Python 3.10
- pandas
- numpy
- scikit-learn
- pytest
- logging

## Installation & Setup

### Local Installation
```bash
# Clone the repository
git clone https://github.com/LEAN-96/ml_unit_testing.git
cd ml_unit_testing

# Create and activate virtual environment
python -m venv ml_unit_testing
source ml_unit_testing/bin/activate  # On Windows: ml_unit_testing\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run all tests with verbose output
python -m unittest -v tests/test_model.py

# Run specific test case
python -m unittest tests.test_model.TestCustomLogisticRegression.test_predict_accuracy_and_confusion_matrix
```

### Running Online

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LEAN-96/ml_unit_testing/HEAD)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LEAN-96/ml_unit_testing/blob/main/notebooks/ML_Unit_Testing.ipynb)

### Option 1: Run in MyBinder
1. Click the "launch binder" badge above
2. Wait for the environment to build (this may take a few minutes)
3. Navigate to `notebooks/ML_Unit_Testing.ipynb`
4. Run the notebook to see test results

### Option 2: Run in Google Colab
1. Click the "Open in Colab" badge above
2. The notebook will open in Google Colab
3. Run the following setup code:
```python
# Clone repository
!git clone https://github.com/LEAN-96/ml_unit_testing.git
%cd ml_unit_testing

# Install requirements
!pip install -r requirements.txt

# Run tests
!python -m unittest -v tests/test_model.py
```

### Option: Manual Setup Colab
1.	Go to Google Colab
2.	File → Open notebook
3.	GitHub tab
4.	Enter repository URL: https://github.com/LEAN-96/ml_unit_testing
5.	Select notebooks/ML_Unit_Testing.ipynb


## Project Requirements Implementation

### Requirement 1: Prediction Function Testing ✅
**Test Case: test_predict_accuracy_and_confusion_matrix()**
- Purpose: Validate model's prediction accuracy
- Implementation:
  - Calculate accuracy score
  - Generate confusion matrix
  - Store test data separately
  - Compare training vs test performance
- Success Criteria:
  - Accuracy > 70%
  - Balanced confusion matrix
  - No significant overfitting

### Requirement 2: Runtime Performance Testing ✅
**Test Case: test_fit_runtime()**
- Purpose: Monitor training function performance
- Implementation:
  - Log baseline runtime
  - Compare test runtime
  - Implement performance threshold
- Success Criteria:
  - Test runtime ≤ 120% of baseline
  - Consistent performance
  - Proper logging

## Expected Results

### Key Metrics Achieved
- Training Accuracy: 97.00%
- Test Accuracy: 96.667%
- Baseline Runtime: ~0.0459 seconds
- Test Runtime: ~0.0436 seconds

You should see output similar to:
```

=== Training Set Performance ===
Training Accuracy: 0.9700
Training Confusion Matrix:
[[346   8]
 [ 13 333]]

=== Test Set Performance ===
Test Accuracy: 0.9667
Test Confusion Matrix:
[[142   4]
 [  6 148]]

Classification Report:
              precision    recall  f1-score   support

           0       0.96      0.97      0.97       146
           1       0.97      0.96      0.97       154

    accuracy                           0.97       300
   macro avg       0.97      0.97      0.97       300
weighted avg       0.97      0.97      0.97       300
```
