import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
import logging
from datetime import datetime, UTC
from src.model import CustomLogisticRegression

# Configure logging with timestamp and more detailed format
logging.basicConfig(
    filename='model_testing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class TestCustomLogisticRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all test methods."""
        logging.info(f"Current Date and Time (UTC): {datetime.now(UTC)}")
        logging.info(f"Current User's Login: LEAN-96")
        
        # Load and prepare data
        data = pd.read_csv('data/Advertising.csv')
        X = data[['Daily Time Spent on Site', 'Age', 'Area Income', 
                 'Daily Internet Usage', 'Male']]
        y = data['Clicked on Ad']
        
        # Split data and save test data
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Save test data
        test_data = pd.concat([cls.X_test, cls.y_test], axis=1)
        test_data.to_csv('data/test_data.csv', index=False)
        logging.info(f"Test data saved to data/test_data.csv")
        
        cls.model = CustomLogisticRegression()

    def test_fit_runtime(self):
        """Test case 2: Check fit function runtime performance."""
        logging.info("\n=== Starting fit runtime test ===")
        
        # First run - baseline
        start_time = time.time()
        self.model.fit(self.X_train, self.y_train)
        baseline_runtime = time.time() - start_time
        logging.info(f"Baseline fit runtime: {baseline_runtime:.4f} seconds")
        print(f"\nBaseline fit runtime: {baseline_runtime:.4f} seconds")

        # Second run - test
        start_time = time.time()
        self.model.fit(self.X_train, self.y_train)
        test_runtime = time.time() - start_time
        logging.info(f"Test fit runtime: {test_runtime:.4f} seconds")
        print(f"Test fit runtime: {test_runtime:.4f} seconds")

        max_runtime = baseline_runtime * 1.2
        logging.info(f"Maximum acceptable runtime (120%): {max_runtime:.4f} seconds")
        
        self.assertLessEqual(
            test_runtime,
            max_runtime,
            f"Runtime ({test_runtime:.4f}s) exceeded limit of {max_runtime:.4f}s"
        )

    def test_predict_accuracy_and_confusion_matrix(self):
        """Test case 1: Verify prediction accuracy and confusion matrix."""
        logging.info("\n=== Starting prediction accuracy test ===")
        
        # Fit the model
        self.model.fit(self.X_train, self.y_train)
        
        # Get training predictions and metrics
        train_predictions = self.model.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, train_predictions)
        train_conf_matrix = confusion_matrix(self.y_train, train_predictions)
        
        # Get test predictions and metrics
        test_predictions = self.model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, test_predictions)
        test_conf_matrix = confusion_matrix(self.y_test, test_predictions)
        
        # Calculate metrics
        class_report = classification_report(self.y_test, test_predictions)
        
        # Print and log training results
        print("\n=== Training Set Performance ===")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Training Confusion Matrix:\n{train_conf_matrix}")
        logging.info("\n=== Training Set Performance ===")
        logging.info(f"Training Accuracy: {train_accuracy:.4f}")
        logging.info(f"Training Confusion Matrix:\n{train_conf_matrix}")
        
        # Print and log test results
        print("\n=== Test Set Performance ===")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Confusion Matrix:\n{test_conf_matrix}")
        print(f"\nClassification Report:\n{class_report}")
        logging.info("\n=== Test Set Performance ===")
        logging.info(f"Test Accuracy: {test_accuracy:.4f}")
        logging.info(f"Test Confusion Matrix:\n{test_conf_matrix}")
        logging.info(f"Classification Report:\n{class_report}")
        
        # Check for overfitting
        accuracy_difference = abs(train_accuracy - test_accuracy)
        logging.info(f"Accuracy difference (train-test): {accuracy_difference:.4f}")
        
        # Assertions
        self.assertGreaterEqual(test_accuracy, 0.7, 
            f"Test accuracy ({test_accuracy:.4f}) below threshold of 0.7")
        self.assertLessEqual(accuracy_difference, 0.1,
            f"Possible overfitting: accuracy difference ({accuracy_difference:.4f}) > 0.1")
        self.assertEqual(np.sum(test_conf_matrix), len(self.y_test),
            "Confusion matrix size mismatch")

    @classmethod
    def tearDownClass(cls):
        logging.info(f"Test suite completed at {datetime.now(UTC)}\n")

if __name__ == '__main__':
    unittest.main(verbosity=2)