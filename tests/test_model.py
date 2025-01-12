import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from src.model import CustomLogisticRegression

class TestCustomLogisticRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all test methods."""
        # Load your data
        data = pd.read_csv('data/Advertising.csv')
        X = data[['Daily Time Spent on Site', 'Age', 'Area Income', 
                 'Daily Internet Usage', 'Male']]
        y = data['Clicked on Ad']
        
        # Split the data
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Initialize the model
        cls.model = CustomLogisticRegression()

    def test_model_initialization(self):
        """Test if model initializes correctly."""
        self.assertIsInstance(
            self.model, 
            CustomLogisticRegression, 
            "Model should be an instance of CustomLogisticRegression"
        )

    def test_fit_method(self):
        """Test if fit method runs without errors."""
        try:
            self.model.fit(self.X_train, self.y_train)
            fitted = True
        except Exception as e:
            fitted = False
        self.assertTrue(fitted, "Model fitting should complete without errors")

    def test_predict_method(self):
        """Test if predict method returns expected shape."""
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        self.assertEqual(
            len(predictions), 
            len(self.X_test), 
            "Predictions length should match test data length"
        )

if __name__ == '__main__':
    unittest.main()