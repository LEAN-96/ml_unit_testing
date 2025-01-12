from sklearn.linear_model import LogisticRegression
from .decorators import my_logger, my_timer

class CustomLogisticRegression:
    def __init__(self):
        self.model = LogisticRegression(max_iter=500)

    @my_logger
    @my_timer
    def fit(self, X_train, y_train):
        """Train the model with the provided data."""
        return self.model.fit(X_train, y_train)

    @my_logger
    @my_timer
    def predict(self, X_test):
        """Make predictions using the trained model."""
        return self.model.predict(X_test)