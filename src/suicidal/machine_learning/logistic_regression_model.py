from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
from typing import List, Tuple
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report, confusion_matrix)

class LogisticRegressionModel:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.model = LogisticRegression()
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.model)
        ])
        self.param_grid = {
        'vectorizer__max_features': [1000, 2000, 3000],
        'classifier__C': [0.1, 1, 10]
        }
        self.best_model = None
    
    def train_and_tune(self, X_train: List[str], y_train: List[int], X_test: List[str], y_test: List[int]):
        grid_search = GridSearchCV(self.pipeline, self.param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        self.best_model = grid_search.best_estimator_
        print(f'Best parameters: {grid_search.best_params_}')
        print(f'Best cross-validation score: {grid_search.best_score_}')
        
        # Predict on test set and print classification report
        y_pred = self.best_model.predict(X_test)
        print('Classification Report on Test Set:')
        print(classification_report(y_test, y_pred))
    
    def save_model(self, file_path: str):
        joblib.dump(self.best_model, file_path)
    
    def load_model(self, file_path: str):
        self.best_model = joblib.load(file_path)
    
    def predict(self, text: str) -> str:
        if self.best_model is None:
            raise ValueError("Model is not loaded. Please load a model first.")
        prediction = self.best_model.predict([text])
        return 'suicide' if prediction[0] == 1 else 'non-suicide'