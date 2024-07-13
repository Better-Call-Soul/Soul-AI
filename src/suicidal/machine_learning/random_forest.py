from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from typing import List

class RandomForestModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.classifier = RandomForestClassifier()
        
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.classifier)
        ])
        
        self.param_grid = {
            'vectorizer__max_features': [1000, 2000],
            'classifier__n_estimators': [50, 100]
        }
        
        self.best_model = None
    
    def train_and_tune(self, X_train: List[str], y_train: List[int], X_test: List[str], y_test: List[int]):
        grid_search = GridSearchCV(self.pipeline, self.param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        
        self.best_model = grid_search.best_estimator_
        print(f'Best parameters: {grid_search.best_params_}')
        print(f'Best cross-validation score: {grid_search.best_score_}')
        
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
