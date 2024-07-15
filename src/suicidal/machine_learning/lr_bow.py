from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from bag_of_words import *

class LogisticRegressionBowModel:
    def __init__(self, bow: BagOfWords = None):
        self.model = LogisticRegression()
        self.bow = bow
    
    def train_and_tune(self, X_train, y_train, X_val, y_val):
        X_train = self.bow.fit_transform(X_train)
        self.model.fit(X_train, y_train)

        X_val = self.bow.transform(X_val)
        predictions = self.model.predict(X_val)
        print(classification_report(y_val, predictions))
    
    def save_model(self, model_path: str):
        # Save both the model and the bow object together
        joblib.dump({'model': self.model, 'bow': self.bow}, model_path)
    
    def load_model(self, model_path: str):
        # Load both the model and the bow object together
        data = joblib.load(model_path)
        self.model = data['model']
        self.bow = data['bow']
    
    def predict(self, text: str) -> str:
        X = self.bow.transform([text])  # Note the list brackets around `text`
        prediction = self.model.predict(X)[0]
        return  'suicide' if prediction == 1 else 'non-suicide'