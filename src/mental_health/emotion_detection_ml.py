from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.model_selection import GridSearchCV
from ..utils.utils import save_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
import os
from dotenv import load_dotenv


class MentalHealthEmotionClassifier:
    def __init__(self,X_train,y_train,model, vectorizer='count'):
        self.model = model
        self.vectorizer = vectorizer
        self.pipeline = None
        self.X_train = [' '.join(sentence) for dialog in X_train for sentence in dialog]
        self.y_train = [label for dialog_labels in y_train for label in dialog_labels]
        self.emotion_map = {
            0: 'neutral',
            1: 'anger',
            2: 'disgust',
            3: 'fear',
            4: 'happiness',
            5: 'sadness',
            6: 'surprise'
        }

    def _create_vectorizer(self):
        if self.vectorizer == 'count':
            self.vectorizer = CountVectorizer()
        elif self.vectorizer == 'tfidf':
            self.vectorizer = TfidfVectorizer()

    def _compute_class_weights(self):
        # Compute class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)

        # Convert class weights to a dictionary
        class_weight_dict = dict(zip(np.unique(self.y_train), class_weights))
        return class_weight_dict

    def train(self):
        # Create vectorizer
        self._create_vectorizer()

        # Compute class weights
        class_weight_dict = self._compute_class_weights()

        # Set class weight in the model
        self.model.set_params(class_weight=class_weight_dict)

        # Build pipeline
        self.pipeline = Pipeline(steps=[('vectorizer', self.vectorizer), ('classifier', self.model)])

        # Fit the model
        self.pipeline.fit(self.X_train, self.y_train)

    def predict(self, X_test, y_test):
        # Flatten X_test into a list of strings
        X_test = [' '.join(sentence) for dialog in X_test for sentence in dialog]
        y_test = [label for dialog_labels in y_test for label in dialog_labels]
        # Make predictions
        y_pred = self.pipeline.predict(X_test)

        # Generate classification report
        report = classification_report(y_test, y_pred)
        return report

    def tune_parameters(self, param_grid, save_path):
        # Define GridSearchCV
        grid_search = GridSearchCV(self.pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)

        # Perform Grid Search
        grid_search.fit(self.X_train, self.y_train)

        # Print best parameters
        print("Best parameters found:")
        print(grid_search.best_params_)

        # Save the best model
        best_model = grid_search.best_estimator_
        self.model = best_model
        save_model(best_model, save_path)
    
    def get_emotion_probabilities(self, preprocessed_text):
        #return probability and predicted emotion of the sentence
        # Transform the preprocessed sentence into vector representation
        if self.vectorizer.__class__.__name__ == 'CountVectorizer':
            vectorized_sentence = self.vectorizer.transform([" ".join(preprocessed_text)])
        elif self.vectorizer.__class__.__name__ == 'TfidfVectorizer':
            vectorized_sentence = self.vectorizer.transform([" ".join(preprocessed_text)])
        else:
            raise ValueError("Unsupported vectorizer type")

        # Predict the label for the transformed sentence
        predicted_emotion = self.model.predict(vectorized_sentence)[0]

        # Get class probabilities
        class_probabilities = self.model.predict_proba(vectorized_sentence)[0]

        # Map probabilities to emotions
        emotion_probabilities = {self.emotion_map[i]: prob for i, prob in enumerate(class_probabilities)}

        return emotion_probabilities,self.emotion_map[predicted_emotion]


if __name__ == "__main__":
    load_dotenv()

    dataset_path = os.getenv("DATASET_RAW_PATH")+"/detection/DailyDialog/dailydialog"

    train_data_path = f"{dataset_path}/train/dialogues_train.txt"
    train_label_path = f"{dataset_path}/train/dialogues_emotion_train.txt"
    test_data_path = f"{dataset_path}/test/dialogues_test.txt"
    test_label_path = f"{dataset_path}/test/dialogues_emotion_test.txt"

    model_save_path = os.getenv("MODEL_SAVE_PATH")
    # Example data
    # X_train, y_train = read_dataset_dailyDialog(train_data_path, train_label_path)
    # X_test, y_test = read_dataset_dailyDialog(test_data_path, test_label_path)

    # Example model (you can replace this with your actual model)
    
    model = LogisticRegression()

    # Example parameter grid for tuning (you can customize this)
    param_grid = {
        'vectorizer__ngram_range': [(1, 1), (1, 2)],  # Example parameter for CountVectorizer/TfidfVectorizer
        'classifier__C': [0.1, 1, 10]  # Example parameter for LogisticRegression
    }

    # Initialize the classifier
    classifier = MentalHealthEmotionClassifier(X_train, y_train, model=model, vectorizer='tfidf')

    # Train the classifier
    classifier.train()

    # Tune parameters
    classifier.tune_parameters(param_grid, save_path=f"{model_save_path}/lr_ml_main.pkl")
    # classifier.predict(X_test,y_test)
    # Example of getting emotion probabilities for a new sentence
    new_sentence = "I am feeling happy and excited"
    #clean
    probabilities, predicted_emotion = classifier.get_emotion_probabilities(preprocessed_text)
    print("Emotion probabilities:", probabilities)
    print("Predicted emotion:", predicted_emotion)
