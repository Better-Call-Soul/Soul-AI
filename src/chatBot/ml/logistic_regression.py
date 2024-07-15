# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score,precision_score,f1_score,accuracy_score
import joblib
import numpy as np

from utils import *
from constants import *

class LogisticRegression:
    
    def __init__(self, learning_rate=0.1, iterations=10000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        # Initialize parameters
        sample_count, feature_count = X.shape
        self.weights = np.zeros(feature_count)
        self.bias = 0

        # Gradient descent
        for _ in range(self.iterations):
            linear_output = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(linear_output)
            
            weight_gradient = np.dot(X.T, (predictions - y)) 
            bias_gradient = np.mean(predictions - y)

            self.weights -= self.learning_rate * weight_gradient
            self.bias -= self.learning_rate * bias_gradient

    def predict_probabilities(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        probabilities = self._sigmoid(linear_output)
        return probabilities

    def predict(self, X):
        probabilities = self.predict_probabilities(X)
        return np.round(probabilities)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

class MultiClassLogisticRegression:
    
    def __init__(self):
        self.classifiers = []

    def train(self, X, y):
        """
        Train a separate model for each class using one-vs-rest approach.
        """
        X_dense = X.toarray() if hasattr(X, 'toarray') else X  # Ensure dense array format
        unique_classes = np.unique(y)

        for current_class in unique_classes:
            # Create binary labels for the current class
            positive_class_mask = y == current_class
            negative_class_mask = y != current_class

            # Extract samples for current binary classification
            X_positive = X_dense[positive_class_mask]
            X_negative = X_dense[negative_class_mask]

            # Concatenate positive and negative samples
            X_combined = np.vstack((X_positive, X_negative))

            # Create binary labels
            y_combined = np.hstack((np.ones(X_positive.shape[0]), np.zeros(X_negative.shape[0])))

            # Train the logistic regression model
            classifier = LogisticRegression()
            classifier.fit(X_combined, y_combined)
            
            # Store the classifier with its corresponding class label
            self.classifiers.append((current_class, classifier))

    def predict(self, X):
        """
        Predict class labels for the given samples.
        """
        X_dense = X.toarray() if hasattr(X, 'toarray') else X  # Ensure dense array format
        class_probabilities = []

        # Get probability predictions from each classifier
        for class_label, classifier in self.classifiers:
            probabilities = classifier.predict_probabilities(X_dense)
            class_probabilities.append((class_label, probabilities))

        predictions = []

        for i in range(X_dense.shape[0]):
            highest_prob = -np.inf
            best_class = None
            
            # Determine the class with the highest probability for each sample
            for class_label, probabilities in class_probabilities:
                if probabilities[i] > highest_prob:
                    highest_prob = probabilities[i]
                    best_class = class_label
            
            predictions.append(best_class)

        return np.array(predictions)
    
class LogisticRegressionModel:

  def __init__(self, save_model = False, get_best_params = False):
    '''
    Initialize the model
    Args:
      save_model (bool): Whether to save the model or not
      get_best_params (bool): Whether to get the best parameters or not      
    '''
    self.param_grid = {
      'C': [0.001, 0.01, 0.1, 1, 10, 100],
      'penalty': ['l1', 'l2'],
      'solver': ['liblinear', 'newton-cg', 'sag', 'saga', 'lbfgs']
    }

    self.grid_search = {}
    self.best_model = None
    self.model = None
    self.save_model = save_model
    self.get_best_params = get_best_params


  def grid_seaarch(self, X_train, y_train):
    '''
    Perform grid search to get the best parameters
    Args:
      X_train (DataFrame): The training data
      y_train (DataFrame): The training labels
    '''
    classifier = LogisticRegression()

    self.grid_search = GridSearchCV(estimator=classifier, param_grid=self.param_grid, cv=5, scoring='accuracy')

    self.grid_search.fit(X_train, y_train)

    self.best_model = self.grid_search.best_estimator_

  def fit(self, X_train, y_train):
    '''
    Fit the model
    Args:
      X_train (DataFrame): The training data
      y_train (DataFrame): The training labels
    '''

    if self.get_best_params:
        self.grid_seaarch(X_train, y_train)

    if hasattr(self.grid_search, 'best_params_'):
        print('Best parameters are set: ', self.grid_search.best_params_)
        self.model = LogisticRegression(**self.grid_search.best_params_)
    else:
        print('Default parameters are set')
        # self.model = LogisticRegression(random_state=0)
        self.model=MultiClassLogisticRegression()


    self.model.train(X_train, y_train)

    # plot_learning_curve(self.model, "Learning Curve (Logistice Regression)", X_train, y_train, cv=5, filename=lr_learning_curve)

    if self.save_model:
      joblib.dump(self.model, lr_model)


  def predict(self, X_test, y_test):
    '''
    Predict the labels
    Args:
      X_test (DataFrame): The testing data
      y_test (DataFrame): The testing labels
    Returns:
      y_pred (DataFrame): The predicted labels
      accuracy (float): The accuracy of the model
      F1 (float): The F1 score of the model
      recall (float): The recall of the model
      precision (float): The precision of the model
    '''
    y_pred = self.model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)

    confusion_matrix_sklearn(y_pred, y_test, filename=lr_confusion_matrix)

    return y_pred, accuracy, F1, recall, precision
  
# lr = LogisticRegressionModel(save_model = True, get_best_params = False)
# model = lr.fit(X_train_vec, y_train)
# accuracy_lr, F1_lr, recall_lr, precision_lr = lr.predict(X_test_vec, y_test)