from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score,precision_score,f1_score,accuracy_score
import joblib

from utils import *

class LogisticRegressionModel:

  def __init__(self, save_model = False, get_best_params = False):

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
    classifier = LogisticRegression()

    self.grid_search = GridSearchCV(estimator=classifier, param_grid=self.param_grid, cv=5, scoring='accuracy')

    self.grid_search.fit(X_train, y_train)

    self.best_model = self.grid_search.best_estimator_

  def fit(self, X_train, y_train):

    if self.get_best_params:
        self.grid_seaarch(X_train, y_train)

    if hasattr(self.grid_search, 'best_params_'):
        print('Best parameters are set: ', self.grid_search.best_params_)
        self.model = LogisticRegression(**self.grid_search.best_params_)
    else:
        print('Default parameters are set')
        self.model = LogisticRegression(random_state=0)

    self.model.fit(X_train, y_train)

    plot_learning_curve(self.model, "Learning Curve (Logistice Regression)", X_train, y_train, cv=5)

    if self.save_model:
      joblib.dump(self.model, '../../../models/chatbot/ml/lr.pkl')

    return self.model

  def predict(self, X_test, y_test):
    y_pred = self.model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)

    confusion_matrix_sklearn(y_pred, y_test)

    return y_pred, accuracy, F1, recall, precision
  
# lr = LogisticRegressionModel(save_model = True, get_best_params = False)
# model = lr.fit(X_train_vec, y_train)
# accuracy_lr, F1_lr, recall_lr, precision_lr = lr.predict(X_test_vec, y_test)