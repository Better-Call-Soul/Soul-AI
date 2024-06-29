from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score,precision_score,f1_score,accuracy_score
import joblib

from utils import * 
from constants import *

class SVCModel:

  def __init__(self, save_model = False, get_best_params = False):

    self.param_grid = {
        'C':[1, 10, 100],
        'gamma':[1, 0.1, 0.001],
        'kernel':['linear','rbf']
    }

    self.grid_search = {}
    self.model = None
    self.save_model = save_model
    self.get_best_params = get_best_params


  def grid_seaarch(self, X_train, y_train):
    # we sample 10k of the dataset to grid search the SVC due to high computatuin time
    np.random.seed(0)
    random_indices = np.random.choice(X_train.shape[0], 10000, replace=False)

    X_train_sampled = X_train[random_indices]
    y_train_sampled = np.array(y_train)[random_indices]

    classifier = SVC(random_state=0)

    self.grid_search = GridSearchCV(estimator = classifier, param_grid = self.param_grid, cv = 5, scoring='accuracy')

    self.grid_search.fit(X_train_sampled, y_train_sampled)


  def fit(self, X_train, y_train):

    if self.get_best_params:
        self.grid_seaarch(X_train, y_train)

    if hasattr(self.grid_search, 'best_params_'):
        print('Best parameters are set: ', self.grid_search.best_params_)
        self.model = SVC(**self.grid_search.best_params_)
    else:
        print('Default parameters are set')
        self.model = SVC(random_state=0)

    self.model.fit(X_train, y_train)

    plot_learning_curve(self.model, "Learning Curve (SVC)", X_train, y_train, cv=5, filename=svc_learning_curve)

    if self.save_model:
      joblib.dump(self.model, svc_model)


  def predict(self, X_test, y_test):
    y_pred = self.model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)

    confusion_matrix_sklearn(y_pred, y_test, filename=svc_confusion_matrix)

    return y_pred, accuracy, F1, recall, precision
  

# sv = SVCModel(save_model = True, get_best_params = False)
# model = sv.fit(X_train_vec, y_train)
# y_pred, accuracy_sv, F1_sv, recall_sv, precision_sv = sv.predict(X_test_vec, y_test)