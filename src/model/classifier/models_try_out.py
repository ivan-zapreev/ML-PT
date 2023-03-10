import time

import warnings

import numpy as np

from tqdm.notebook import tqdm

from xgboost import XGBClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

from sklearn.model_selection import RepeatedKFold
from sklearn.exceptions import UndefinedMetricWarning

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from src.utils.logger import logger

warnings.filterwarnings("ignore", category=UndefinedMetricWarning) 

_OUTPUT_ROUND_PRECISION = 4

# Define the rounding lambda for output
round_val = lambda val: round(val, _OUTPUT_ROUND_PRECISION)

# Define the possible models
_CLASSIFIERS = {
    'Gaussian NB Classifier'        : GaussianNB(),                                 # 
    'Bernoulli NB Classifier'       : BernoulliNB(),                                # 
    'Multinomial NB Classifier'     : MultinomialNB(),                              # 
    'Logistic Regression'           : LogisticRegression(max_iter=1000),            # 
    'Random Forest Classifier'      : RandomForestClassifier(n_estimators=50),      # 
    'Ada Boost Classifier'          : AdaBoostClassifier(n_estimators=50),          # 
    'XGB Classifier'                : XGBClassifier(n_estimators=50),               # 
    'KNeighbors Classifier'         : KNeighborsClassifier(),                       # 
    'Extra Trees Classifier'        : ExtraTreesClassifier(n_estimators=50),        # 
    'Gradient Boosting Classifier'  : GradientBoostingClassifier(n_estimators=50),  # 
    'SVC Classifier'                : SVC(kernel='sigmoid', gamma=1.0),             # 
    'Bagging Classifier'            : BaggingClassifier(n_estimators=50),           # 
    'Decision Tree Classifier'      : DecisionTreeClassifier(max_depth=10)          # 
}

def _compute_mode_metrics(y, y_pred):
    # Compute F1 score
    f1_value = f1_score(y, y_pred, average='weighted')
    # Compute accuracy
    accuracy = accuracy_score(y, y_pred)
    # Compute precision
    precision = precision_score(y, y_pred, average='weighted')
    # Compute recall
    recall = recall_score(y, y_pred, average='weighted')
    
    return f1_value, accuracy, precision, recall

def report_model_metrics(y, y_pred, name):
    # Compute model metrics
    f1_value, accuracy, precision, recall = _compute_mode_metrics(y, y_pred)

    logger.info(f'"{name}" model, F1-score: {round_val(f1_value)}, Accuracy (TP+TN): {round_val(accuracy)}, ' \
                f'Precision (FP): {round_val(precision)}, Recall (FN): {round_val(recall)}')

def __compute_model_metrics(y, y_pred, start, end, model_stats):
    logger.debug(f'Computing model metrics...')
    
    # Compute model metrics
    f1_value, accuracy, precision, recall = _compute_mode_metrics(y, y_pred)
    
    # Store the values
    model_stats['f1_score'].append(f1_value)
    model_stats['accuracy'].append(accuracy)
    model_stats['precision'].append(precision)
    model_stats['recall'].append(recall)
    model_stats['time'].append(round(end - start, 2))

def __train_classifier(clf, X_train, y_train, X_test, y_test, model_stats):
    start = time.time()   
    
    logger.debug(f'Start fitting the model...')
    # Fit the model on the training set
    clf.fit(X_train, y_train)
    
    logger.debug(f'Start testing the model...')
    # Predict on the test set
    y_pred = clf.predict(X_test)
    
    # Compute and return model metrics
    __compute_model_metrics(y_test, y_pred, start, time.time(), model_stats) 
    
# Evaluate a model using repeated k-fold cross-validation
def __evaluate_model(clf, name, X, y, n_splits, n_repeats, random_state):
    # Define evaluation procedure
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    
    # Enumerate folds
    model_stats = { 'accuracy' : [], 'precision' : [], 'recall' : [], 'f1_score' : [], 'time' : []}
    for train_ix, test_ix in tqdm(cv.split(X), desc=f'"{name}" - repeated k-fold cross-validation'):
        # prepare data
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        
        # Execute model training testing
        __train_classifier(clf, X_train, y_train, X_test, y_test, model_stats)
        
    return model_stats
    
def __compute_average_stats(model_stats):
    accuracy = np.mean(model_stats['accuracy'])
    precision = np.mean(model_stats['precision'])
    recall = np.mean(model_stats['recall'])
    f1_score = np.mean(model_stats['f1_score'])
    time = np.mean(model_stats['time'])
    
    return accuracy, precision, recall, f1_score, time
    
def train_test_single_model(model, name, X, y, n_splits=10, n_repeats=3, random_state=1):
    # Evaluae the model
    model_stats = __evaluate_model(model, name, X, y, n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    # Compute the average model stats
    accuracy, precision, recall, f1_score, time = __compute_average_stats(model_stats)

    logger.info(f'The "{name}" model f1-score: {round_val(f1_score)}, accuracy: {round_val(accuracy)}, '\
                f'precision: {round_val(precision)}, recall: {round_val(recall)}, time: {round_val(time)} sec.')
    
    return f1_score, accuracy, precision, recall, time
    
def train_test_on_models(X, y, n_splits=10, n_repeats=3, random_state=1):
    logger.info(f'Going to use k-fold cross-validation with n_splits: {n_splits}, n_repeats: {n_repeats}, random_state: {random_state}')
    
    # Create and train the models
    results = []
    for name, model in tqdm(_CLASSIFIERS.items(), desc=f'Trying out classifiers'):
        # Check on the model fitness
        f1_score, accuracy, precision, recall, time = train_test_single_model(model, name, X, y, n_splits, n_repeats, random_state)
        
        # Remember the results
        results.append((f1_score, accuracy, precision, recall, time, name))

    # Sort to get the best accuracy with the best precision
    results = sorted(results, reverse=True)

    return results
