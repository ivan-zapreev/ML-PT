import time

import warnings

from tqdm.notebook import tqdm

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from xgboost import XGBClassifier

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

from sklearn.model_selection import train_test_split

from src.utils.logger import logger

from src.model.classifier.dnn_model import instantiate_dnn_model

warnings.filterwarnings("ignore", category=UndefinedMetricWarning) 

_OUTPUT_ROUND_PRECISION = 4

# Define the possible models
_CLASSIFIERS = {
    'Gaussian NB Classifier'        : GaussianNB(),                                                              # 
    'Bernoulli NB Classifier'       : BernoulliNB(),                                                             # 
    'Multinomial NB Classifier'     : MultinomialNB(),                                                           # 
    'Logistic Regression'           : LogisticRegression(max_iter=1000),                                         # 
    'Random Forest Classifier'      : RandomForestClassifier(n_estimators=50, random_state=2),                   # 
    'Ada Boost Classifier'          : AdaBoostClassifier(n_estimators=50, random_state=2),                       # 
    'XGB Classifier'                : XGBClassifier(n_estimators=50,random_state=2),                             # 
    'KNeighbors Classifier'         : KNeighborsClassifier(),                                                    # 
    'Extra Trees Classifier'        : ExtraTreesClassifier(n_estimators=50, random_state=2),                     # 
    'Gradient Boosting Classifier'  : GradientBoostingClassifier(n_estimators=50,random_state=2),                # 
    'SVC Classifier'                : SVC(kernel='sigmoid', gamma=1.0),                                          # 
    'Bagging Classifier'            : BaggingClassifier(n_estimators=50, random_state=2),                        # 
    'Decision Tree Classifier'      : DecisionTreeClassifier(max_depth=10)                                       # 
}

def __train_classifier(clf, X_train, y_train, X_test, y_test):
    start = time.time()   
    logger.debug(f'Start fitting the model...')
    # Fit the model on the training set
    clf.fit(X_train, y_train)
    
    logger.debug(f'Start testing the model...')
    # Predict on the test set
    y_pred = clf.predict(X_test)
    
    logger.debug(f'Computing accuracy and precision...')
    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Compute precision
    precision = precision_score(y_test, y_pred, average='weighted')
    
    # Compute recall
    recall = recall_score(y_test, y_pred, average='weighted')
    
    return accuracy, precision, recall, round(time.time() - start, 2)
    
def train_test_on_models(X, y, test_size=0.3):
    # Split data for train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    logger.info(f'Going to tain/test on: {len(X_train)}/{len(X_test)} elements')
    
    # Define the rounding lambda for output
    round_val = lambda val: round(val, _OUTPUT_ROUND_PRECISION)
    
    # Create and train the models
    results = []
    for name, model in tqdm(_CLASSIFIERS.items(), desc=f'Trying out classifiers'):
        logger.info('--')
        logger.info(f'Considering the model: "{name}"')
        accuracy, precision, recall, time = __train_classifier(model, X_train, y_train, X_test, y_test )
        logger.info(f'The "{name}" model accuracy: {round_val(accuracy)}, precision: {round_val(precision)}, recall: {round_val(recall)}, time: {time} sec.')
        results.append((accuracy, precision, recall, time, name))

    # Sort to get the best accuracy with the best precision
    results = sorted(results, reverse=True)

    return results

def train_test_dnn_model(X, y, test_size=0.3, emb_dim = 30, num_epochs = 100, batch_size = 32, verbose = 1):
    # Split data for train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    logger.info(f'Going to tain/test on: {len(X_train)}/{len(X_test)} elements')
    
    # Define the rounding lambda for output
    round_val = lambda val: round(val, _OUTPUT_ROUND_PRECISION)

    # Instantiate the activity discovery model
    model = instantiate_dnn_model(X.shape[1], emb_dim=emb_dim, num_epochs=num_epochs, batch_size=batch_size, verbose=verbose)

    # Train and test the model
    accuracy, precision, recall, time = __train_classifier(model, X_train, y_train, X_test, y_test )
    logger.info(f'The "Deep Neural Network" model accuracy: {round_val(accuracy)}, precision: {round_val(precision)}, recall: {round_val(recall)}, time: {time} sec.')

    return accuracy, precision, recall, time