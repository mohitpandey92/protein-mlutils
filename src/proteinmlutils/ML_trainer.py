import os
import time
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
#from tqdm.notebook import tqdm

import multiprocessing
from joblib import dump,load
from joblib import Parallel, delayed 

from scipy.optimize import curve_fit
from scipy import stats

import sklearn as sk
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import GridSearchCV
#from flashtext import KeywordProcessor
import datetime
from sklearn.ensemble import RandomForestClassifier

sns.set_theme(context='talk', style='white', palette='deep', font='sans-serif', font_scale=1)



class classifier_model_trainer:
    '''
    It trains a classifier model using RandomForestClassifier. It can take a list of linear sequences and split the data into training and testing sets. It can also perform k-group cross-validation. The function is used in the RandomForest_classifier_trainer method.
    Input:
    -----------------
    X_train: a 2D array of size (N_features, N_samples)
    y_train: a 1D of size N_samples
    X_test: a 2D array of size (N_features, N_samples)
    y_test: a 1D of size N_samples
    strategy: a string. Options are 'random', 'groupkfold'
    kfold: an integer. The number of folds for cross-validation.
    **kwargs: a dictionary. It can have 'cv_splitter' and 'param_dict' keys. It is used to provide the cv_splitter and parameters for the model.

    Output:
    ----------------
    best_model_found: a trained RandomForest model
    random_search_results_df: a pandas dataframe that has the results of RandomizedSearchCV
    (X_train, y_train): tuple of training data
    (X_test, y_test): tuple of test data
    '''
    def __init__(self, X_train, y_train, strategy='random', kfold=3, **kwargs):
    
        self.X_train = X_train
        self.y_train = y_train
        self.kfold = kfold
        self.strategy = strategy


        if kwargs.get("scoring") is not None:
            self.scoring = kwargs["scoring"]
        else:
            self.scoring = ["f1", "precision", "recall", "accuracy"]
            
        if kwargs.get("n_iter") is not None:
            self.n_iter = kwargs["n_iter"]
        else:
            self.n_iter = 2
        if kwargs.get("scoring") is not None:
            self.scoring = kwargs["scoring"]
        else:
            self.scoring = ["f1", "precision", "recall", "accuracy"]
        if kwargs.get("primary_scoring") is not None:
            self.primary_scoring = kwargs["primary_scoring"]
        else:
            self.primary_scoring = 'f1'
        if kwargs.get('cv_splitter') is not None:
            self.cv_splitter = kwargs['cv_splitter']
        else:
            if self.strategy == 'random':
                self.cv_splitter = sk.model_selection.KFold(n_splits=self.kfold, shuffle=True, random_state=42)
            elif self.strategy == 'groupkfold':
                self.cv_splitter = sk.model_selection.GroupKFold(n_splits=self.kfold)
            else:
                raise ValueError("Strategy must be either 'random' or 'groupkfold'")
        

    def RandomForestClassifier_trainer(self, **kwargs):
        '''
        Returns a Random Forest Classifier model with default parameters.
        '''
        classifier_model = RandomForestClassifier()

        if kwargs.get('param_dict') is not None:
            self.param_dict = kwargs['param_dict']
        else:
            self.param_dict  = {'n_estimators': [10,20,30],
        'min_samples_leaf': [50,100],
        'max_depth': [10,20],
        'max_features': ["sqrt"],
        'n_jobs':[-1], 'random_state':[0]}

        random_search = sk.model_selection.RandomizedSearchCV(classifier_model, self.param_dict, cv=self.cv_splitter, n_iter=self.n_iter, scoring=self.scoring, refit=self.primary_scoring, n_jobs=1, verbose=2, return_train_score=False)
        random_search.fit(self.X_train,self.y_train)
        random_search_results_df=pd.DataFrame(random_search.cv_results_)
        best_model_found=random_search.best_estimator_
        y_pred_train=best_model_found.predict(self.X_train)
        
        #bunch of print statements to see the performance of the model on training and test model
        print("Score on training: Recall=", sk.metrics.recall_score(self.y_train,y_pred_train), "Precision=", sk.metrics.precision_score(self.y_train,y_pred_train), "F1 score=",sk.metrics.f1_score(self.y_train, y_pred_train))

        return best_model_found, random_search_results_df

