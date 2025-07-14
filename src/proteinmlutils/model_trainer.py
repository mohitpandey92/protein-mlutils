import pandas as pd
import numpy as np

import sklearn as sk

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import KFold, GroupKFold

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
        

    def RandomForestClassifier_trainer(self, training_mode='default', **kwargs):
        '''
        Returns a Random Forest Classifier model with default parameters.
        '''
        
        if kwargs.get('param_grid') is not None:
            param_grid = kwargs['param_grid']
        else:
            param_grid = {
            'n_estimators': [50,100, 200],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5,10],
            'min_samples_leaf': [1, 2,3,4],
            'bootstrap': [True, False]
        }
        classifier_model = RandomForestClassifier()
        if training_mode == 'default':
            classifier_model.fit(self.X_train, self.y_train)
            y_pred_train = classifier_model.predict(self.X_train)
            
            print("Best parameters found: ", random_search.best_params_)
            print("Score on training: Recall=", sk.metrics.recall_score(self.y_train,y_pred_train), "Precision=", sk.metrics.precision_score(self.y_train,y_pred_train), "F1 score=",sk.metrics.f1_score(self.y_train, y_pred_train))
            
        
            return classifier_model, pd.DataFrame({'y_true': self.y_train, 'y_pred': y_pred_train})
        
        
        elif training_mode == 'CrossVal':
            
            if kwargs.get('n_jobs') is not None:
                n_jobs = kwargs['n_jobs']
            else:
                n_jobs = -1

            random_search = sk.model_selection.RandomizedSearchCV(classifier_model,param_grid, cv=self.cv_splitter, n_iter=self.n_iter, scoring=self.scoring, refit=self.primary_scoring, n_jobs=n_jobs, verbose=2, return_train_score=False)
            random_search.fit(self.X_train,self.y_train)
            random_search_results_df=pd.DataFrame(random_search.cv_results_)
            best_model_found=random_search.best_estimator_
            y_pred_train=best_model_found.predict(self.X_train)
            print("Best parameters found: ", random_search.best_params_)
            print("Score on training: Recall=", sk.metrics.recall_score(self.y_train,y_pred_train), "Precision=", sk.metrics.precision_score(self.y_train,y_pred_train), "F1 score=",sk.metrics.f1_score(self.y_train, y_pred_train))
            return best_model_found, random_search_results_df


        else:
            raise ValueError("Training mode must be either 'default' or 'CrossVal'")
        

