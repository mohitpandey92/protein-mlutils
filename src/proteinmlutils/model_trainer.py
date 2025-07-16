import pandas as pd
import numpy as np

from scipy import stats
import sklearn as sk

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from sklearn.model_selection import KFold, GroupKFold

class classifier_model_trainer:
    '''
    It trains a classifier model. It can take a list of linear sequences and split the data into training and testing sets. It can also perform k-group cross-validation. The function is used in the RandomForest_classifier_trainer method.
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
        


    def kfold_trainer_using_random_grid_search(self, classifier_model, param_grid, n_jobs=1):
        
        random_search = sk.model_selection.RandomizedSearchCV(classifier_model,param_grid, cv=self.cv_splitter, n_iter=self.n_iter, scoring=self.scoring, refit=self.primary_scoring, n_jobs=n_jobs, verbose=2, return_train_score=False)
        random_search.fit(self.X_train,self.y_train)
        random_search_results_df=pd.DataFrame(random_search.cv_results_)
        best_model_found=random_search.best_estimator_
        y_pred_train=best_model_found.predict(self.X_train)
        print("Best parameters found: ", random_search.best_params_)
        print("Score on training: Recall=", sk.metrics.recall_score(self.y_train,y_pred_train), "Precision=", sk.metrics.precision_score(self.y_train,y_pred_train), "F1 score=",sk.metrics.f1_score(self.y_train, y_pred_train))
        
        return best_model_found, random_search_results_df

    def RandomForestClassifier_trainer(self, training_mode='default', **kwargs):
        '''
        Returns a Random Forest Classifier model with default parameters. The model can be trained for multi-class or binary classification.
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
            
            print("Score on training: Recall=", sk.metrics.recall_score(self.y_train,y_pred_train), "Precision=", sk.metrics.precision_score(self.y_train,y_pred_train), "F1 score=",sk.metrics.f1_score(self.y_train, y_pred_train))
            
        
            return classifier_model, pd.DataFrame({'y_true': self.y_train, 'y_pred': y_pred_train})
        
        
        elif training_mode == 'CrossVal':
            
            if kwargs.get('n_jobs') is not None:
                n_jobs = kwargs['n_jobs']
            else:
                n_jobs = -1

            best_model_found, random_search_results_df=self.kfold_trainer_using_random_grid_search(classifier_model, param_grid, n_jobs=1)
            return best_model_found, random_search_results_df


        else:
            raise ValueError("Training mode must be either 'default' or 'CrossVal'")
        

    def XGBoost_classifier_trainer(self, N_categories, training_mode='default', **kwargs):
        #TODO: Add docstrings. Add purpose of the function. Add documentation for the variables and datatypes. Add comment on the fact the XGBoost is preferred.

        '''
        The purpose it to train a XGBoost classifier
        Input:
        N_categories: an integer. The number of categories for classification.
        kfold: an integer. The number of folds for cross-validation.
        **kwargs: a dictionary. It can have 'n_iter' key. It is used to provide the number of iterations for RandomizedSearchCV.
        Output:
        best_model_found: a trained XGBoost model
        grid_search_results_df: a pandas dataframe that has the results of RandomizedSearchCV
        data_training: a tuple of X_train, y_train
        data_test: a tuple of X_test, y_test
        linear_seq_data: a tuple of linear_seq_train, linear_seq_test
        activity_data: a tuple of activity_train, activity_test
        '''
       
        classifier_model = xgb.XGBClassifier()

        #bifurcating binary and multi-class classification.
        
        if kwargs.get('param_grid') is not None:
            param_grid = kwargs['param_grid']
        else:
            if N_categories>2:
                param_grid = {'objective':['multi:softmax'],
                        'learning_rate': stats.uniform(0.01,0.1),
                        'max_depth': [3,4,5],
                        'min_child_weight': [1,3,5],
                        'gamma': stats.uniform(0.0,0.6),
                        'subsample':  stats.uniform(0.3,0.6),
                        'colsample_bytree': stats.uniform(0.5, 0.4),
                        'n_estimators': [50, 100, 250, 500], 'n_jobs':[-1]}
    
            elif N_categories==2:
                    param_grid = {'objective':['binary:logistic'], 
                        'learning_rate': stats.uniform(0.01,0.1),
                    'max_depth': [1,2,5,10,20],
                    'min_child_weight': [1,3,5],
                    'gamma': stats.uniform(0.0,0.6),
                    'subsample':  stats.uniform(0.3,0.6),
                    'colsample_bytree': stats.uniform(0.5, 0.4),
                    'n_estimators': [30,50,100,500,1000], 'n_jobs':[-1]} 
    
            
        if training_mode == 'default':
            classifier_model.fit(self.X_train, self.y_train)
            y_pred_train = classifier_model.predict(self.X_train)
            
            print("Score on training: Recall=", sk.metrics.recall_score(self.y_train,y_pred_train), "Precision=", sk.metrics.precision_score(self.y_train,y_pred_train), "F1 score=",sk.metrics.f1_score(self.y_train, y_pred_train))
        
    
            return classifier_model, pd.DataFrame({'y_true': self.y_train, 'y_pred': y_pred_train})

        elif training_mode == 'CrossVal':
        
            if kwargs.get('n_jobs') is not None:
                assert n_jobs ==1, "n_jobs should be 1 for XGBoost CrossVal training mode"
            else:
                n_jobs = 1


            best_model_found, random_search_results_df=self.kfold_trainer_using_random_grid_search(classifier_model, param_grid, n_jobs=1)
            return best_model_found, random_search_results_df
        else:
            raise ValueError("Training mode must be either 'default' or 'CrossVal'")
        