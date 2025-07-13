import sklearn as sk
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

import sklearn as sk
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

def data_cross_validation_stratification( X_data, Y_data, strategy='random', kfold=5, group_string_arr=None):

    '''
    It takes the input data and stratifies it based on the strategy provided. It can be random or groupkfold.
    Input:
    -----------------
    X_data: a 2D array of size (N_features, N_samples)
    y_data: a 1D of size N_samples
    strategy: a string. Options are 'random', 'groupkfold'
    group_string_arr: a 1D array of size N_samples that has the name of chemical series or protein target. It is used for groupkfold stratification.


    Output:
    ----------------
    A sklearn splitter object that can be used to partition the data into training and validation sets.
    '''
    if strategy=='random':
        kfold_cv = KFold(n_splits=kfold)
        return kfold_cv.split(X_data, Y_data)

    elif strategy=='groupkfold':
        kfold_cv = GroupKFold(n_splits=kfold)
        if group_string_arr is not None:
            le = LabelEncoder()
            le.fit(group_string_arr)
            print('Number of Target classes', len(le.classes_))
            group_arr=le.transform(group_string_arr)
        #TODO: Check for scenario where group_string_arr is None
        return kfold_cv.split(X_data, Y_data, groups=group_arr)

    else:
        raise ValueError("Unexpected stratification method")


def data_train_vs_test_splitting(X_data, Y_data, data_shuffle=True, test_size=0.2):
        '''
        It takes the input data and splits it into training and testing sets. Before splitting it, it can randomize the data if shuffle is set to True. 
        ToDo: It should output a DataFrame 
        Input:
        -----------------
        X_data: a 2D array of size (N_features, N_samples)
        y_data: a 1D of size N_samples
        test_size: data split for training+validation and test


        Output:
        ----------------
        X_test, y_test, X_train, y_train
        '''
        
        if data_shuffle:
            X_data, Y_data = sk.utils.shuffle(X_data, Y_data, random_state=0)

        
        X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X_data, Y_data, test_size=test_size, random_state=42)
        print("Training+validation sample size:", len(y_train))
        print("Test sample size:", len(y_test))

        return X_train, X_test, y_train, y_test

