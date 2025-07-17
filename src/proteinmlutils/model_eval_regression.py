
import datetime
import os
import numpy as np
import pandas as pd
import sklearn as sk
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from joblib import dump
def save_regression_results(folder_name, model, y_pred, y_test, train_set=False):

   '''
    Its purpose is to plot the regression results showing prediction and actual data.
    Input:
    folder_name: the folder where the model will be saved. Important to put a '_' at the end of the folder name as the script removes the last character and appends date to it.
    y_pred: a 1D array of size N_samples. It has the predicted values.
    y_test: a 1D array of size N_samples. It has the actual values.
    train_set: a boolean. If True, it will save the plot for the training set. If False, it will save the plot for the test set.
    '''
    today = datetime.date.today()
    date = today.strftime("%Y_%m_%d")
    folder_name=folder_name[0:-1]+"_"+str(date)+"/"
    os.makedirs(folder_name, exist_ok=True)
    dump_file = folder_name + 'sklearn_model.joblib'
    dump(model, dump_file)    

    def func(x, m,c):
        return m*x+c

    popt, pcov = curve_fit(func, xdata=y_test, ydata=y_pred)
    sns.set_theme(context='talk', style='white', palette='deep', font='sans-serif', font_scale=1)
    sns.regplot(x=y_test, y=y_pred, color='b')
    plt.plot(y_test, func(y_test, *popt), 'b', label=r'fit: %.2fx+ %.2f, $R_{fit}^2$=%.2f' % (tuple(popt)[0],tuple(popt)[1], sk.metrics.r2_score(y_test,func(y_test, *popt) ) ))
    plt.legend(loc='lower right', fontsize=11)
    plt.ylabel('Predicted values')
    plt.xlabel('Actual values')
    plt.title('MAE=%.2f, MAPE=%.2f' % (sk.metrics.mean_absolute_error(y_test,y_pred), sk.metrics.mean_absolute_percentage_error(y_test, y_pred),))
    plt.tight_layout()
    if train_set:
        plt.savefig(folder_name+'regression_results_train.png')
        plt.close()
    else:
        plt.savefig(folder_name+'regression_results_test.png')
    return None
