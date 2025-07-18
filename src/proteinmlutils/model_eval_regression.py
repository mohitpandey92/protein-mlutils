
import datetime
import os
import numpy as np
import pandas as pd
import sklearn as sk
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from scipy.stats import kendalltau, pearsonr, spearmanr
from joblib import dump
def save_plot_results(folder_name, model, y_pred, y_test, train_set=False):
    '''
    Its purpose is to plot the regression results showing prediction and actual data.
    Input:
    folder_name: the folder where the model will be saved. Important to put a '_' at the end of the folder name as the script removes the last character and appends date to it.
    y_pred: a 1D array of size N_samples. It has the predicted values.
    y_test: a 1D array of size N_samples. It has the actual values.
    train_set: a boolean. If True, it will save the plot for the training set. If False, it will save the plot for the test set.
    '''
    

    def func(x, m,c):
        return m*x+c

    popt, pcov = curve_fit(func, xdata=y_test, ydata=y_pred)
    
    
    spearmanr_corr, _ = spearmanr(y_test, y_pred)
    sns.set(font_scale=1.5, style='white')
    ax=sns.regplot(x=y_test, y=y_pred, color='b', scatter_kws={'alpha': 0.5, 's': 10}, line_kws={'alpha': 0.5, 'lw':1, 'ls':'-.'}, ci=90)
    plt.setp(ax.collections[1], alpha=0.5)
    plt.plot(y_test, func(y_test, *popt), 'b', alpha=0.35, label=r'fit: %.2fx+ %.2f, $R_{fit}^2$=%.2f' % (tuple(popt)[0],tuple(popt)[1], sk.metrics.r2_score(y_test,func(y_test, *popt) ) ))
    plt.legend(loc='lower right', fontsize=11)
    plt.ylabel('Predicted values')
    plt.xlabel('Actual values')
    plt.title('MAE=%.2f, MAPE=%.2f, \n Spearman Corr=%.2f' % (sk.metrics.mean_absolute_error(y_test,y_pred), sk.metrics.mean_absolute_percentage_error(y_test, y_pred), spearmanr_corr))
    plt.tight_layout()
    if train_set:
        plt.savefig(folder_name+'regression_results_train.png')
        plt.close()
    else:
        plt.savefig(folder_name+'regression_results_test.png')
    return None


def compute_regression_metrics(folder_name, y_test, y_pred, train_set=False):
    kendall_tau, _ = kendalltau(y_test, y_pred)
    pearson_corr, _ = pearsonr(y_test, y_pred)
    spearmanr_corr, _ = spearmanr(y_test, y_pred)
    regression_metrics = {
        'MAE': sk.metrics.mean_absolute_error(y_test, y_pred),
        'MAPE': sk.metrics.mean_absolute_percentage_error(y_test, y_pred),
        'R2': sk.metrics.r2_score(y_test, y_pred),
        'Spearman Correlation': spearmanr_corr,
        'Kendall Tau': kendall_tau,
        'Pearson Correlation': pearson_corr}

    metrics_name_list=list(regression_metrics.values())
    metrics_value_list=list(regression_metrics.keys())

    regression_metrics_df = pd.DataFrame({'Metric': metrics_value_list, 'Value': metrics_name_list})
    if train_set:
        regression_metrics_df.to_csv(folder_name + "regression_metrics_train.csv", index=False)
    else:
        regression_metrics_df.to_csv(folder_name + "regression_metrics_test.csv", index=False)
    
    return None

def feature_importance_plot(model, feature_columns, folder_name):
    '''
    Its purpose is to plot the feature importance of the model.
    Input:
    model: the trained model.
    feature_columns: a list of feature names.
    folder_name: the folder where the model will be saved.
    '''
    features_df=pd.DataFrame({ "importance": model.feature_importances_, "feature":feature_columns})
    features_df=features_df.sort_values(by="importance", ascending=False).reset_index(drop=True)
    features_df.to_csv(folder_name + "feature_importances.csv", index=False)
    
    #sns.set_theme(context='talk', style='white', palette='deep', font='sans-serif', font_scale=1)
    sns.barplot(data=features_df.head(20), x="importance", y="feature")
    plt.title("Top 20 features by importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.savefig(folder_name + "feature_importance.png")
    

def save_regression_results(folder_name, model, y_pred, y_test, train_set=False, feature_columns=None):
    '''
    Its purpose is to plot the regression results showing prediction and actual data.
    Input:
    folder_name: the folder where the model will be saved. Important to put a '_' at the end of the folder name as the script removes the last character and appends date to it.
    y_pred: a 1D array of size N_samples. It has the predicted values.
    y_test: a 1D array of size N_samples. It has the actual values.
    y_continuous: a 1D array of size N_samples. It has the continuous values for boxplot.
    train_set: a boolean. If True, it will save the plot for the training set. If False, it will save the plot for the test set.
    '''
    
    today = datetime.date.today()
    date = today.strftime("%Y_%m_%d")
    folder_name=folder_name[0:-1]+"_"+str(date)+"/"
    os.makedirs(folder_name, exist_ok=True)
    dump_file = folder_name + 'sklearn_model.joblib'
    dump(model, dump_file)    
    
    save_plot_results(folder_name, model, y_pred, y_test, train_set=train_set)
    compute_regression_metrics(folder_name, y_test, y_pred, train_set=train_set)
    if (hasattr(model, 'feature_importances_')) and feature_columns is not None:
        feature_importance_plot(model, feature_columns, folder_name)
    else:
        print("Model does not have feature importances or feature columns are not provided.")
    print("Results saved in folder: ", folder_name)
    return None