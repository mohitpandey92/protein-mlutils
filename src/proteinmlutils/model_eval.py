import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.metrics import roc_curve, auc, confusion_matrix
from matplotlib.ticker import PercentFormatter
from sklearn.preprocessing import label_binarize
import os
import numpy as np
import pandas as pd
import datetime    


sns.set_theme(context='talk', style='white', palette='deep', font='sans-serif', font_scale=1)

def save_classifier_box_plot(y_pred, y_continuous, folder_name,y_label="avg_log2enrich_23.11_and_23.13"):

    sns.catplot(y=y_continuous, x=y_pred,  kind="box")#, data=df_master[0:11100],
    plt.xlabel("Predicted Class")
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(folder_name+"ML_classifier_pred_vs_experiment_values_boxplot.png")

    return None






def save_classification_report(y_test, y_pred, folder_name, y_train=None, y_pred_train=None):
    f = folder_name + 'classification_report_test_data.txt'

    with open(f, 'w') as myfile:
        myfile.write(sk.metrics.classification_report(y_test, y_pred))

    if y_train is not None:
        f = folder_name + 'classification_report_train_data.txt'

        with open(f, 'w') as myfile:
            myfile.write(sk.metrics.classification_report(y_train, y_pred_train))


def save_ROC_curve(y_test, y_pred, y_probability, N_categories,folder_name):
    # Binary classification
    if N_categories==2:
        # Plotting ROC curve.
        sns.set_context('talk')
        scores=y_probability[:, 1]
        fpr, tpr, thresholds = sk.metrics.roc_curve(y_test, scores, pos_label=1)
        plt.figure()
        lw = 2
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label="ROC curve (area = %.2f)"%sk.metrics.roc_auc_score(y_test,y_pred),
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(folder_name+'classifier_ROC_curve_test.png')

    
    # Multi-class classification
    elif N_categories>2:
        #TO DO: under-construction
        # Plotting ROC curve.
        sns.set_context('talk')
        sns.set_style('white')
        y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
        #y_predict_proba=best_model_found.predict_proba(X_test)
        fpr = {}
        tpr = {}
        thresh = {}
        #roc_auc = dict()
        for i in range(N_categories):
            fpr[i], tpr[i], thresh[i] = roc_curve(y_test_binarized[:, i],y_probability[:, i])
            lw = 2
            plt.plot(
                fpr[i],
                tpr[i],
                lw=lw,
                label="class %i(area = %.2f)"%(i,auc(fpr[i], tpr[i]) ))
            #plt.show()

        
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("One versus Rest scheme")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(folder_name+'classifier_ROC_curve.png')

        # Plotting confusion matrix.
        sns.set_context('talk')
        confusion_matrix = sk.metrics.confusion_matrix(y_test, y_pred)
        cm_display = sk.metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)#, display_labels = ["Bad binder", "Good binder"])
        cm_display.plot()
        plt.title("Recall = %.2f, Precision=%.2f \n F1 score=%.2f"%(sk.metrics.recall_score(y_test,y_pred, average= 'macro'),sk.metrics.precision_score(y_test,y_pred, average= 'macro'), sk.metrics.f1_score(y_test,y_pred, average='macro')))
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(folder_name+'classifier_confusion_matrix.png')



def save_confusion_matrix(y_true, y_pred, foldername, labels, ymap=None, figsize=(10,8)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    #sns.set(font_scale=2.0)
    

    if ymap is not None:
        y_pred = [ymap[int(yi)] for yi in y_pred]
        y_true = [ymap[int(yi)] for yi in y_true]
        labels = [ymap[int(yi)] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.2f%%\n%d/%d' % (p, c, s)
            #elif c == 0:
            #    annot[i, j] = ''
            else:
                annot[i, j] = '%.2f%%\n%d' % (p, c)
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm = cm * 100
    cm.index.name = 'True Label'
    cm.columns.name = 'Predicted Label'
    fig, ax = plt.subplots(figsize=figsize)
    plt.yticks(va='center')
    #plt.title("F1_positive=%.2f, F1_negative=%.2f"%(sk.metrics.f1_score(y_true, y_pred, average="binary", pos_label=labels[1]), 
    #                                                  sk.metrics.f1_score(y_true, y_pred, average="binary", pos_label=labels[0])), fontsize=20)

    plt.title("Performance \n Recall = %.2f, Precision=%.2f \n F1 score=%.2f"%(sk.metrics.recall_score(y_true,y_pred, pos_label=labels[1]),sk.metrics.precision_score(y_true,y_pred,pos_label=labels[1]), sk.metrics.f1_score(y_true,y_pred, pos_label=labels[1])))
    ax=sns.heatmap(cm, annot=annot, fmt='', ax=ax, xticklabels=labels, cbar=True, cbar_kws={'format':PercentFormatter()}, yticklabels=labels, cmap="Blues")
    ax.yaxis.set_ticklabels(labels)
    ax.xaxis.set_ticklabels(labels)
    plt.savefig(foldername+'classifier_confusion_matrix.png',  bbox_inches='tight')



  

def save_plots_for_classifier_fn(folder_name, y_test, y_pred, y_continuous, probs, y_label="avg_score",ymap=["low_enrichment", "high_enrichment"]):
    N_categories=len(np.unique(y_test))
    today = datetime.date.today()
    date = today.strftime("%Y_%m_%d")
    folder_name=folder_name[0:-1]+"_"+str(date)+"/"
    os.makedirs(folder_name, exist_ok=True)
    save_classification_report(y_test, y_pred, folder_name=folder_name)
    save_classifier_box_plot(y_pred, y_continuous, folder_name=folder_name, y_label=y_label)
    save_ROC_curve(y_test, y_pred, probs,  N_categories, folder_name=folder_name)
    filename=folder_name+"confusion_matrix.png"
    labels=[0,1]
    save_confusion_matrix(y_test, y_pred, filename, labels,  ymap=ymap, figsize=(10,7))
    print("Saved classification report, box plot and ROC curve successfully")
