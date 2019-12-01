from django.shortcuts import render

# Create your views here.
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, precision_recall_curve, auc, confusion_matrix, roc_auc_score
from imblearn.over_sampling import ADASYN
from rest_framework.views import APIView

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_iris
from rest_framework.response import Response
from django.http import HttpResponse
from rest_framework.exceptions import ValidationError

import matplotlib.pyplot as plt
from matplotlib import cm
import os
from scipy import interp

import numpy as np
import pandas as pd
import csv

from skrules import SkopeRules

# Set the ipython display in such a way that helps the visualization of the rulematrix outputs.
from IPython.display import display, HTML

from classify.send_email import send_email_results

display(HTML(data="""
<style>
    div#notebook-container    { width: 95%; }
    div#menubar-container     { width: 65%; }
    div#maintoolbar-container { width: 99%; }
</style>
"""))

folder_datasets = '/Users/westerops/Desktop/cmpe/cmpe492/RE-2019-Materials/Datasets with features/' #can be an url
filenames = ['dronology','ds3']
#filenames = ['esa-eucl-est', 'ds2', 'ds3', 'dronology', 'reqview', 'leeds', 'wasp']
labels = ['ESA Euclid', 'Helpdesk', 'User mgmt', 'Dronology', 'ReqView', 'Leeds library', 'WASP']
remove = [('dronology', 'f'),('dronology', 'oq'),('wasp', 'f'),('wasp', 'oq')]
oversample = [('ds3', 'f'), ('ds3', 'oq')]
targets = ['IsFunctional', 'IsQuality', 'OnlyFunctional', 'OnlyQuality']




def drop_descriptive_columns(dataset):
    """
    Removes from a dataset, descriptive columns before using it for training the classifiers
    @param dataset: the dataset enriched with features
    @return: the new 'cleaned' dataset
    """
    for c in dataset.columns:
        if c in ['RequirementText', 'Class', 'ProjectID']:
            dataset = dataset.drop(c, axis = 1)
    return dataset

def split_tr_te(dataset, target, to_drop):
    """
    Splits a dataset in training and test set (75 and 25%)
    @param dataset: the dataset to split
    @param target: the target class
    @param to_drop: some additional columns to drop before splitting
    @return: a tuple train_x, test_x, train_y, test_y, with y the target column, x the rest
    """
    return train_test_split(dataset.drop(to_drop, axis=1), dataset[target], test_size=0.25, random_state=42)


def print_scores(actual, pred, name, prob):
    """
    Prints the confusion matrix given the results of a classifier and calculates precision, recall, f1 and AUC score
    @param actual: the original annotation of the dataset (to use for the comparison in order to calculate the above metrics)
    @param pred: the predictions made by the classifier
    @param name: some textual variable to use for verbosity purposes
    @param prob: vector with the probabilities for the predictions in pred
    @return: a list [name, precision, recall, f1, auc]
    """
    f1 = f1_score(actual, pred, average='micro')
    prec = precision_score(actual, pred)
    rec = recall_score(actual, pred)
    auc = roc_auc_score(actual, prob)
    print('=====', name)
    print('Confusion matrix (test)\n', confusion_matrix(actual, pred))
    #     print('F1-Score (micro)', f1)
    #     print('Precision', prec)
    #     print('Recall (train)', rec, '\n')
    return [name, prec, rec, f1, auc]


def build_plot(y_true=[], scores=[], labels=[]):
    """
    Generates two plots: a roc plot and a preision/recall plot
    """
    gradient = np.linspace(0, 1, 10)
    color_list = [cm.tab10(x) for x in gradient]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             sharex=True, sharey=True)
    ax = axes[0]
    n_line = 0
    for i_score, score in enumerate(scores):
        fpr, tpr, _ = roc_curve(y_true[n_line], score, drop_intermediate=False)
        n_line = n_line + 1
        ax.plot(fpr, tpr, linestyle='-.', c=color_list[i_score], lw=1, label=labels[i_score])
    ax.set_title("ROC", fontsize=20)
    ax.set_xlabel('False Positive Rate', fontsize=18)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=18)
    ax.legend(loc='lower center', fontsize=8)

    ax = axes[1]
    n_line = 0
    for i_score, score in enumerate(scores):
        precision, recall, _ = precision_recall_curve(y_true[n_line], score)
        n_line = n_line + 1
        ax.step(recall, precision, linestyle='-.', c=color_list[i_score], lw=1, where='post', label=labels[i_score])
    ax.set_title("Precision-Recall", fontsize=20)
    ax.set_xlabel('Recall (True Positive Rate)', fontsize=18)
    ax.set_ylabel('Precision', fontsize=18)
    ax.legend(loc='lower center', fontsize=8)
    plt.show()


def train_classifier(model, train_x, train_y, name):
    """
    Train a classifier and returns the fit scores for the training set
    """
    model.fit(train_x, train_y)
    pred_train = model.predict(train_x)
    prob = model.predict_proba(train_x)[:, 1]
    scores_line = print_scores(train_y, pred_train, name, prob)
    return scores_line, pred_train, prob


def evaluate_classifier(model, test_x, test_y, name):
    """
    Executes the classifiers on the test set and returns the obtained scores
    """
    pred_test = model.predict(test_x)
    prob = model.predict_proba(test_x)[:, 1]
    scores_line = print_scores(test_y, pred_test, name, prob)
    return scores_line, pred_test, prob


def makeOverSamplesADASYN(X, y):
    """
    Creates new data with oversampled variables by using ADASYN
    @param X: Independent Variable in DataFrame
    @param y: dependent variable in Pandas DataFrame formats
    @return: an oversampled version of the variables
    """
    sm = ADASYN()
    X, y = sm.fit_sample(X, y)
    return X, y


def make_roc_curve(appendix, target, to_drop, golds, probs, names, scores, nrfeat, colors):
    """
    Generates a ROC plot (used in the paper)
    """
    cv = StratifiedKFold(n_splits=10)
    # classifier = svm.SVC(kernel='linear', probability=True, random_state=0)
    # classifier = RandomForestClassifier()
    # classifier = GaussianNB()
    # classifier = KNeighborsClassifier()
    #classifier = MultinomialNB()
    # classifier = DecisionTreeClassifier()
    classifier = LogisticRegression()
    # For fast processing
    # from sklearn.ensemble import GradientBoostingClassifier
    # classifier = GradientBoostingClassifier(random_state=42, n_estimators=30, max_depth = 5)

    tprs = []
    aucs = []
    paucs = []
    ptprs = []
    mean_fpr = np.linspace(0, 1, 100)
    pmean_fpr = np.linspace(0, 1, 100)

    #plt.figure(figsize=(10, 6))

    dataz = pd.read_csv(folder_datasets + 'promise-reclass' + '-' + appendix + '.csv', engine='python')

    # Attempt with project-based fold -- TODO: try another partitioning
    # projects = [[3, 9, 11], [1, 5, 12], [6, 10, 13], [1, 8, 14], [3, 10, 12], [2, 5, 11], [4, 6, 14], [7, 8, 13], [2, 9, 15], [4, 7, 15] ]
    projects = [[3, 9, 11], [1, 5, 12], [6, 10, 13], [1, 8, 14], [3, 12, 15], [2, 5, 11], [6, 9, 14], [7, 8, 13],
                [2, 4, 15], [4, 7, 10]]

    print(target + 'p-fold')
    prec = 0.0
    rec = 0.0
    f1 = 0.0
    for k in projects:
        mytest = dataz.loc[dataz['ProjectID'].isin(k)]
        mytrain = dataz.loc[~dataz['ProjectID'].isin(k)]
        mytest = drop_descriptive_columns(mytest)
        mytest = mytest.drop(mytest.columns[0], axis=1)
        mytrain = drop_descriptive_columns(mytrain)
        mytrain = mytrain.drop(mytrain.columns[0], axis=1)
        myprobs = classifier.fit(mytrain.drop(to_drop, axis=1),
                                 mytrain[target]).predict_proba(mytest.drop(to_drop, axis=1))
        pred = classifier.predict(mytest.drop(to_drop, axis=1))
        prec += precision_score(mytest[target].values.tolist(), pred)
        rec += recall_score(mytest[target].values.tolist(), pred)
        f1 += f1_score(mytest[target].values.tolist(), pred)
        print(k, 'Precision', prec, 'Recall', rec)
        myfpr, mytpr, _ = roc_curve(mytest[target].values.tolist(), myprobs[:, 1], drop_intermediate=False)
        ptprs.append(interp(pmean_fpr, myfpr, mytpr))
        ptprs[-1][0] = 0.0
        my_auc = auc(myfpr, mytpr)
        #     my_auc = roc_auc_score(mytest[target].values.tolist(), myprobs[:, 1])
        paucs.append(my_auc)
        #plt.plot(myfpr, mytpr, lw=1, color=colors['Promise test'], alpha=0.8, linestyle='--',
        #label='Projects bundle %s (AUC = %0.2f)' % (str(k), my_auc))

    print('p-fold', 'Precision', str(prec / 10.0), 'Recall', str(rec / 10.0), 'F1', str(f1 / 10.0), 'AUC',
          str(my_auc / 10.0))

    pmean_tpr = np.mean(ptprs, axis=0)
    pmean_tpr[-1] = 1.0
    #   pmean_auc = auc(pmean_fpr, pmean_tpr)
    pmean_auc = np.mean(paucs, axis=0)
    std_auc = np.std(paucs)
    #plt.plot(pmean_fpr, pmean_tpr, color=colors['Promise test'], linestyle='--',
             #label=r'Mean p-fold (AUC = %0.2f $\pm$ %0.2f)' % (pmean_auc, std_auc),
             #lw=2, alpha=.8)

    std_tpr = np.std(ptprs, axis=0)
    tprs_upper = np.minimum(pmean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(pmean_tpr - std_tpr, 0)
    #plt.fill_between(pmean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
     #                label=r'$\pm$ 1 std. dev. from p-fold')

    #plt.xlim([-0.01, 1.01])
    #plt.ylim([-0.01, 1.01])
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    #plt.legend(loc="lower right")
    #plt.show()

    #plt.figure(figsize=(10, 6))

    dataz = drop_descriptive_columns(dataz)
    dataz = dataz.drop(dataz.columns[0], axis=1)

    X = dataz.drop(to_drop, axis=1)
    y = dataz[target]

    # This code plots the ROC curve with cross validation
    print(target + 'k-fold')
    i = 0
    prec = 0.0
    rec = 0.0
    f1 = 0.0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
        pred = classifier.predict(X.iloc[test])
        prec += precision_score(y.iloc[test], pred)
        rec += recall_score(y.iloc[test], pred)
        f1 += f1_score(y.iloc[test], pred)
        print(i, 'Precision', prec, 'Recall', rec)
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1], drop_intermediate=False)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        #       roc_auc = roc_auc_score(y.iloc[test], probas_[:, 1])
        aucs.append(roc_auc)
        #plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 #label='k-fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1

    print('k-fold', 'Precision', str(prec / 10.0), 'Recall', str(rec / 10.0), 'F1', str(f1 / 10.0), 'AUC',
          str(roc_auc / 10))

    #plt.xlim([-0.01, 1.01])
    # plt.ylim([-0.01, 1.01])
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    #plt.legend(loc="lower right")
    # plt.show()

    # plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
    #          label='Chance', alpha=.8)

    #plt.plot(pmean_fpr, pmean_tpr, color=colors['Promise test'], linestyle=':',
             #l#abel=r'Mean p-fold (AUC = %0.2f $\pm$ %0.2f)' % (pmean_auc, std_auc),
             #lw=2, alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    #   mean_auc = auc(mean_fpr, mean_tpr)
    mean_auc = np.mean(aucs, axis=0)
    std_auc = np.std(aucs)
    #plt.plot(mean_fpr, mean_tpr, color=colors['Promise test'], linestyle='--',
             #label=r'Mean k-fold (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             #lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    #plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     #label=r'$\pm$ 1 std. dev. from k-fold')

    idx = 0
    # colors = ['green', 'brown', 'darkolivegreen', 'purple', 'yellow', 'black', 'red', 'peru']
    for gold in golds:
        fpr, tpr, thresholds = roc_curve(gold, probs[idx])
        #     the_auc = auc(fpr, tpr)
        #plt.plot(fpr, tpr, lw=2, color=colors[names[idx]], alpha=0.8,
        # label='%s (AUC = %0.2f)' % (names[idx], scores[idx]))
        idx += 1

    #plt.xlim([-0.01, 1.01])
    # plt.ylim([-0.01, 1.01])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    # plt.legend(loc="lower right")

    #handles, labels = plt.gca().get_legend_handles_labels()
    order = [2, 1, 0]

    #for i in range(3, len(handles)):
        #order.append(i)

    print('The order is', order)
    #plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc="lower right")

    # plt.show()
    print('roc-' + str(nrfeat) + '-' + appendix + '.pdf')
    #plt.savefig('roc-' + str(nrfeat) + '-' + appendix + '.pdf', dpi=300, bbox_inches='tight')


class ClassifyAPIView(APIView):

    def post(self, request, *args, **kwargs):

        colorpalette = ['#000000', '#e69f00', '#56b4e9', '#009e73', '#f0e442', '#0072b2', '#d55e00', '#cc79a7']
        colors = {'Promise test': colorpalette[0]}
        f = open("results.csv", "w+")
        f.write("'IsFunctional' \t 'IsQuality' \t 'OnlyFunctional' \t 'OnlyQuality'\n")
        writer = csv.writer(f, delimiter='\t')
        for i in range(0, len(filenames)):
            colors.update({labels[i]: colorpalette[i + 1]})

        pd.set_option('precision', 3)

        allfiles = ['Promise train', 'Promise test', 'Macro-average', 'Micro-average']
        allfiles += labels
        allresults = pd.DataFrame(allfiles, columns=['Dataset'])

        feature_sets = []
        feature_set=request.data.get('feature_set',None)
        type = request.data.get('type', None)
        if feature_set is None:
            #raise ValidationError("You have to give feature set")
            feature_set='FinalSel'
        if feature_set != 'FinalSel':
            raise ValidationError("Give valid featureset")
        feature_sets.append(feature_set)
        # classify all datasets with all possible feature sets and for all target class.
        # print the results and the plots
        return_value=[]
        return_test_value = []
        data_all = pd.read_csv(request.data['file'])
        for feature_set in feature_sets:
            for target in targets:
                print("======== Results for feature set '" + feature_set + "' with target '" + target + "' ========")

                to_drop = ['IsFunctional', 'IsQuality']

                appendix = 'ling-' + feature_set

                # read the promise dataset, it is used to train the classifier, which will be then tested on all other datasets
                data = pd.read_csv(folder_datasets + 'promise-reclass' + '-' + appendix + '.csv', engine='python')

                tag = ''
                if target == 'IsFunctional':
                    tag = 'f'
                if target == 'IsQuality':
                    tag = 'q'
                if target == 'OnlyFunctional':
                    tag = 'of'
                    data['IsFunctional'] = data['IsFunctional'] & ~data[
                        'IsQuality']  # calculating the right value for the column
                    target = 'IsFunctional'
                if target == 'OnlyQuality':
                    tag = 'oq'
                    data['IsQuality'] = ~data['IsFunctional'] & data['IsQuality']
                    target = 'IsQuality'

                probs = []
                names = []
                golds = []
                auc_scores = []

                data = drop_descriptive_columns(data)

                # split promise in 75/25
                train_x, test_x, train_y, test_y = split_tr_te(data, target, to_drop)
                res = []
                results=[]
                results.append(target)
                # train the classifier on the 75% of promise
                # model = SVC(kernel='linear', C=1, random_state=0, probability=True)
                # model = RandomForestClassifier()
                # model = GaussianNB()
                # model = KNeighborsClassifier()
                #model = MultinomialNB()
                model = LogisticRegression()
                # model = DecisionTreeClassifier()
                scores_line, _, _ = train_classifier(model, train_x, train_y, 'Promise train')
                # test the performances on the remaining 25
                scores_line, svm_te, svm_pr = evaluate_classifier(model, test_x, test_y, 'Promise test')
                print(scores_line)
                res.append(scores_line)
                probs.append(svm_pr)
                names.append('Promise test')
                golds.append(test_y)
                auc_scores.append(scores_line[4])

                # retrain the classifier on entire promise and test it on the other datasets
                model.fit(data.drop(to_drop, axis=1), data[target])

                precisions = []
                recalls = []
                f1s = []
                aucs = []
                idx = 0

                #print(filename)
                data3=data_all
                #data3 = pd.read_csv(folder_datasets + filename + '-' + appendix + '.csv', engine='python')
                if target == 'OnlyQuality':
                    data3['IsQuality'] = ~data3['IsFunctional'] & data3['IsQuality']
                    target = 'IsQuality'

                if target == 'OnlyFunctional':
                    data3['IsFunctional'] = data3['IsFunctional'] & ~data3['IsQuality']
                    target = 'IsFunctional'

                data3 = drop_descriptive_columns(data3)
                #if (filename, tag) in oversample:
                #    print('Oversampling', filename)
                #    X, y = makeOverSamplesADASYN(data3.drop(to_drop, axis=1), data3[target])
                #else:
                X = data3.drop(to_drop, axis=1)
                y = data3[target]
                scores_line, svm_te, svm_pr = evaluate_classifier(model, X, y, "filename")
                precisions.append(scores_line[1])
                recalls.append(scores_line[2])
                f1s.append(scores_line[3])
                aucs.append(scores_line[4])
                res.append(scores_line)
                #if (filename, tag) not in remove:
                probs.append(svm_pr)
                names.append(labels[idx])
                auc_scores.append(scores_line[4])
                #if (filename, tag) in oversample:
                #  golds.append(y)
                #else:
                golds.append(y.values.tolist())
                idx = idx + 1

                res.append(['Macro-average', np.mean(precisions), np.mean(recalls), np.mean(f1s), np.mean(aucs)])
                res.append(['Std-dev', np.std(precisions), np.std(recalls), np.std(f1s), np.std(aucs)])

                print("Feature set '" + feature_set + "' Target '" + target + "' ========")
                return_value.append(svm_te)
                return_test_value.append({target:res})
                # display the results in the form of tables precision recall f1 auc, plots
                #build_plot(y_true=golds, scores=probs, labels=names)
                make_roc_curve(appendix, target, to_drop, golds, probs, names, auc_scores, '', colors)
                results = pd.DataFrame(res, columns=['Dataset', 'Prec-' + appendix, 'Rec-' + appendix, 'F1-' + appendix,
                                                     'AUC-' + appendix])
                #display(HTML(results.to_html()))


                allresults = pd.merge(allresults, results, on='Dataset')
                #print('Done Innerrrr')

            # display(HTML(allresults.to_html()))
            #print('Done Inner')

        #print('Done')
        if type == 'test':
            return Response(return_test_value, status=200)
        writer.writerows(zip(return_value[0], return_value[1],return_value[2],return_value[3]))
        f.close()
        f = open("results.csv", "r")
        email=request.data.get('email',None)
        if email is not None:
            send_email_results(request.data.get('email'),f)
            os.remove("results.csv")
            return Response("email sended", status=200)
        os.remove("results.csv")
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="somefilename.csv"'

        writer = csv.writer(response)
        writer.writerow(['IsFunctional', 'IsQuality', 'OnlyFunctional', 'OnlyQuality'])
        writer.writerows(zip(return_value[0], return_value[1],return_value[2],return_value[3]))

        return response