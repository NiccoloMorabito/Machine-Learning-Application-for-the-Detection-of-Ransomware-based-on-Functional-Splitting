# -*- coding: utf-8 -*-
"""
RANSOMWARE DETECTOR

@author: NiccoloMorabito

"""

import os
import csv
import numpy
# from joblib import dump, load
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Plotting
import matplotlib.pyplot as plt
import scikitplot as skplt

import json
import new_loader

# paths of datasets
BENIGN_CENTRIC_PATH = 'datasets/benign_process_centric/'
RW_CENTRIC_PATH = 'datasets/ransomware_process_centric/'
RW_SINGLESPLIT_PATH = 'datasets/single_functional_splitting/'
RW_COMBINEDSPLIT1_PATH = 'datasets/combined_splitting_dlrd_wtrn/'
RW_COMBINEDSPLIT2_PATH = 'datasets/combined_splitting_dlwt_rdrn/'

# other paths
IMAGES_PATH = 'images/'
SUMMARY_PATH = 'summary.txt'
CSV_PATH = 'summary.csv'
AVERAGE_PATH = 'average.csv'
RECALL_TIERTICK_PATH = 'recall_ticks_tier{}.png'

# metrics against nr_processes paths
SINGLE_ACC_CSV_PATH = 'graphs/accuracy_processes_single.csv'
SINGLE_ACC_PNG_PATH = 'graphs/accuracy_processes_single.png'
SINGLE_REC_CSV_PATH = 'graphs/recall_processes_single.csv'
SINGLE_REC_PNG_PATH = 'graphs/recall_processes_single.png'
SINGLE_BALACC_CSV_PATH = 'graphs/balacc_processes_single.csv'
SINGLE_BALACC_PNG_PATH = 'graphs/balacc_processes_single.png'
SINGLE_FPR_CSV_PATH = 'graphs/FPR_processes_single.csv'
SINGLE_FPR_PNG_PATH = 'graphs/FPR_processes_single.png'
COMB_ACC_CSV_PATH   = 'graphs/accuracy_processes_combined.csv'
COMB_ACC_PNG_PATH   = 'graphs/accuracy_processes_combined.png'
COMB_REC_CSV_PATH   = 'graphs/recall_processes_combined.csv'
COMB_REC_PNG_PATH   = 'graphs/recall_processes_combined.png'
COMB_BALACC_CSV_PATH = 'graphs/balacc_processes_combined.csv'
COMB_BALACC_PNG_PATH = 'graphs/balacc_processes_combined.png'
COMB_FPR_CSV_PATH = 'graphs/FPR_processes_combined.csv'
COMB_FPR_PNG_PATH = 'graphs/FPR_processes_combined.png'

# metrics against nr_processes with partial training
ACC_CSV_PATH = 'graphs_partial_training/accuracy_processes_partial_training.csv'
ACC_PNG_PATH = 'graphs_partial_training/accuracy_processes_partial_training.png'
REC_CSV_PATH = 'graphs_partial_training/recall_processes_partial_training.csv'
REC_PNG_PATH = 'graphs_partial_training/recall_processes_partial_training.png'
BALACC_CSV_PATH = 'graphs_partial_training/balacc_processes_partial_training.csv'
BALACC_PNG_PATH = 'graphs_partial_training/balacc_processes_partial_training.png'
FPR_CSV_PATH = 'graphs_partial_training/FPR_processes_partial_training.csv'
FPR_PNG_PATH = 'graphs_partial_training/FPR_processes_partial_training.png'


# classifier types
DECISION_TREE = "DecisionTree"
RANDOM_FOREST = 'RandomForest'
KNN = "KNN"
SVM = "SVM"
NEURAL_NETWORK = "NeuralNetwork"
LOGISTIC_REGRESSION = "LogisticRegression"

# other costants
FEATURES_LABELS = ['LS', 'R', 'W', 'RN', 'F.C.', 'ENT']
CSV_HEADER = ['Tier', 'Tick', 'Accuracy', 'Precision', '#TN', '#FN', '#TP', '#FP','Recall (True positive rate)', 'True Negative Rate', 'False Positive Rate', 'False Negative rate']
NR_SPLITS_LIST = list(range(1,11)) + list(range(15, 55, 5))
BENIGN_LABEL = [1]
MALIGN_LABEL = [-1]
BENIGN = 'benign'
MALIGN = 'ransomware'
SINGLE  = 'single_functional_splitting'
COMB    = 'combined_functional_splitting'

tierToTicks = {
    1: [0.1, 0.13, 0.17, 0.22, 0.29, 0.37, 0.48, 0.63, 0.82, 1, 1.38,
        1.79, 2.3, 3, 3.9, 5, 6.65, 8.65, 11.25, 14.65, 19, 24.7, 32.1, 41, 54, 70.5, 91, 100],
    2: [0.13,  0.22, 0.37,  0.63, 1, 1.79, 3, 5, 8.65, 14.65, 24.7, 41, 70.5, 100],
    3: [0.17, 0.37, 0.82, 1.79, 3.9, 8.65, 19, 41, 100],
    4: [0.22, 0.63, 1.79, 5, 14.65, 41, 100],
    5: [0.29, 1, 3.9, 14.65, 54, 100],
    6: [0.37, 1.79, 8.65, 41, 100],
    7: [0.48, 3, 19, 100],
    8: [0.63, 5, 41, 100],
    9: [0.82, 8.65, 100],
    10: [1, 14.65, 100],
    11: [1.38, 24.7, 100],
    12: [1.79, 41, 100],
    13: [2.3, 70.5, 100],
    14: [3, 100],
    15: [3.9, 100],
    16: [5, 100],
    17: [6.65, 100],
    18: [8.65, 100],
    19: [11.25, 100],
    20: [14.65, 100],
    21: [19, 100],
    22: [24.7, 100],
    23: [32.1, 100],
    24: [41, 100],
    25: [54, 100],
    26: [70.5, 100],
    27: [91, 100],
    28: [100]
}
            
def datasets_to_trainsets():
    '''Load all csv in datasets and create the training set and the test set.'''
    # getting all the 5 datasets
    benign_fv   = new_loader.get_classic_dataset(BENIGN)
    print("benign loaded")
    rans_fv     = new_loader.get_classic_dataset(MALIGN)
    print("malign loaded")
    single_fv   = new_loader.get_single_dataset_2()
    print("single loaded")
    comb1_fv    = new_loader.get_comb_dataset_2(1)
    print("comb1 loaded")
    comb2_fv    = new_loader.get_comb_dataset_2(2)
    print("comb2 loaded")

    # divide training set into two parts: trainset and testset
    x_train = dict()
    y_train = dict()
    x_test = dict()
    y_test = dict()
    for tier, tick2fv in benign_fv.items():
        x_train[tier] = dict()
        y_train[tier] = dict()
        x_test[tier] = dict()
        y_test[tier] = dict()
        for tick, fv in tick2fv.items():
            X = benign_fv[tier][tick] + rans_fv[tier][tick] + single_fv[tier][tick] + comb1_fv[tier][tick] + comb2_fv[tier][tick]
            benign_examples = len(benign_fv[tier][tick])
            malign_examples = len(rans_fv[tier][tick]) + len(single_fv[tier][tick]) + len(comb1_fv[tier][tick]) + len(comb2_fv[tier][tick])
            
            Y = BENIGN_LABEL * benign_examples + MALIGN_LABEL * malign_examples
            try:
                x_train[tier][tick], x_test[tier][tick], y_train[tier][tick], y_test[tier][tick] = train_test_split(X, Y, test_size=0.33)
            except:
                x_train[tier][tick] = X
                y_train[tier][tick] = Y
                x_test[tier][tick] = list()
                y_test[tier][tick] = list()
    
    return x_train, y_train, x_test, y_test

def train_classifiers(x_train, y_train, x_test, y_test, classifier_type):
    ''' It creates models (one for each tier-tick combination), plots images and generates metrics summaries.'''
    
    # Change current working directory according to classifer type
    os.chdir(classifier_type)
    
    # Create or clean summary files
    with open(classifier_type + "_" + SUMMARY_PATH, 'w') as f:
        f.write("")
    with open(classifier_type + "_" + CSV_PATH, 'w', newline='') as csv_file:                 
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(CSV_HEADER)
        
    # Variables for averaging all the metrics
    accuracies = list()
    precisions = list()
    recalls = list()
    TNs = list()
    FNs = list()
    TPs = list()
    FPs = list()
    # TPRs = list()
    TNRs = list()
    # PPVs = list()
    # NPVs = list()
    FPRs = list()
    FNRs = list()
    # FDRs = list()
    
    # Variables for weighted averages
    w_accuracies = list()
    w_precisions = list()
    w_recalls = list()
    w_TNs = list()
    w_FNs = list()
    w_TPs = list()
    w_FPs = list()
    # w_TPRs = list()
    w_TNRs = list()
    # w_PPVs = list()
    # w_NPVs = list()
    w_FPRs = list()
    w_FNRs = list()
    # FDRs = list()
    
    tiertick2weight = get_tiertick_weights()
    
    # Train one model for each tier-tick combination
    for tier, tick2fv in x_train.items():
        for tick, fv in tick2fv.items():
            x = numpy.asarray(fv, dtype='float64')
            y = numpy.asarray(y_train[tier][tick])
            
            weight = tiertick2weight['{}.{}'.format(tier, tick)]
            try:
                # create classifier according to classifier type
                # TODO try to change parameters (these are just default)
                if classifier_type == DECISION_TREE:
                    classifier = DecisionTreeClassifier(max_depth=5)
                elif classifier_type == RANDOM_FOREST:
                    classifier = RandomForestClassifier(criterion='entropy', n_jobs=-1, max_depth=100)
                elif classifier_type == NEURAL_NETWORK:
                    classifier = MLPClassifier(hidden_layer_sizes=(100,300), max_iter=250, learning_rate_init=0.025)
                elif classifier_type == KNN:
                    classifier = KNeighborsClassifier(3)
                elif classifier_type == SVM:
                    classifier = SVC(gamma=2, C=1, probability=True)
                elif classifier_type == LOGISTIC_REGRESSION:
                    classifier = LogisticRegression()
                else:
                    print("Error in classifier_type")
                    return
                    
                classifier.fit(x, y)
            except Exception as err:
                print(err)
                print("No model for Tier: {}, Tick: {}".format(tier, tick))
                print()
                continue
            
            # saving the model
            # filename = MODELS_PATH + classifier_type + "_Tier" + str(tier) + "_Tick" + str(tick) + ".joblib"
            # dump(classifier, filename)
            
            # MODEL METRICS
            testx = numpy.asarray(x_test[tier][tick], dtype='float64')
            testy = numpy.asarray(y_test[tier][tick])
            if len(x_test[tier][tick])==0:
                testx.reshape(1,-1)
            try:
                pred = classifier.predict(testx)
                accuracy = metrics.accuracy_score(testy, pred)
                precision = metrics.precision_score(testy, pred, zero_division=0)
                recall = metrics.recall_score(testy, pred, zero_division=0)
                matrix = metrics.confusion_matrix(testy, pred)
            except:
                accuracy = 0
                precision = 0
                recall = 0
            # False positives, False negatives, True positives, True negatives
            try: TN = matrix[0][0] 
            except: TN = 0
            try: FN = matrix[1][0]
            except: FN = 0
            try: TP = matrix[1][1]
            except: TP = 0
            try: FP = matrix[0][1]
            except: FP = 0
            # Sensitivity, hit rate, recall, or true positive rate
            # TPR = TP/(TP+FN) if TP+FN>0 else 0
            # Specificity or true negative rate
            TNR = TN/(TN+FP) if TN+FP>0 else 0
            # Precision or positive predictive value
            # PPV = TP/(TP+FP)
            # Negative predictive value
            # NPV = TN/(TN+FN)
            # Fall out or false positive rate
            FPR = FP/(FP+TN) if FP+TN>0 else 0
            # False negative rate
            FNR = FN/(TP+FN) if TP+FN>0 else 0
            # False discovery rate
            # FDR = FP/(TP+FP)
            
            # create csv row with relevant metrics
            csv_row = [str(tier), str(tick), accuracy, precision, TN, FN, TP, FP, recall, TNR, FPR, FNR]
            
            # accumulate values
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            TNs.append(TN)
            FNs.append(FN)
            TPs.append(TP)
            FPs.append(FP)
            # TPRs.append(TPR)
            TNRs.append(TNR)
            # PPVs.append(PPV)
            # NPVs.append(NPV)
            FPRs.append(FPR)
            FNRs.append(FNR)
            # FDRs.append(FDR)
            
            # accumulate weighted values
            w_accuracies.append(accuracy*weight)
            w_precisions.append(precision*weight)
            w_recalls.append(recall*weight)
            w_TNs.append(TN*weight)
            w_FNs.append(FN*weight)
            w_TPs.append(TP*weight)
            w_FPs.append(FP*weight)
            # w_TPRs.append(TPR*weight)
            w_TNRs.append(TNR*weight)
            # w_PPVs.append(PPV*weight)
            # w_NPVs.append(NPV*weight)
            w_FPRs.append(FPR*weight)
            w_FNRs.append(FNR*weight)
            # w_FDRs.append(FDR*weight)           
            
            recap = "MODEL OF TIER {}, TICK {}\n".format(tier, tick) + "\tAccuracy: {}\n".format(accuracy) + "\tPrecision: {}\n".format(precision) + "\tRecall: {}\n".format(recall) + "\tTNR: {}\n".format(TNR) + "\tFPR: {}\n".format(FPR) + "\tFNR: {}\n".format(FNR)
            
            # plotting and recapping
            try:
                plot_confusion_matrix(testx, testy, pred, tier, tick, accuracy, precision, recall, classifier_type)
            except:
                recap += "No confusion matrix available for tier {} tick {}.\n".format(tier, tick)
            try:
                plot_learning_curves(classifier, x, y, tier, tick, classifier_type)
            except:
                recap += "No learning curves available for tier {} tick {}.\n".format(tier, tick)
            try:
                probasy = classifier.predict_proba(testx)
                plot_roc_curve(testy, probasy, tier, tick, classifier_type)
            except:
                recap += "No roc curve available for tier {} tick {}.\n".format(tier, tick)
            
            recap += "\n"
            with open (classifier_type + "_" + SUMMARY_PATH, 'a') as f:
                f.write(recap)
            print(recap)
            
            # writing on csv a new row for this model
            with open(classifier_type + "_" + CSV_PATH, 'a', newline='') as csv_file:                 
                csv_writer = csv.writer(csv_file, delimiter=',')
                csv_writer.writerow(csv_row)
    
    # compute the average of all the metrics
    avg_acc =   sum(accuracies) / len(accuracies)
    avg_prec =  sum(precisions) / len(precisions)
    avg_rec =   sum(recalls) / len(recalls)
    avg_TN =    sum(TNs) / len(TNs)
    avg_FN =    sum(FNs) / len(FNs)
    avg_TP =    sum(TPs) / len(TPs)
    avg_FP =    sum(FPs) / len(FPs)
    # avg_TPR =   sum(TPRs) / len(TPRs)
    avg_TNR =   sum(TNRs) / len(TNRs)
    # avg_PPV =   sum(PPVs) / len(PPVs)
    # avg_NPV =   sum(NPVs) / len(NPVs)
    avg_FPR =   sum(FPRs) / len(FPRs)
    avg_FNR =   sum(FNRs) / len(FNRs)
    # avg_FDR =   sum(FDRs) / len(FDRs)
    
    # compute weighted average of all metrics
    w_avg_acc =   sum(w_accuracies)
    w_avg_prec =  sum(w_precisions)
    w_avg_rec =   sum(w_recalls)
    w_avg_TN =    sum(w_TNs)
    w_avg_FN =    sum(w_FNs)
    w_avg_TP =    sum(w_TPs)
    w_avg_FP =    sum(w_FPs)
    # w_avg_TPR =   sum(w_TPRs)
    w_avg_TNR =   sum(w_TNRs)
    # w_avg_PPV =   sum(w_PPVs)
    # w_avg_NPV =   sum(w_NPVs)
    w_avg_FPR =   sum(w_FPRs)
    w_avg_FNR =   sum(w_FNRs)
    # w_avg_FDR =   sum(w_FDRs)
    
    
    # write averaged metrics in another file
    with open(classifier_type + "_" + AVERAGE_PATH, 'w', newline='') as csv_file:                 
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(CSV_HEADER[2:])
        csv_writer.writerow([avg_acc, avg_prec, avg_TN, avg_FN, avg_TP, avg_FP, avg_rec, avg_TNR, avg_FPR, avg_FNR])
        csv_writer.writerow([w_avg_acc, w_avg_prec, w_avg_TN, w_avg_FN, w_avg_TP, w_avg_FP, w_avg_rec, w_avg_TNR, w_avg_FPR, w_avg_FNR])
        

def plot_confusion_matrix(testx, testy, pred, tier, tick, accuracy, precision, recall, classifier_type):
    '''Plot the confusion matrix for the model tier-tick and save it in the images folder.'''
    _title = 'Confusion matrix (tier{}, tick{})'.format(tier, tick)
    xlabel = 'Predicted label\n\nAccuracy = {:0.4f}   -   Precision = {:0.4f}   -   Recall = {:0.4f}'.format(accuracy, precision, recall)
    filename = IMAGES_PATH + classifier_type + '_tier{}_tick{}_confusionmatrix.png'.format(tier, tick)

    skplt.metrics.plot_confusion_matrix(testy, pred, cmap='RdPu', title=_title, figsize=(10, 6))
    plt.xlabel(xlabel)
    plt.savefig(filename)
    plt.close()
    
def plot_learning_curves(classifier, trainx, trainy, tier, tick, classifier_type):
    '''Plot the learning curves for the model tier-tick and save it in the images folder.'''
    skplt.estimators.plot_learning_curve(classifier, trainx, trainy)
    filename = IMAGES_PATH + classifier_type + '_tier{}_tick{}_learningcurves.png'.format(tier, tick)
    plt.savefig(filename)
    plt.close()

def plot_roc_curve(testy, probasy, tier, tick, classifier_type):
    '''Plot the roc curve for the model tier-tick and save it in the images folder.'''    
    skplt.metrics.plot_roc(testy, probasy)
    filename = IMAGES_PATH + classifier_type + '_tier{}_tick{}_roccurve.png'.format(tier, tick)
    plt.savefig(filename)
    plt.close()
    
















    
def get_tiertick_weights_of_models_in(x_test):
    '''
    It returns a dictionary in the form:
        1.0 -> weigth of tier1-tick0 combination
        1.1 -> weight of tier1-tick1 combination
        ...
        2.0 -> weight of tier1-tick0 combination
        ...
        
    Reading from the file 'numexamples_summary.csv'
    '''
    
    tiertick2numexamples = dict()
    with open('numexamples_summary.csv', 'r') as f:
        for line in f.read().strip().split('\n')[1:]:
            line = line.split(',')
            # print(line)
            tier = line[0]
            tick = line[1]
            numexamples = int(line[7])
            tiertick2numexamples['{}.{}'.format(tier,tick)] = numexamples
            
    total = 0
    for tier, tick2fv in x_test.items():
        for tick, fv in tick2fv.items():
            total += tiertick2numexamples['{}.{}'.format(tier,tick)]
    tiertick2weight = dict()
    for tier, tick2fv in x_test.items():
        for tick, fv in tick2fv.items():
            tiertick2weight['{}.{}'.format(tier,tick)] = tiertick2numexamples['{}.{}'.format(tier,tick)] / total
    
    return tiertick2weight

def get_tiertick_weights():
    '''
    It returns a dictionary in the form:
        1.0 -> weigth of tier1-tick0 combination
        1.1 -> weight of tier1-tick1 combination
        ...
        2.0 -> weight of tier1-tick0 combination
        ...
        
    Reading from the file 'numexamples_summary.csv'
    '''
    tiertick2weight = dict()
    with open('numexamples_summary.csv', 'r') as f:
        for line in f.read().strip().split('\n')[1:]:
            line = line.split(',')
            # print(line)
            tier = line[0]
            tick = line[1]
            weight = float(line[8])
            tiertick2weight['{}.{}'.format(tier,tick)] = weight
    
    return tiertick2weight

def get_metrics(classifier, x_test, y_test):
    testx = numpy.asarray(x_test, dtype='float64')
    testy = numpy.asarray(y_test)
    if len(x_test)==0:
        testx.reshape(1,-1)
    # compute metrics of interest
    try:
        pred = classifier.predict(testx)
        try: accuracy = metrics.accuracy_score(testy, pred)
        except: accuracy = 0
        try: recall = metrics.recall_score(testy, pred, zero_division=0)
        except: recall = 0
    except:
        accuracy = 0
        recall = 0
        
    try:
        matrix = metrics.confusion_matrix(testy, pred)
        try: TN = matrix[0][0] 
        except: TN = 0
        try: FP = matrix[0][1]
        except: FP = 0
    except:
        TN = 0
        FP = 0

    
    TNR = TN/(TN+FP) if TN+FP>0 else 0
    FPR = FP/(FP+TN) if FP+TN>0 else 0
    
    
    return accuracy, recall, TNR, FPR

def plot_processes_for(single, x_train, y_train, nrproc2tiers2ticks2x, nrproc2tiers2ticks2y):
    tiertick2weight = get_tiertick_weights()
    
    # GRAPHS FOR SINGLE FUNCTIONAL SPLITTING
    if single:
        print("Plotting processes for single functional splitting")
        # multiplier in single functional splitting is 4 (for nrprocess=2, it creates 8 processes ecc.)
        multiplier = 4
        acc_csv_path = SINGLE_ACC_CSV_PATH
        rec_csv_path = SINGLE_REC_CSV_PATH
        balacc_csv_path = SINGLE_BALACC_CSV_PATH
        FPR_csv_path = SINGLE_FPR_CSV_PATH
        
        acc_png_path = SINGLE_ACC_PNG_PATH
        rec_png_path = SINGLE_REC_PNG_PATH
        balacc_png_path = SINGLE_BALACC_PNG_PATH
        FPR_png_path = SINGLE_FPR_PNG_PATH
    
    # GRAPH FOR COMBINED FUNCTIONAL SPLITTING
    else:
        print("Plotting processes for combined functional splitting")
        # multiplier in combined functional splitting is 2
        multiplier = 2
        acc_csv_path = COMB_ACC_CSV_PATH
        rec_csv_path = COMB_REC_CSV_PATH
        balacc_csv_path = COMB_BALACC_CSV_PATH
        FPR_csv_path = COMB_FPR_CSV_PATH
        
        acc_png_path = COMB_ACC_PNG_PATH
        rec_png_path = COMB_REC_PNG_PATH
        balacc_png_path = COMB_BALACC_PNG_PATH
        FPR_png_path = COMB_FPR_PNG_PATH
        
    # clear csv files
    with open(acc_csv_path, 'w', newline='') as f:                 
        f.write('')
    with open(rec_csv_path, 'w', newline='') as f:                 
        f.write('')
    with open(balacc_csv_path, 'w', newline='') as f:
        f.write('')
    with open(FPR_csv_path, 'w', newline='') as f:
        f.write('')
    
    nrsplit2accuracies  = dict()
    nrsplit2recalls     = dict()
    # balaccs -> balanced_accuracies
    nrsplit2balaccs     = dict()
    nrsplit2FPRs        = dict()
    
    for nrsplit in NR_SPLITS_LIST:
        nrsplit = str(nrsplit)
        nrsplit2accuracies[nrsplit] = list()
        nrsplit2recalls[nrsplit]    = list()
        nrsplit2balaccs[nrsplit]    = list()
        nrsplit2FPRs[nrsplit]       = list()
        
    for tier, tick2fv in x_train.items():
        for tick, fv in tick2fv.items():
            tier = str(tier)
            tick = str(tick)
            # one classifier for each tier-tick combination
            x = numpy.asarray(x_train[tier][tick], dtype='float64')
            y = numpy.asarray(y_train[tier][tick])
            classifier = RandomForestClassifier(criterion='entropy', n_jobs=-1, max_depth=100)
            classifier.fit(x, y)
            
            for nr_splits in NR_SPLITS_LIST:
                # print("nr_splits=", nr_splits)
                y = nr_splits*multiplier
                nr_splits = str(nr_splits)
                # get x_test for a certain tier-tick and based on nr_splits
                x_test = nrproc2tiers2ticks2x[nr_splits][tier][tick]
                y_test = nrproc2tiers2ticks2y[nr_splits][tier][tick]
                
                acc, rec, TNR, FPR = get_metrics(classifier, x_test, y_test)
                # added this if-then statement because FPR should be 1-acc when we train with no positive examples
                #   and in some case we had acc=0 and FPR=0 due to try-except statements
                if int(acc)==0:
                    FPR = 1
                # metrics weighted
                weight = tiertick2weight['{}.{}'.format(tier,tick)]
                balacc = (rec + TNR) / 2 * weight       # balanced accuracy is (TPR + TNR) / 2 = (recall + TNR) / 2
                acc *= weight
                rec *= weight
                FPR *= weight
                nrsplit2accuracies[nr_splits].append(acc)
                nrsplit2recalls[nr_splits].append(rec)
                nrsplit2balaccs[nr_splits].append(balacc)
                nrsplit2FPRs[nr_splits].append(FPR)
    
    processes = list()
    accuracies = list()
    recalls = list()
    balaccs = list()
    FPRs = list()
    # compute average accuracy and average recall for each nr_splits
    for nrsplits in NR_SPLITS_LIST:
        y = nrsplits*multiplier
        nrsplits = str(nrsplits)
        # weighted accuracy and recall (in percentage)
        avg_acc = sum(nrsplit2accuracies[nrsplits]) * 100
        avg_rec = sum(nrsplit2recalls[nrsplits]) * 100
        avg_balacc = sum(nrsplit2balaccs[nrsplits]) * 100
        avg_FPR = sum(nrsplit2FPRs[nrsplits]) * 100
        print("For nrsplits=", nrsplits, " avg_acc=", avg_acc, " - avg_rec=", avg_rec, " - avg_balacc=", avg_balacc, " - avg_FPR=", avg_FPR)
        processes.append(y)
        accuracies.append(avg_acc)
        recalls.append(avg_rec)
        balaccs.append(avg_balacc)
        FPRs.append(avg_FPR)
        with open(acc_csv_path, 'a', newline='') as csv_file:                 
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow([y, avg_acc])
        with open(rec_csv_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow([y, avg_rec])
        with open(balacc_csv_path, 'a', newline='') as csv_file:                 
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow([y, avg_balacc])
        with open(FPR_csv_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow([y, avg_FPR])
        
    
    # Plot accuracy and processes
    X = numpy.array(processes)
    Y = numpy.array(accuracies)
    plt.scatter(X,Y)
    plt.plot(X,Y)
    plt.xlabel('Number of "functional split" ransomware processes')
    plt.ylabel('Detector Accuracy (%)')
    plt.savefig(acc_png_path)
    plt.close()
    
    # Plot recall and processes
    X = numpy.array(processes)
    Y = numpy.array(recalls)
    plt.scatter(X,Y)
    plt.plot(X,Y)
    plt.xlabel('Number of "functional split" ransomware processes')
    plt.ylabel('Detector Recall (%)')
    plt.savefig(rec_png_path)
    plt.close()
    
    # Plot balacc and processes
    X = numpy.array(processes)
    Y = numpy.array(balaccs)
    plt.scatter(X,Y)
    plt.plot(X,Y)
    plt.xlabel('Number of "functional split" ransomware processes')
    plt.ylabel('Detector Balanced-accuracy (%)')
    plt.savefig(balacc_png_path)
    plt.close()
    
    # Plot FPR and processes
    X = numpy.array(processes)
    Y = numpy.array(FPRs)
    plt.scatter(X,Y)
    plt.plot(X,Y)
    plt.xlabel('Number of "functional split" ransomware processes')
    plt.ylabel('Detector FPR (%)')
    plt.savefig(FPR_png_path)
    plt.close()
    

def plot_processes():
    '''
    Plotta diverse metriche in funzione del numero di processi.
    '''
    
    # x_train and y_train (the main training set) are standard: they contain all data
    # x_train, y_train, nrproc2tiers2ticks2x_single, nrproc2tiers2ticks2x_comb, nrproc2tiers2ticks2y_single, nrproc2tiers2ticks2y_comb = new_loader.get_alltrainset_and_parametric_testsets()
    # Read data from file:
    x_train = json.load( open('obj/tier2ticks2xtrain.json') )
    y_train = json.load( open('obj/tier2ticks2ytrain.json') )
    nrproc2tiers2ticks2x_single = json.load( open('obj/nrproc2tiers2ticks2x_single.json') )
    nrproc2tiers2ticks2y_single = json.load( open('obj/nrproc2tiers2ticks2y_single.json') )
    nrproc2tiers2ticks2x_comb = json.load( open('obj/nrproc2tiers2ticks2x_comb.json') )
    nrproc2tiers2ticks2y_comb = json.load( open('obj/nrproc2tiers2ticks2y_comb.json') )
    
    plot_processes_for(True, x_train, y_train, nrproc2tiers2ticks2x_single, nrproc2tiers2ticks2y_single)
    print()
    print('''--------------------------------------------------------------------''')
    print()
    plot_processes_for(False, x_train, y_train, nrproc2tiers2ticks2x_comb, nrproc2tiers2ticks2y_comb)
    
def plot_processes_with_partial_trainset():
    '''
    Plotta varie metriche in funzione del numero dei processi ma con un trainset parziale (fino a un certo nrsplit) mentre il testset contiene anche i successivi.
    '''
    tier2ticks2xtrain, tier2ticks2ytrain, nrsplit2tier2ticks2xtest, nrsplit2tier2ticks2ytest = new_loader.get_partial_trainset_and_parametric_testsets()
    tiertick2weight = get_tiertick_weights()
    
    # clear csv files
    with open(ACC_CSV_PATH, 'w', newline='') as f:                 
        f.write('')
    with open(REC_CSV_PATH, 'w', newline='') as f:                 
        f.write('')
    with open(BALACC_CSV_PATH, 'w', newline='') as f:
        f.write('')
    with open(FPR_CSV_PATH, 'w', newline='') as f:
        f.write('')
        
    nrsplit2accuracies  = dict()
    nrsplit2recalls     = dict()
    # balaccs -> balanced_accuracies
    nrsplit2balaccs     = dict()
    nrsplit2FPRs        = dict()
    
    for nrsplit in NR_SPLITS_LIST:
        nrsplit = str(nrsplit)
        nrsplit2accuracies[nrsplit] = list()
        nrsplit2recalls[nrsplit]    = list()
        nrsplit2balaccs[nrsplit]    = list()
        nrsplit2FPRs[nrsplit]       = list()
        
    for tier, tick2fv in tier2ticks2xtrain.items():
        for tick, fv in tick2fv.items():
            tier = str(tier)
            tick = str(tick)
            # one classifier for each tier-tick combination
            x = numpy.asarray(tier2ticks2xtrain[tier][tick], dtype='float64')
            y = numpy.asarray(tier2ticks2ytrain[tier][tick])
            classifier = RandomForestClassifier(criterion='entropy', n_jobs=-1, max_depth=100)
            classifier.fit(x, y)
            
            for nr_splits in NR_SPLITS_LIST:
                # print("nr_splits=", nr_splits)
                nr_splits = str(nr_splits)
                # get x_test for a certain tier-tick and based on nr_splits
                x_test = nrsplit2tier2ticks2xtest[nr_splits][tier][tick]
                y_test = nrsplit2tier2ticks2ytest[nr_splits][tier][tick]
                
                acc, rec, TNR, FPR = get_metrics(classifier, x_test, y_test)
                # accuracy and recall weighted
                weight = tiertick2weight['{}.{}'.format(tier,tick)]
                balacc = (rec + TNR) / 2 * weight       # balanced accuracy is (TPR + TNR) / 2 = (recall + TNR) / 2
                acc *= weight
                rec *= weight
                FPR *= weight
                nrsplit2accuracies[nr_splits].append(acc)
                nrsplit2recalls[nr_splits].append(rec)
                nrsplit2balaccs[nr_splits].append(balacc)
                nrsplit2FPRs[nr_splits].append(FPR)
    
    processes = list()
    accuracies = list()
    recalls = list()
    balaccs = list()
    FPRs = list()
    # compute average accuracy and average recall for each nr_splits
    for nrsplits in NR_SPLITS_LIST:
        nrsplits = str(nrsplits)
        # weighted accuracy and recall (in percentage)
        avg_acc = sum(nrsplit2accuracies[nrsplits]) * 100
        avg_rec = sum(nrsplit2recalls[nrsplits]) * 100
        avg_balacc = sum(nrsplit2balaccs[nrsplits]) * 100
        avg_FPR = sum(nrsplit2FPRs[nrsplits]) * 100
        print("For nrsplits=", nrsplits, " avg_acc=", avg_acc, " - avg_rec=", avg_rec, " - avg_balacc=", avg_balacc, " - avg_FPR=", avg_FPR)
        processes.append(nrsplits)
        accuracies.append(avg_acc)
        recalls.append(avg_rec)
        balaccs.append(avg_balacc)
        FPRs.append(avg_FPR)
        with open(ACC_CSV_PATH, 'a', newline='') as csv_file:                 
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow([nrsplits, avg_acc])
        with open(REC_CSV_PATH, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow([nrsplits, avg_rec])
        with open(BALACC_CSV_PATH, 'a', newline='') as csv_file:                 
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow([nrsplits, avg_balacc])
        with open(FPR_CSV_PATH, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow([nrsplits, avg_FPR])
     
    # Plot accuracy and processes
    X = numpy.array(processes)
    Y = numpy.array(accuracies)
    plt.scatter(X,Y)
    plt.plot(X,Y)
    plt.xlabel('Number of splits')
    plt.ylabel('Detector Accuracy (%)')
    plt.savefig(ACC_PNG_PATH)
    plt.close()
    
    # Plot recall and processes
    X = numpy.array(processes)
    Y = numpy.array(recalls)
    plt.scatter(X,Y)
    plt.plot(X,Y)
    plt.xlabel('Number of splits')
    plt.ylabel('Detector Recall (%)')
    plt.savefig(REC_PNG_PATH)
    plt.close()
    
    # Plot balacc and processes
    X = numpy.array(processes)
    Y = numpy.array(balaccs)
    plt.scatter(X,Y)
    plt.plot(X,Y)
    plt.xlabel('Number of splits')
    plt.ylabel('Detector Balanced-accuracy (%)')
    plt.savefig(BALACC_PNG_PATH)
    plt.close()
    
    # Plot FPR and processes
    X = numpy.array(processes)
    Y = numpy.array(FPRs)
    plt.scatter(X,Y)
    plt.plot(X,Y)
    plt.xlabel('Number of splits')
    plt.ylabel('Detector FPR (%)')
    plt.savefig(FPR_PNG_PATH)
    plt.close()
    

        
def plot_recall_ticks(plotted_tier):
    '''Method to plot recall of the tier in plotted_tier against tick (taking data from Random Forest summary).'''
    
    os.chdir(RANDOM_FOREST)
    csv_summary = 'RandomForest_summary.csv'
    
    recalls = list()
    ticks = list()
    
    
    with open(csv_summary, 'r') as csv_file:
        for row in csv_file.read().strip().split('\n')[1:]:
            row = row.split(',')
            tier = int(row[0])
            if tier==plotted_tier:
                tick = int(row[1])
                recall = float(row[8])
                
                ticks.append(tick)
                recalls.append(recall*100)
        
    # Plot X and Y
    X = numpy.array(ticks)
    Y = numpy.array(recalls)
    plt.xticks(numpy.arange(0, len(ticks), 1.0))
    plt.yticks(numpy.arange(0, 110, 10))
    plt.scatter(X,Y)
    plt.plot(X,Y)
    plt.xlabel('Tick of tier' + str(plotted_tier))
    plt.ylabel('Recall (%)')
    # plt.xlim(xmin=1)
    # plt.xlim(xmax=24)
    plt.savefig(RECALL_TIERTICK_PATH.format(plotted_tier))
    plt.close()
    
    os.chdir('..')

if __name__=='__main__':
    print()
    # x_train, y_train, x_test, y_test = datasets_to_trainsets()
    
    # Decision tree classifiers
    # train_classifiers(x_train, y_train, x_test, y_test, DECISION_TREE)
    
    # Random Forest classifiers
    # train_classifiers(x_train, y_train, x_test, y_test, RANDOM_FOREST)
    
    # Neural Network classifiers
    # train_classifiers(x_train, y_train, x_test, y_test, NEURAL_NETWORK)
    
    # KNN classifiers
    # train_classifiers(x_train, y_train, x_test, y_test, KNN)
    
    # SVM classifiers
    # train_classifiers(x_train, y_train, x_test, y_test, SVM)
    
    # Logistic Regression classifiers
    # train_classifiers(x_train, y_train, x_test, y_test, LOGISTIC_REGRESSION)
    
    # produce metrics against processes for single-functional-splitting and combined-functional-splitting
    plot_processes()
    
    # plot_recall_ticks(1)
    # plot_recall_ticks(2)
    
    # produce metrics against processes with partial training set
    # plot_processes_with_partial_trainset()
