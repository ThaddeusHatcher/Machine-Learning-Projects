#!/anaconda2/bin/python
# -*- coding: utf-8 -*-

import os
import csv
import sys
import glob
import errno
import random
import numpy as np
from sklearn import svm
from scipy import sparse
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_extraction.text import TfidfTransformer


# ========== GLOBALS ============= #
cwd = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
dsd = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'DataSets')) 
# ================================ #

def Get_Casis_CUDataset():
    """
    Reads the "CASIS-25_CU.txt" file in the CWD and creates two 2D
    numpy vector arrays X and Y split into Training Data and Training Classifiers,
    respectively.
    @param
    """    
    X = []
    Y = []
    with open(os.path.join(cwd,"CASIS-25_CU.txt"), "r") as feature_file: 
        for line in feature_file:
            line = line.strip().split(",")
            Y.append(line[0][:4])
            X.append([float(x) for x in line[1:]])  
    return np.array(X), np.array(Y)

def Get_Vanderbilt_CUDataset():
    """
    Reads the "VANDERBILT-9_CU.csv" file in the CWD and creates two 2D
    numpy vector arrays X and Y split into Training Data and Training Classifiers,
    respectively.
    @params: N/A
    """        
    X = []
    Y = []
    with open(os.path.join(cwd, "VANDERBILT-9_CU.csv"), "r") as feature_file: 
        for line in feature_file:
            line = line.strip().split(",")
            Y.append(line[0][:4])
            X.append([float(x) for x in line[1:]])
    return np.array(X), np.array(Y)

def ResetVanderbiltCUFile(n_splits):
    """
    Reads in the Vanderbilt SEC Sports Writers Dataset file by file,
    creates char unigram feature vectors, and outputs to 'VANDERBILT-9_CU.csv'
    in current working directory. 
    @params:
        n_splits   - Required  : The number of samples per author (Int)
    """    
    feature_vects = []
    del feature_vects[:] 
    files = sorted(glob.glob(os.path.join(dsd, "VANDERBILT_Dataset/*.txt"))) 
    try:
        for file in files: 
            with open(file) as f: 
                feature_dict = {} 
                feature_vector = [] 
                for i in range(32,127):
                    feature_dict[chr(i)] = 0.0
                for letter in f.read():
                    if letter in feature_dict:
                        feature_dict[letter] += 1.0
                for key in sorted(feature_dict.keys()):
                    feature_vector.append(feature_dict[key])
                feature_vects.append(feature_vector)
        with open('VANDERBILT-9_CU.csv', 'w') as outfile:
            mywriter = csv.writer(outfile)
            # manually add header here (optional)
            label = 1000
            sampleCount = 0
            for fv in feature_vects:
                if sampleCount == n_splits:
                    label += 1
                    sampleCount = 0
                mywriter.writerow([label] + fv)
                sampleCount += 1        
    except IOError as exc:
        if exc.errno != errno.EISDIR: # Do not fail if a directory is found, just ignore it.
            raise # Propagate other kinds of IOError.                      

def getFMPopulation(M):
    """
    Retrieves an initial randomized population of feature masks
    of population size = M
    @params:
        M   - Required  : The number of members desired in the population (Int)
    """
    fms = np.random.randint(2, size=(M, 95))
    for fm in fms:
        np.random.shuffle(fm)
    return np.array(fms)


def evaluateLSVMAccuracy(datasetOpt, splitSample, fmask = np.ones((95,), dtype=np.int)): 
    """
    The evaluation function which returns the associated accuracy using a LSVM
    @params:
        datasetOpt   - Required  : Which dataset: 'casis' OR 'vanderbilt' (Str)
        splitSample  - Required  : The number of samples per author (Int)
        fmask        - Optional  : The feature mask. Defaults to baseline (list)
    """    
    if datasetOpt == 'vanderbilt':
        CU_X, Y = Get_Vanderbilt_CUDataset() # n_splits = 6 Baseline = 0.72222222
    elif datasetOpt == 'casis':
        CU_X, Y = Get_Casis_CUDataset() # n_splits = 4 Baseline = 0.73
    else:
        print('Dataset \"datasetOpt\" not specified. System exiting.')
        sys.exit()
    
    # Performs feature masking
    for i in range(len(CU_X)):
        for j in range(95):
            if fmask[j] == 0:
                CU_X[i][j] = 0.0 

    lsvm = svm.LinearSVC()
    # skf here is the split function which creates the k-folds
    skf = StratifiedKFold(n_splits=splitSample, shuffle=True, random_state=0) 
    fold_accuracy = []

    scaler = StandardScaler()
    tfidf = TfidfTransformer(norm=None)
    dense = DenseTransformer()

    for train, test in skf.split(CU_X, Y): # Train is the indices relating the training data and same for test except test
        #train split
        CU_train_data = CU_X[train] # CU_X is the 100  raw feature vectors of CASIS
        train_labels = Y[train] # Y is the labels 1-25

        #test split
        CU_eval_data = CU_X[test]
        eval_labels = Y[test]

        # tf-idf (term frequency-inverse document frequency)
        tfidf.fit(CU_train_data)
        CU_train_data = dense.transform(tfidf.transform(CU_train_data))
        CU_eval_data = dense.transform(tfidf.transform(CU_eval_data))

        # standardization (0 mean -- 0 variance) get mean from all and divide by std dev
        scaler.fit(CU_train_data)
        CU_train_data = scaler.transform(CU_train_data)
        CU_eval_data = scaler.transform(CU_eval_data)

        # normalization
        CU_train_data = normalize(CU_train_data)
        CU_eval_data = normalize(CU_eval_data)

        train_data =  CU_train_data
        eval_data = CU_eval_data

        # evaluation # Learning the training data and labels
        lsvm.fit(train_data, train_labels)
        lsvm_acc = lsvm.score(eval_data, eval_labels)

        fold_accuracy.append((lsvm_acc))

    return np.mean(fold_accuracy, axis = 0)


def uniformCrossover(fm1, fm2):
    """
    UNIFORM CROSS: Select a random parents feature mask for every index
    @params:
        fm1       - Required  : feature mask parent 1 (list)
        fm2       - Required  : feature mask parent 2 (list)
    """
    child = []
    for k in range(95):
        if random.randint(0, 1) == 0:
            child.append(fm1[k])
        else:
            child.append(fm2[k])
    return child

def SinglePointCrossover(fm1, fm2):
    """
    Slices both fm1 and fm2 into two parts at a random index within their
    length and merges them. Both keep their initial sublist up to the crossover
    index, but their ends are swapped.
    @params:
        fm1       - Required  : feature mask parent 1 (list)
        fm2       - Required  : feature mask parent 2 (list)
    """
    pos = int(random.random()*95)
    child = []
    child[:pos] = fm1[:pos]
    child[pos:] = fm2[pos:]
    return child

def getInitialVanderbiltPopulation(n_splits, updatedData = False):
    """
    Creates the initial Vanderbilt Writers Population as a list of tuples whose
    values are (accuracy, feature mask)
    @params:
        n_splits       - Required  : The number of samples per author (Int)
        updatedData    - Optional  : True if Vanderbilt CU file does not exist or data needs to be updated (bool)
    """    
    accList = []
    population = getFMPopulation(25) # The FM Population
    if updatedData:
        ResetVanderbiltCUFile(n_splits)
    for individual in population:
        accuracy = evaluateLSVMAccuracy('vanderbilt', n_splits, individual)
        tup = (round(accuracy, 3), individual)
        accList.append(tup)
    return accList

def getInitialCasisPopulation():
    """
    Creates the initial CASIS-25 Population as a list of tuples whose
    values are (accuracy, feature mask)
    @params: N/A
    """
    accList = []
    population = getFMPopulation(25) # The FM Population
    for individual in population:
        accuracy = evaluateLSVMAccuracy('casis', 4, individual)
        tup = (round(accuracy, 3), individual)
        accList.append(tup)
    return accList 

def getNewPopulation(M, datasetOpt, n_splits):
    """
    Creates a new randomized Population of size M as a list of tuples whose
    values are (accuracy, feature mask)
    @params:
        M           - Required  : The new population size (Int)
        datasetOpt  - Required  : Which dataset: 'casis' OR 'vanderbilt' (Str)
        n_splits    - Required  : XXXX
    """
    # # Mutate the last gen's weakest fit
    # for member in lastGen:
    #     for bit in member:
    #         if random.random() <= mutRate:
    #             bit = 1-bit
    accList = []
    newPop = getFMPopulation(M) # The FM Population
    for member in newPop:
        accuracy = evaluateLSVMAccuracy(datasetOpt, n_splits, member)
        tup = (round(accuracy, 3), member)
        accList.append(tup)
    return accList 

# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def getLSVMClassification(datasetOpt, feat_vect): 
    """    
    @params:
        datasetOpt   - Required  : Which dataset: 'casis' OR 'vanderbilt' (Str)
        feat_vect    - Required  : The raw feature vector we want to exam for Mean Squared Error (list)
    """    
    if datasetOpt == 'vanderbilt':
        CU_X, Y = Get_Vanderbilt_CUDataset() 
        # Optimal feature mask for Vanderbilt dataset -- Accuracy = 86.11%
        OPTIMAL_FM = [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 
                      0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 
                      0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 
                      0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 
                      0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0]
        # How many authors are in this data set?
        num_authors = 8
    elif datasetOpt == 'casis':
        CU_X, Y = Get_Casis_CUDataset() 
        # Optimal feature mask for Casis-25 dataset -- Accuracy = 87%
        OPTIMAL_FM = [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 
                      1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 
                      1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 
                      0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 
                      1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0]
        # How many authors are in this data set?
        num_authors = 25 
    else:
        print('Dataset \"datasetOpt\" not specified. System exiting.')
        sys.exit()
    
    '''
    # Performs feature masking
    for i in range(len(CU_X)):
        for j in range(95):
            if OPTIMAL_FM[j] == 0:
                CU_X[i][j] = 0.0 
    '''
    
    # Linear Support Vector Classification
    lsvm = svm.LinearSVC()
    
    # Preprocessing initializers
    scaler = StandardScaler()
    tfidf = TfidfTransformer(norm=None)
    dense = DenseTransformer()
    
    train_labels = Y

    # tf-idf (term frequency-inverse document frequency)
    tfidf.fit(CU_X)
    CU_train_data = CU_X
    CU_eval_data = dense.transform(sparse.hstack(feat_vect)) 
    CU_eval_data = [feat_vect]
    # Where feature_vect is the random feature vector we want to optimize

    # standardization (0 mean -- 0 variance) get mean from all and divide by std dev
    scaler.fit(CU_train_data)
    CU_train_data = scaler.transform(np.float_(CU_train_data))
    CU_eval_data = scaler.transform(np.float_(CU_eval_data))

    # normalization
    CU_eval_data = normalize(CU_eval_data)
    CU_train_data = normalize(CU_train_data)

    train_data = CU_train_data
    eval_data = CU_eval_data

    # ML the training data and labels
    lsvm.fit(train_data, train_labels)
    
    df = lsvm.decision_function(CU_eval_data)[0]
    classification = lsvm.predict(CU_eval_data)
    return classification[0], df

def getAuthorSampleDFs(datasetOpt, author):
    if datasetOpt == 'vanderbilt':
        ds_samples, ds_authors = Get_Vanderbilt_CUDataset()
        # Optimal feature mask for Vanderbilt dataset -- Accuracy = 86.11%
        OPTIMAL_FM = [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 
                      0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 
                      0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 
                      0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 
                      0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0]
        # How many authors are in this data set?
        num_authors = 8
    elif datasetOpt == 'casis':
        ds_samples, ds_authors = Get_Casis_CUDataset() 
        # Optimal feature mask for Casis-25 dataset -- Accuracy = 87%
        OPTIMAL_FM = [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 
                      1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 
                      1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 
                      0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 
                      1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0]
        # How many authors are in this data set?
        num_authors = 25 
    else:
        print('Dataset \"datasetOpt\" not specified. System exiting.')
        sys.exit()
    # Get authors and corresponding author samples from Vanderbilt dataset

    # Find first in-order occurence of specified author in dataset
    index = 0
    for i in range(0, len(ds_authors)):
        if (ds_authors[i] == author):
            index = i
            break
            

    dfs = []

    for fold in range(0, 8):
        lsvm = svm.LinearSVC()
        # Only samples by the specified author will be run as targets
        train_samples = []
        train_labels = []

        target = []
        target.append(ds_samples[index + fold])

        for i in range(0, len(ds_samples)):
            if ((i - fold) % 8 != 0):
                train_samples.append(ds_samples[i])
                train_labels.append(ds_authors[i])
    
        target = normalize(target)
        train_samples = normalize(train_samples)
        
        lsvm.fit(train_samples, train_labels)
        dfs.append(lsvm.decision_function(target)[0])

    return dfs



        

            

class DenseTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray()


