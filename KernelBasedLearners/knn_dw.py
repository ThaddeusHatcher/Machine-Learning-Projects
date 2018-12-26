#!/usr/bin/env python

#===============================================================================
#     Copyright (C) 2018 First Last
# 
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
# 
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
# 
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see https://www.gnu.org/licenses.
#===============================================================================

#===============================================================================
# Created on October 3, 2018
# 
# @author: thaddeushatcher
# 
# Course:         COMP7976
# Assignment:     1
# Date:           10/3/18
# AUBGID:         903655081
# Name:           Hatcher, Thaddeus
# 
# Description:    This file contains an implementation of a distance-weighted K-Nearest Neighbor classifier
#                 algorithm as well as functions for extracting character unigram feature vectors from .txt files 
#                 located at a given directory to be used for training and testing of the classifier. 
#===============================================================================

############################
#        IMPORTS           #
############################
import glob
import errno
import math
import os
from numpy import linalg
from numpy import std
from numpy import mean
from sys import platform
import matplotlib.pylab as plt

############################
#    GLOBAL Variables      #
############################
FV_OUT_LIST = [] # Contains list-list of runtime raw feature vector data on a given dataset
NV_OUT_LIST = [] # Contains list-list of runtime normalized feature vector data on a given dataset
ALL_DATASET_VECTORS = [] # Contains list-list of the complete runtime dataset
DATASET_NORM_VECTORS = []
TRAINING_SAMPLES = [] # Contains list-list of runtime training samples from ALL_DATA_VECTORS
TEST_SAMPLES = [] # Contains list-list of runtime test samples from ALL_DATA_VECTORS
AUTHORS = [] # numerical representation of the authors from the training set
SELECTED_DATASET = 0  # tracks which dataset has been selected (0 = Vanderbilt dataset, 1 = CASIS-25 dataset)

############################
# Input Dataset File Paths #
############################
path_to_casis_data = r'..\PyMachineLearning\DataSets\CASIS-25_Dataset'
path_to_vanderbilt_data = r'..\PyMachineLearning\DataSets\VANDERBILT_Dataset'

'''
    Class: TrainingSample
    @summary: TrainingSample is a class that is used to maintain values pertinent to individual training samples
'''
class TrainingSample: 
    def __init__(self, author, feature_vector, normalized_vector):
        self.author = author
        self.feature_vector = feature_vector
        self.normalized_vector = normalized_vector
        self.distance = 0

'''
    Function: vectorProcessFile(fileN = None) 
    @summary: Processes feature vector and normalized feature vector on a specified text file
    @param fileN:  A text file name. 
    @warning: Perform IOError checks before calling vectorProcessFile(fileN = None). Subsequent releases may account for IO Error checking.
              This function is called from outputFileDatasetResults(fPathIn = None, fPathOutRaw = None, fPathOutNorm = None) to which it is co-
              dependent for a relevant, functional output. 
'''
def extractFileFeatures(fileN):
    global ALL_DATASET_VECTORS
    global DATASET_NORM_VECTORS
    global AUTHORS
    feature_dict = {} # Dictionary -- Has same function as HashMap
    feature_vector = [] # The vector for counting character occurrences
    normalized_vector = []

    for i in range(32,127):
        feature_dict[chr(i)] = 0.0
    
    for letter in fileN.read():
        if letter in feature_dict:
            feature_dict[letter] += 1.0

    for key in sorted(feature_dict.keys()):
        feature_vector.append(feature_dict[key])

    dist = linalg.norm(feature_vector)
    
    author = getAuthor(fileN.name)
    if author not in AUTHORS:
        AUTHORS.append(author)

    for key in sorted(feature_dict.keys()):
        normalized_vector.append(feature_dict[key]/dist)
 
    sample = TrainingSample(author, feature_vector, normalized_vector)

    ALL_DATASET_VECTORS.append(sample) 
    DATASET_NORM_VECTORS.append(sample)


'''
    Function: buildDataset()
    @summary: Creates the dataset to be divided into a training set and test set
    @param:   fPathIn     -> The file path directory name for Dataset to be analyzed
'''
def buildDataset(fPathIn):
    global ALL_DATASET_VECTORS
    global AUTHORS
    global SELECTED_DATASET
    global DATASET_NORM_VECTORS
    del ALL_DATASET_VECTORS[:] # Delete the list-list containing runtime raw vector data
    del AUTHORS[:] # Delete the list-list containing runtime vector authors
    del DATASET_NORM_VECTORS[:]
    if (fPathIn == path_to_casis_data):
        SELECTED_DATASET = 1
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(fPathIn)
    files = sorted(os.listdir('.')) # File object of sub-directory path name (sorted)
    del files[0]
    for name in files: 
        try:
            with open(name, errors='ignore') as f: 
                extractFileFeatures(f)        
        except IOError as exc:
            if exc.errno != errno.EISDIR: # Do not fail if a directory is found, just ignore it.
                raise # Propagate other kinds of IOError.

''' 
    @summary:  Creates a list of the samples to be used for training 
    @param:    all_vectors -> The set containing all feature vectors for a given data set
    @param:    num_samples -> The number of samples per author in the data set
    @param:    offset      -> An integer in range [1 up-to num_samples]   
''' 
def extractTrainingSet(all_vectors, num_samples, offset):
    global TRAINING_SAMPLES 
    del TRAINING_SAMPLES[:]
    k = num_samples - offset    
    for i in range(0, len(all_vectors)):
        if i < k:
            TRAINING_SAMPLES.append(all_vectors[i])
        elif i >= k:
            k += num_samples

''' 
    @summary:  Creates a list of the set of samples to be used for testing
    @param:    all_vectors -> The set containing all feature vectors for a given data set
    @param:    num_samples -> The number of samples per author in the data set
    @param:    offset      -> An integer in range [num_samples down-to 1] 
'''       
def extractTestSet(all_vectors, num_samples, offset):
    global TEST_SAMPLES
    del TEST_SAMPLES[:]
    k = num_samples - offset
    for i in range(0, len(all_vectors)):
        if i >= k:
            TEST_SAMPLES.append(all_vectors[i])
            k += num_samples

'''
    Function: getAuthor(fname)
    @summary: Gets the author name from the file path string
    @param:    fname       -> the name of the file
'''
def getAuthor(fname): 
    if (SELECTED_DATASET == 0):
        index1 = fname.find("_") + 1
        index2 = fname.find("_", index1, len(fname))
        return fname[index1 : index2]
    else:
        index1 = fname.find("_")
        return fname[0 : index1]


'''
    Function: distance(target, neighbor)
    @summary: computes the euclidean distance between the target sample and a neighboring sample using
              their normalized feature vectors
    @param target: the target sample to be classified
    @param neighbor: the neighboring sample next to the target sample
    @warning:  
'''
def distance(target_vector, neighbor_vector):
    sum = 0
    for i in range(0, len(target_vector)):
        sum += pow((target_vector[i] - neighbor_vector[i]), 2)
    return math.sqrt(sum)

'''
    Function: getNeighborhood(target, k)
    @summary: generates a list of the k nearest neighbors to the target
    @param target: the target sample to be classified
    @param k: the number of samples in the neighborhood
'''
def getNeighborhood(target, k): 
    global TRAINING_SAMPLES
    neighborhood = [] # list of the k-nearest training samples to the target
    for sample in TRAINING_SAMPLES:
        sample.distance = distance(target.normalized_vector, sample.normalized_vector)
    list.sort(TRAINING_SAMPLES, key=lambda s: s.distance)

    for i in range(0, k):
        neighborhood.append(TRAINING_SAMPLES[i])
        
    return neighborhood

'''
    Function: classifySample()
    @summary: determines the classification (i.e. author) of the sample based on the samples in the neighborhood
    @param target: the target sample to be classified
    @param k: the number of samples in the neighborhood
    @param use_advanced_weighting: boolean value that determines whether advanced weighting will be used
    @return author: the author of the sample
'''
def classifySample(target, k, use_advanced_weighting):
    # generate the k-sized neighborhood
    neighborhood = getNeighborhood(target, k)
    
    # compute weights for each neighbor 
    weights = [] # list of the weights of each neighbor corresonding to
                 # their index in the neighborhood 
    
    # Compute weights for each sample in the neighborhood.
    # The weight for a given sample is equal to the inverse
    # of its euclidean distance from the target.
    for sample in neighborhood:
        weight = (1 / sample.distance)
        weights.append(weight)

    sum1 = 0
    sum2 = 0

    for i in range(0, k):
        d = AUTHORS.index(neighborhood[i].author)
        if (use_advanced_weighting):
            w = 1 / ((1/(weights[i] * getClusterDistanceWeightComponent(neighborhood[i]))))
        else:
            w = weights[i]
        sum1 += d * w
        sum2 += w
    author = int(sum1 / sum2)
    return author

'''
    Function: getAccuracy()
    @summary: computes the accuracy of the classifier by dividing the number of properly identified samples 
              by the total number of samples tested
    @param actual_authors: list of the authors determined by the classifier
    @param expected_authors: list of the authors that correspond to the test samples
    @return accuracy: the computed accuracy of the classifier based on the actual and expected authors
'''
def getAccuracy(actual_authors, expected_authors):
    num_matches = 0
    for i in range(0, len(actual_authors)):
        if (actual_authors[i] == expected_authors[i]):
            num_matches += 1
    accuracy = num_matches/len(actual_authors)
    return accuracy

def runTestsAndGetAccuracy(num_samples, use_advanced_weighting):
    overall_accuracy = 0.0
    for i in range(0, num_samples):
        extractTrainingSet(DATASET_NORM_VECTORS, num_samples, i) 
        extractTestSet(DATASET_NORM_VECTORS, num_samples, i) 

        actual_authors_1 = []
        actual_authors_3 = []
        actual_authors_5 = []
        actual_authors_n = []
        expected_authors = []

        for sample in TEST_SAMPLES:
            expected_authors.append(AUTHORS.index(sample.author))
            actual_authors_1.append(classifySample(sample, 1, use_advanced_weighting))
            actual_authors_3.append(classifySample(sample, 3, use_advanced_weighting))
            actual_authors_5.append(classifySample(sample, 5, use_advanced_weighting))
            actual_authors_n.append(classifySample(sample, len(TRAINING_SAMPLES), use_advanced_weighting))
        
        actual_authors = [actual_authors_1, actual_authors_3, actual_authors_5, actual_authors_n]

        if (i == 1):
            if (SELECTED_DATASET == 0):
                print('Computed accuracies for Vanderbilt Dataset: \n')
            else:
                print('Computed accuracies for Casis-25 Dataset: \n')

        print('offset = %s' %(i))

        accuracy_current_offset = 0.0

        accuracy_1 = getAccuracy(actual_authors_1, expected_authors)
        overall_accuracy += accuracy_1
        accuracy_current_offset += accuracy_1
        print('Accuracy for k = 1:  %s' %(accuracy_1))

        accuracy_3 = getAccuracy(actual_authors_3, expected_authors)
        overall_accuracy += accuracy_3
        accuracy_current_offset += accuracy_3
        print('Accuracy for k = 3:  %s' %(accuracy_3))

        accuracy_5 = getAccuracy(actual_authors_5, expected_authors)
        overall_accuracy += accuracy_5
        accuracy_current_offset += accuracy_5
        print('Accuracy for k = 5:  %s' %(accuracy_5))

        accuracy_n = getAccuracy(actual_authors_n, expected_authors)
        overall_accuracy += accuracy_n
        accuracy_current_offset += accuracy_n
        print('Accuracy for k = n:  %s' %(accuracy_n))
        print('Average Accuracy:  %s' %(accuracy_current_offset / 4))
        print('\n')

        #plotResults(expected_authors, actual_authors, i)
    
    return (overall_accuracy / (4 * num_samples))

''' 
    Function:  meanFeatureVector()
    @summary:  Computes and returns the mean feature vector for an author in AUTHORS by dividing the total number
               of occurences of each of the 95 features by the number of samples in the fv_list
    @param:    fv_list -> A list containing the feature vectors of samples from an author in AUTHORS
    @return:   mean_fv -> The mean feature vector of all feature vectors in the list
''' 
def meanFeatureVector(fv_list):
    mean_fv = [0.0] * 95
    for i in range(0, len(fv_list)):
        for j in range(0, len(fv_list[i])):
            mean_fv[j] += fv_list[i][j]
    for i in range(0, len(mean_fv)):
        mean_fv[i] = mean_fv[i] / len(fv_list)
    return mean_fv

'''
    Function:  getAuthorsSampleList()
    @summary:  Creates a list containing the feature vectors of all samples from each author in AUTHORS
    return:    authors_sample_list -> A 2D list that contains lists of samples from each author, size is same as AUTHORS list
'''
def getAuthorsSampleList():
    authors_sample_list = [] 

    for author in AUTHORS:
        samples = []  # a list is created for each author that contains the samples belonging to them
        for sample in DATASET_NORM_VECTORS:
            if (sample.author == author):
                samples.append(sample)
        authors_sample_list.append(samples)
    
    return authors_sample_list

'''
    Function:  getAuthorClusters()
    @summary:  Creates a cluster for each author/classifier that is simply the mean of the feature vectors
               belonging to their samples
    return:    clusters -> A list containing a mean-cluster for each author in AUTHORS
'''
def getAuthorClusters():
    clusters = []

    authors_sample_list = getAuthorsSampleList()
    
    for sample_list in authors_sample_list:
        norm_fv_list = []
        for sample in sample_list:
            norm_fv_list.append(sample.normalized_vector)
        cluster = meanFeatureVector(norm_fv_list) # the author cluster will simply be the mean of all the normalized feature vectors from their samples
        clusters.append(cluster)

    return clusters
        
def getClusterDistanceWeightComponent(neighbor):
    clusters = getAuthorClusters()
    authors_sample_list = getAuthorsSampleList()
 
    index = AUTHORS.index(neighbor.author)
    author_samples = authors_sample_list[index]
    distances = []
    for sample in author_samples:
        distances.append(distance(clusters[index], sample.normalized_vector))
    dist_std_dev = std(distances)
        
    weight = distance(neighbor.normalized_vector, clusters[index])
    return distance(neighbor.normalized_vector, clusters[index]) / (1.5 * dist_std_dev)


'''
    Function: plotResults()
    @summary: Creates graphs mapping the expected and actual authors for each k value
    @param expected_authors: list of the authors that correspond to the test samples
    @param actual_authors: list of the authors determined by the classifier
    @param offset: determines which sample from selected authors to be tested will be pulled for the test set (in range 1 to num_samples)
'''
def plotResults(expected_authors, actual_authors, offset):
    plt.figure(figsize=(18, 5))
    if (SELECTED_DATASET == 0):
        plt.title('KNN Distance Weighted Authorship Attribution (Vanderbilt Sports Writers Data) Results, Offset: %s' %(offset))
    else:
        plt.title('KNN Distance Weighted Authorship Attribution (CASIS-25 Data) Results, Offset: %s' %(offset))
    for i in range(0, len(actual_authors)):
        plt.plot(expected_authors, actual_authors[i], linewidth=4.0, marker='o', linestyle='-.', alpha=0.4, label='k = 1')
    plt.plot(expected_authors, expected_authors, linewidth=2.0)
    plt.gca().legend(('k = 1','k = 3', 'k = 5', 'k = n'), bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    plt.grid(True)
    plt.xlabel('Desired Output - x')
    plt.ylabel('Actual Output - y')
    plt.show()

buildDataset(path_to_casis_data)
avg_accuracy_d = runTestsAndGetAccuracy(3, True)
print('Average accuracy for Casis-25 Dataset with Optimization:  %s' %(avg_accuracy_d))

avg_accuracy_b = runTestsAndGetAccuracy(3, False)
print('Average accuracy for Casis-25 Dataset without Optimization:  %s' %(avg_accuracy_b))

buildDataset(path_to_vanderbilt_data)
avg_accuracy_a = runTestsAndGetAccuracy(3, False)
print('Average accuracy for Vanderbilt Dataset without Optimization:  %s' %(avg_accuracy_a))


avg_accuracy_c = runTestsAndGetAccuracy(3, True)
print('Average accuracy for Vanderbilt Dataset with Optimization:  %s' %(avg_accuracy_c))

