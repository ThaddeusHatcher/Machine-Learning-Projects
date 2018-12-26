#!/usr/bin/env python

#===============================================================================
#     Copyright (C) 2018 Donald Tran
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
# Created on Sep 14, 2018
# 
# @author: donaldtran
# 
# Course:         COMP7976
# Assignment:     1
# Date:           09/14/18
# AUBGID:         DZT0021
# Name:           Tran, Don
# 
# Description:    This module performs file I/O and contains logic for character unigram/ 
#                 feature vector analysis on a given subsequence of text to determine the 
#                 character frequency on ASCII values 0x20 (inclusive) through 0x7F (exclusive). 
#                 File I/O is performed on text files within a specified subdirectory.
# 
# Sources/Uses:   1) https://www.tutorialspoint.com/python/python_files_io.htm 
#                    // File I/O in Python
#                 2) https://askubuntu.com/a/352202  
#                    // Reading all files in a specified directory
#                 3) https://docs.python.org/3/library/functions.html#chr
#                    // About built-in Python function chr(i)
#                 4) https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html
#                    // Vector normalization documentation
#                 5) https://stackoverflow.com/a/1401828/9718306
#                    // Implementation of numpy linalg.norm() 
#                 6) https://stackoverflow.com/a/41915174/9718306
#                    // Delete file contents before writing or appending (Zero-indexing)
#                 7) https://stackoverflow.com/a/12093995/9718306
#                    // Iteratively reading an entire sub-directory as sorted *** 
#                 8) https://stackoverflow.com/a/6340411/9718306
#                    // Helpful list-list iteration
#                 9) https://www.sanfoundry.com/python-program-count-number-words-characters-file/
#                    // How to count number of words in a file -- Helpful for collecting stats
#                 10) http://dataaspirant.com/2015/04/11/five-most-popular-similarity-measures-implementation-in-python/
#                    // How to compute Euclidean & Manhattan Distance between two vectors which "should" be equal
#                    // xRef: function printCasisOutputSimilarity() for usage
#===============================================================================


############################
#        IMPORTS           #
############################
import glob
import errno
from numpy import linalg
from math import sqrt

############################
#    GLOBAL Variables      #
############################
FV_OUT_LIST = [] # Contains list-list of runtime raw feature vector data on a given dataset
NV_OUT_LIST = [] # Contains list-list of runtime normalized feature vector data on a given dataset

############################
# Input Dataset File Paths #
############################
path_to_casis_data = '../DataSets/CASIS-25_Dataset/*.txt'
path_to_casis_origin_fv =  '../DataSets/CASIS-25_Output_Origin/Homework1_CASIS-25fvs_rawcu.txt'
path_to_vanderbilt_data = '../DataSets/VANDERBILT_Dataset/*.txt'  

#############################
# Output Results File Paths #
#############################
path_to_casis_raw_output = 'DataOutput/CASIS-25/Homework1_CASIS-25fvs_rawcu.txt'
path_to_casis_norm_output = 'DataOutput/CASIS-25/Homework1_CASIS-25fvs_ncu.txt'
path_to_vanderbilt_raw_output = 'DataOutput/VANDERBILT/Homework1_VANDERBILTfvs_rawcu.txt'
path_to_vanderbilt_norm_output = 'DataOutput/VANDERBILT/Homework1_VANDERBILTfvs_ncu.txt'



'''
    Function: vectorProcessFile(fileN = None) 
    @summary: Processes feature vector and normalized feature vector on a specified text file
    @param fileN:  A text file name. 
    @warning: Perform IOError checks before calling vectorProcessFile(fileN = None). Subsequent releases may account for IO Error checking.
              This function is called from outputFileDatasetResults(fPathIn = None, fPathOutRaw = None, fPathOutNorm = None) to which it is co-
              dependent for a relevant, functional output. 
'''
def vectorProcessFile(fileN = None):
    #########################################################
    # Initializes all values in the feature dictionary to   #
    # float 0.0 for our initial histogram                   #
    #########################################################
    feature_dict = {} # Dictionary -- Has same function as HashMap
    feature_vector = [] # The vector for counting character occurrences
    normalized_vector = [] # The normalized vector
    global FV_OUT_LIST
    global NV_OUT_LIST 
    
    #########################################################
    # Decimal values 32 - 127 are the relevant ASCII vals   #
    # This initializes our keys and sets all values to 0    #
    #########################################################
    for i in range(32,127):
        feature_dict[chr(i)] = 0.0
    
    #########################################################
    # Performs a file read and analysis on every char       #
    # until EOF                                             #
    #########################################################
    for letter in fileN.read():
        if letter in feature_dict:
            feature_dict[letter] += 1.0
     
    #########################################################
    # Populate the feature vector from our completed        #
    # dictionary                                            #
    #########################################################
    for key in sorted(feature_dict.keys()):
        feature_vector.append(feature_dict[key])
        
    # Compute magnitude using the completed feature vector
    dist = linalg.norm(feature_vector)     
    
    # Append feature vector to global array list
    FV_OUT_LIST.append(feature_vector) 
    
    ###########################################################
    # Populate the normalized vector based on magnitude ratio #
    ###########################################################
    for key in sorted(feature_dict.keys()):
        normalized_vector.append(feature_dict[key]/dist)
    
    # Append normalized vector to global array list
    NV_OUT_LIST.append(normalized_vector) 
    

'''
    Function: outputFileDatasetResults(fPathIn = None, fPathOutRaw = None, fPathOutNorm = None)
    @summary: Performs feature vector list output to text file on a specified dataset delimited via carriage return
    @param fPathIn: The file path directory name for Dataset to be analyzed
    @param fPathOutRaw: The output file path for raw feature vector data on the specified dataset
    @param fPathOutNorm: The output file path for the normalized feature vector data on the specified dataset
    @warning: Has function dependency vectorProcessFile(fileN = None)
'''
def outputFileDatasetResults(fPathIn = None, fPathOutRaw = None, fPathOutNorm = None):
    ###############################################################
    # Read the charUnigramData sub-directory and perform feature  #
    # vector analysis on specified text files. Afterwards, output #
    # to a text file.                                             #
    ###############################################################
    del FV_OUT_LIST[:] # Delete the list-list containing runtime raw vector data
    del NV_OUT_LIST[:] # Delete the list-list containing runtime normalized vector data
    files = sorted(glob.glob(fPathIn)) # File object of sub-directory path name (sorted)
    for name in files: 
        try:
            with open(name) as f: 
                vectorProcessFile(f)           
        except IOError as exc:
            if exc.errno != errno.EISDIR: # Do not fail if a directory is found, just ignore it.
                raise # Propagate other kinds of IOError.
    
    out_01 = open(fPathOutRaw, 'a') # Raw character unigram feature vector for specified dataset
    out_02 = open(fPathOutNorm, 'a') # Normalized character unigram feature vector for specified dataset
    out_01.seek(0) # Append to file and overwrite current contents
    out_02.seek(0) # Append to file and overwrite current contents
    out_01.truncate() # Truncates at the current position
    out_02.truncate() # Truncates at the current position 
    
    i = 0
    while i < len(FV_OUT_LIST):
        out_01.write(str(FV_OUT_LIST[i]) + '\n')
        out_02.write(str(NV_OUT_LIST[i]) + '\n')
        i += 1
                
    out_01.close()
    out_02.close()

    
'''
    Function: printStatistics(dataSetName = None)
    @summary: Outputs Dataset Averages for character, word, and sentence count.
    @param dataSetName:  A string Dataset name
    @warning: Must be called following outputFileDatasetResults or results will be erroneous. 
'''
def printStatistics(dataSetName = None):
    global FV_OUT_LIST 
    total_char_count = 0
    total_word_count = 0
    rough_sentence_count = 0
    
    for fv_list in FV_OUT_LIST:
        for number in fv_list:
            total_char_count += number
        total_word_count += fv_list[0]
        rough_sentence_count += fv_list[14]
        
    print(str(dataSetName) + ' Dataset Character Average: ' + str(total_char_count / len(FV_OUT_LIST)))
    print(str(dataSetName) + ' Dataset Word Average: ' + str(total_word_count / len(FV_OUT_LIST)))
    print(str(dataSetName) + ' Dataset Rough Sentence Average: ' + str(rough_sentence_count / len(FV_OUT_LIST)) + '\n\n')



'''
    FUNCTION: printCasisOutputSimilarity(fPathOriginOut = None)
    
    Description: Computes similarity between origin and newly computed CASIS-25 Output results. 
                 Prints Euclidean and Manhattan distance between both. Ideally, we want both 
                 distance values to be 0.
    
    RETURNS:
    
        Euclidean Distance: The Euclidean distance between two points is the length of the path connecting them.
    
        Manhattan Distance: Manhattan distance is a metric in which the distance between two points is the sum 
                            of the absolute differences of their Cartesian coordinates. In a simple way of 
                            saying it is the total sum of the difference between the x-coordinates and y-coordinates.
    
'''    
def printCasisOutputSimilarity(fPathOriginOut = None):
    global FV_OUT_LIST
    newer_char_list = []
    origin_char_list = []
    for fv_list in FV_OUT_LIST:
        for number in fv_list:
            newer_char_list.append(number)      
    try:
        with open(fPathOriginOut) as f:
            for line in f:
                currentline = line.split(',')
                for i in range(1, 96):
                    origin_char_list.append(float(currentline[i]))       
    except IOError as exc:
            if exc.errno != errno.EISDIR: # Do not fail if a directory is found, just ignore it.
                raise # Propagate other kinds of IOError.
    
    euclidean_dist = sqrt(sum(pow(a-b,2) for a, b in zip(origin_char_list, newer_char_list)))
    manhattan_dist = sum(abs(a-b) for a,b in zip(origin_char_list, newer_char_list))
    
    print('Euclidean Distance between Origin and New CASIS-25 Dataset Result: ' + str(euclidean_dist))
    print('Manhattan Distance between Origin and New CASIS-25 Dataset Result: ' + str(manhattan_dist) + '\n\n')
    ################### CONSOLE OUTPUT AS OF 9/18/18 @ 2348 Hours ###################
    #                                                                               #
    #     CASIS-25 Dataset Character Average: 1640.18                               #
    #     CASIS-25 Dataset Word Average: 274.5                                      #
    #     CASIS-25 Dataset Rough Sentence Average: 13.08                            #
    #                                                                               #
    #                                                                               #
    #     Euclidean Distance between Origin and New CASIS-25 Dataset Result: 2.0    #
    #     Manhattan Distance between Origin and New CASIS-25 Dataset Result: 4.0    #
    #                                                                               #
    #                                                                               #
    #     VANDERBILT Dataset Character Average: 3571.14814815                       #
    #     VANDERBILT Dataset Word Average: 611.740740741                            #
    #     VANDERBILT Dataset Rough Sentence Average: 39.2962962963                  #
    #################################################################################


## PERFORM FEATURE VECTOR ANALYSIS ON CASIS-25 Data ##   
outputFileDatasetResults(path_to_casis_data, path_to_casis_raw_output, path_to_casis_norm_output)
printStatistics('CASIS-25') # Console print statistics for CASIS Dataset
printCasisOutputSimilarity(path_to_casis_origin_fv) # Console prints CASIS Similarity between origin result and newly computed result

## PERFORM FEATURE VECTOR ANALYSIS ON VANDERBILT Data ##  
outputFileDatasetResults(path_to_vanderbilt_data, path_to_vanderbilt_raw_output, path_to_vanderbilt_norm_output)
printStatistics('VANDERBILT') # Console print statistics for VANDERBILT Dataset