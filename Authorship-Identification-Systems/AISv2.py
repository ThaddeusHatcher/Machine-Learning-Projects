#!/anaconda2/bin/python
# -*- coding: utf-8 -*-

import os
import errno
import EDA
import Data_Utils as utils
import nltk 
from nltk.corpus import wordnet 

#####################################################################################
# Copy and paste masked sample text into the 'target.txt' file to run AISv2.        #
# If testing masks for a different author, change value of self.author in __init__  #
# to the true author of the sample being masked.                                    #        
#####################################################################################

'''
    Class: AISv1 (Authorship Identification System version 1)
    Components: 
        - LSVM classifier
        - 95 Feature-Set Character Unigram feature extractor
        - Optimal feature bit-mask 
'''
class AISv2:

    def __init__(self, dataset_select):
        self.dataset_select = dataset_select
        self.target_fv = []
        self.author = '1000'    # the known author of the masked sample

    def extractTargetFV(self):
        feature_dict = {}
        feature_vector = []
        fPath = os.path.join(os.getcwd(), 'Authorship-Identification-Systems','target.txt')
        try:
            with open(fPath, encoding="utf-8") as f:
                for i in range(32, 127):
                    feature_dict[chr(i)] = 0.0
            
                for letter in f.read():
                    if letter in feature_dict:
                        feature_dict[letter] += 1.0
                
                for key in sorted(feature_dict.keys()):
                    feature_vector.append(feature_dict[key])

                self.target_fv = feature_vector
        except IOError as exc:
            if exc.errno != errno.EISDIR: # Do not fail if a directory is found, just ignore it.
                raise # Propagate other kinds of IOError.

    def classify(self):
        self.extractTargetFV()

        dfs = []

        # Pass the target feature vector of the masked sample to the modified EDA 
        # to find the best mask for proper author identification
        eda = EDA.EDA(self.target_fv)
        target_fm, avg_acc = eda.runOptimizedEvolution(5000, 0.07)
        for i in range(0, 95):
            if (target_fm[i] == 0.0):
                self.target_fv[i] = 0.0
        identified, df = utils.getLSVMClassification(self.dataset_select, self.target_fv)
        dfs = utils.getAuthorSampleDFs(self.dataset_select, identified)
        
        print('Target Decision Function: ')
        print(df)
        
        if (self.author == identified):
            print('AISv2 Success')
        else:
            print('AISv2 Failure')
            print('Classified as %s' %identified)
        

identifier = AISv2('vanderbilt')
identifier.classify()