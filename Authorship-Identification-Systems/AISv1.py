#!/anaconda2/bin/python
# -*- coding: utf-8 -*-

import os
import errno
import Data_Utils as utils

OPTIMAL_FM_VANDERBILT = [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 
                         0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 
                         0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 
                         0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 
                         0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0]

OPTIMAL_FM_CASIS = [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 
                    1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 
                    1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 
                    0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 
                    1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0]

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
class AISv1:

    def __init__(self, dataset_select):
        self.dataset_select = dataset_select
        self.target_fv = []
        self.author = '1000'

    def extractFV(self):
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

    def applyOptimalFM(self):
        fm = []
        if (self.dataset_select == 'casis'):
            fm = OPTIMAL_FM_CASIS
        else:
            fm = OPTIMAL_FM_VANDERBILT

        for i in range(0, len(fm)):
            if (fm[i] == 0.0):
                self.target_fv[i] = 0.0

    def classify(self):
        self.extractFV()
        self.applyOptimalFM()
        
        identified, decision_function = utils.getLSVMClassification(self.dataset_select, self.target_fv)
        if (self.author == identified):
            print('AISv1 Success')
        else:
            print('AISv1 Failure')
            print('Classified as %s' %identified)

identifier = AISv1('vanderbilt')
identifier.classify()