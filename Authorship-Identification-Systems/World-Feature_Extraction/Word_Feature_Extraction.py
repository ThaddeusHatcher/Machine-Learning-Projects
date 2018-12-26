import os
import csv
import sys
import glob
import sklearn 
import errno
import nltk
import codecs
import itertools
import nltk.tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import words
from ftfy import fix_file
from sklearn.feature_extraction.text import CountVectorizer

dsd = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'DataSets')) 

'''
    ONLY WORKS FOR VANDERBILT AT THE MOMENT.
    WILL BE ALTERED TO WORK FOR BOTH DATASETS. 
'''
def extractWordList():
    feature_vects = []
    file_vectorizers = []
    del feature_vects[:] 
    words = []
    files = sorted(glob.glob(os.path.join(dsd, "VANDERBILT_Dataset/*.txt"))) 
    try:
        for file in files: 
            with open(file, encoding="ISO-8859-1") as f: 
                feature_dict = {} 
                feature_vector = [] 
                vectorizer = CountVectorizer(input=f)
                X = vectorizer.fit_transform(f)
                Y = vectorizer.get_feature_names()
                file_vectorizers.append(Y)
                for word in Y:
                    words.append(word)
                    print(word)

        '''
                for i in range(32,127):
                    feature_dict[chr(i)] = 0.0
                for letter in f.read():
                    if letter in feature_dict:
                        feature_dict[letter] += 1.0
                for key in sorted(feature_dict.keys()):
                    feature_vector.append(feature_dict[key])
                feature_vects.append(feature_vector)
        '''
        with open('VANDERBILT_Words.txt', 'w') as outfile:
            for word in words:  
                outfile.write(word + "\n")

        with open('VANDERBILT_Words.txt', encoding="ISO-8859-1") as f:
            word_list = []
            vectorizer = CountVectorizer(input=f)
            X = vectorizer.fit_transform(f)
            Y = vectorizer.get_feature_names()
            for word in Y:
                word_list.append(word)
            return word_list, file_vectorizers
        

        
    except IOError as exc:
        if exc.errno != errno.EISDIR: # Do not fail if a directory is found, just ignore it.
            raise # Propagate other kinds of IOError.

def extractFileFeatures():
    files = sorted(glob.glob(os.path.join(dsd, "VANDERBILT_Dataset/*.txt")))
    word_list = words.words()
    feature_vects = []
    labels = []
    feature_dict = {}
    try:
        for file in files:
            with codecs.open(file, encoding="utf-8") as f:
                n = [0.0 for i in range(0, len(word_list))]
                feature_dict = dict(zip(word_list, n))
                fix_file(f)
                raw = f.read()
                tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
                tokens = tokenizer.tokenize(raw)
                for token in tokens:
                    if token in feature_dict:
                        feature_dict[token] += 1.0
                feature_vector = [feature_dict[k] for k in feature_dict]
                feature_vects.append(feature_vector)

        with open('VANDERBILT_Words.txt', 'w') as outfile:
            label = 1000
            sampleCount = 0
            fvs = []
            for fv in feature_vects:
                if sampleCount == 9:
                    label += 1
                    sampleCount = 0
                fv.insert(0, label)
                i = 0
                for x in fv:
                    if (i == len(fv) - 1):
                        outfile.write("%s" %x)
                    else:
                        outfile.write("%s," %x)
                        i += 1
                outfile.write("\n")
                sampleCount += 1   

    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

    


extractFileFeatures()
#word_list, file_vectorizers = extractWordList(9)
print("Done")
