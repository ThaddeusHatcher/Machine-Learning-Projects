#~/anaconda2/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import errno
import random
import numpy as np
from sklearn import svm
from scipy import sparse
import HTMLParser
from google.cloud import translate # Imports the Google Cloud client library
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_extraction.text import TfidfTransformer

# ========== GLOBAL VARS ============= #
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/donaldtran/Documents/Working_Projx_Dir/cred-cloud-tran-886a9251966a.json"
parser = HTMLParser.HTMLParser()

class AISv1:
    '''
        Author Identification System Version1
        
        https://console.developers.google.com/apis/api/translate.googleapis.com/overview?project=991438246010
    '''
    # ========== CLASS VARS ============= #
    cwd = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    dsd = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'DataSets')) 
    
    def __init__(self):
        pass
    
    def iterativeTranslateN(self, n, dSet, fName, outLabel = 9999):
        """    
            Performs n iterative language translations and then translates back to English
            @params:
                n         - Required  : The number of times to translate (Int)
                dSet      - Required  : The data set: 'casis', 'vanderbilt', or <None> if using Adversaries text (Str)
                fName     - Required  : The file name {excluding file extension} of the originating text (Str)
                outLabel  - Optional  : The output file name of the translated text (Str or Int)
        """    
        # Instantiates a client
        translate_client = translate.Client()
        randLangs = [] # The list containing sequence of translations
        if dSet == 'casis':
            absFile = os.path.join(self.dsd, "CASIS-25_Dataset/%s.txt" %(fName))
            outFile = os.path.join(self.dsd, "MaskedSamples/CASIS/%s_1.txt" %(outLabel))
        elif dSet == 'vanderbilt':
            absFile = os.path.join(self.dsd, "VANDERBILT_Dataset/%s.txt" %(fName))
            outFile = os.path.join(self.dsd, "MaskedSamples/VANDERBILT/%s.txt" %(outLabel))
        else:
            absFile = os.path.join(self.dsd, "ADVERSARIES/%s.txt" %(fName))
            outFile = os.path.join(self.dsd, "MaskedSamples/OUTGOING_MASKED_TEXT/%s.txt" %(outLabel))
        try:
            # Read in the originating text from file to 2D list
            targetTxt = []
            with open(absFile, "rb") as txtFile:
                for line in txtFile:
                    targetTxt.append(line.decode(errors='replace'))             
            # Iterative Loop Translation
            k = 0
            while k < n:
                tmpTarget = []
                lang = self.getRandLangDest()
                randLangs.append(lang)
                for txt in targetTxt:
                    translation = translate_client.translate(txt, target_language=lang)
                    tmpTarget.append(translation['translatedText'])                      
                targetTxt = tmpTarget[:] 
                k += 1
            # Converting back to English after n-iterative translations
            tmpTarget = []
            for txt in targetTxt:
                translation = translate_client.translate(txt, target_language='en')
                tmpTarget.append(translation['translatedText']) 
            targetTxt = tmpTarget[:] 
            randLangs.append('en') 
            print('\nSequence of Translations: {}\n'.format(randLangs))
            
            # PRINT TO CONSOLE
            for txt in targetTxt:
                print('{}'.format(txt.encode('utf-8')))
                   
            # Output to text file in MaskedSamples directory
            #with open(outFile, 'w') as outfile:
            #    for txt in targetTxt:
            #        outfile.write('{}\n'.format(parser.unescape(txt.encode('utf-8')))
        except IOError as exc:
            if exc.errno != errno.EISDIR: # Do not fail if a directory is found, just ignore it.
                raise # Propagate other kinds of IOError. 
    
    def getRandLangDest(self):
        """    
        Retrieves a randomly selected language.
        """  
        LANGUAGES = {
            'af': 'afrikaans','sq': 'albanian','am': 'amharic',
            'ar': 'arabic','hy': 'armenian','az': 'azerbaijani',
            'eu': 'basque','be': 'belarusian','bn': 'bengali',
            'bs': 'bosnian','bg': 'bulgarian','ca': 'catalan',
            'ceb': 'cebuano','ny': 'chichewa','zh-cn': 'chinese (simplified)',
            'zh-tw': 'chinese (traditional)','co': 'corsican','hr': 'croatian',
            'cs': 'czech','da': 'danish','nl': 'dutch',
            'eo': 'esperanto','et': 'estonian','tl': 'filipino',
            'fi': 'finnish','fr': 'french','fy': 'frisian',
            'gl': 'galician','ka': 'georgian','de': 'german',
            'el': 'greek','gu': 'gujarati','ht': 'haitian creole',
            'ha': 'hausa','haw': 'hawaiian','iw': 'hebrew',
            'hi': 'hindi','hmn': 'hmong','hu': 'hungarian',
            'is': 'icelandic','ig': 'igbo','id': 'indonesian',
            'ga': 'irish','it': 'italian','ja': 'japanese',
            'jw': 'javanese','kn': 'kannada','kk': 'kazakh',
            'km': 'khmer','ko': 'korean','ku': 'kurdish (kurmanji)',
            'ky': 'kyrgyz','lo': 'lao','la': 'latin',
            'lv': 'latvian','lt': 'lithuanian','lb': 'luxembourgish',
            'mk': 'macedonian','mg': 'malagasy','ms': 'malay',
            'ml': 'malayalam','mt': 'maltese','mi': 'maori',
            'mr': 'marathi','mn': 'mongolian','my': 'myanmar (burmese)',
            'ne': 'nepali','no': 'norwegian','ps': 'pashto',
            'fa': 'persian','pl': 'polish','pt': 'portuguese',
            'pa': 'punjabi','ro': 'romanian','ru': 'russian',
            'sm': 'samoan','gd': 'scots gaelic','sr': 'serbian',
            'st': 'sesotho','sn': 'shona','sd': 'sindhi',
            'si': 'sinhala','sk': 'slovak','sl': 'slovenian',
            'so': 'somali','es': 'spanish','su': 'sundanese',
            'sw': 'swahili','sv': 'swedish','tg': 'tajik',
            'ta': 'tamil','te': 'telugu','th': 'thai',
            'tr': 'turkish','uk': 'ukrainian','ur': 'urdu',
            'uz': 'uzbek','vi': 'vietnamese','cy': 'welsh',
            'xh': 'xhosa','yi': 'yiddish','yo': 'yoruba',
            'zu': 'zulu','fil': 'Filipino','he': 'Hebrew'
        }
        return random.choice(list(LANGUAGES))
        
    def getVandyMaskedCU(self, teamDir, readFromTeams=True):
        if readFromTeams:
            absFile = os.path.join(self.dsd, "MaskedSamples/%s/vanderbilt.txt" %(teamDir))
        else:
            absFile = os.path.join(self.dsd, "MaskedSamples/VANDERBILT/%s.txt" %(teamDir))
        try:
            with open(absFile, "r") as f: 
                feature_dict = {} 
                feature_vector = [] 
                for i in range(32,127):
                    feature_dict[chr(i)] = 0.0
                for letter in f.read():
                    if letter in feature_dict:
                        feature_dict[letter] += 1.0
                for key in sorted(feature_dict.keys()):
                    feature_vector.append(feature_dict[key])
                return feature_vector
        except IOError as exc:
            if exc.errno != errno.EISDIR: # Do not fail if a directory is found, just ignore it.
                raise # Propagate other kinds of IOError.  
            
    def getCASISMaskedCU(self, authLabel):
        '''
            authLabel: 1020, 1021, 1015
        ''' 
        absFile = os.path.join(self.dsd, "MaskedSamples/CASIS/%s_1.txt" %(authLabel))
        try:
            with open(absFile, "r") as f: 
                feature_dict = {} 
                feature_vector = [] 
                for i in range(32,127):
                    feature_dict[chr(i)] = 0.0
                for letter in f.read():
                    if letter in feature_dict:
                        feature_dict[letter] += 1.0
                for key in sorted(feature_dict.keys()):
                    feature_vector.append(feature_dict[key])
                return feature_vector
        except IOError as exc:
            if exc.errno != errno.EISDIR: # Do not fail if a directory is found, just ignore it.
                raise # Propagate other kinds of IOError. 
        
    def Get_Casis_CUDataset(self):
        """
        Reads the "CASIS-25_CU.txt" file in the CWD and creates two 2D
        numpy vector arrays X and Y split into Training Data and Training Classifiers,
        respectively.
        """    
        X = []
        Y = []
        with open(os.path.join(self.cwd,"CASIS-25_CU.txt"), "r") as feature_file: 
            for line in feature_file:
                line = line.strip().split(",")
                Y.append(line[0][:4])
                X.append([float(x) for x in line[1:]])  
        return np.array(X), np.array(Y)

    def Get_Vanderbilt_CUDataset(self):
        """
        Reads the "VANDERBILT-9_CU.csv" file in the CWD and creates two 2D
        numpy vector arrays X and Y split into Training Data and Training Classifiers,
        respectively.
        """        
        X = []
        Y = []
        with open(os.path.join(self.cwd, "VANDERBILT-9_CU.csv"), "r") as feature_file: 
            for line in feature_file:
                line = line.strip().split(",")
                Y.append(line[0][:4])
                X.append([float(x) for x in line[1:]])
        return np.array(X), np.array(Y)
        
    def getAuthLabel(self, datasetOpt, feat_vect): 
        if datasetOpt == 'vanderbilt':
            CU_X, Y = self.Get_Vanderbilt_CUDataset() 
            # Optimal feature mask for Vanderbilt dataset -- Accuracy = 86.11%
            OPTIMAL_FM = [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 
                          0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 
                          0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 
                          0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 
                          0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0]
        elif datasetOpt == 'casis':
            CU_X, Y = self.Get_Casis_CUDataset() 
            # Optimal feature mask for Casis-25 dataset -- Accuracy = 87%
            OPTIMAL_FM = [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 
                          1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 
                          1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 
                          0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 
                          1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0]
        
        else:
            print('Dataset \"datasetOpt\" not specified. System exiting.')
            sys.exit()
        
        # Performs feature masking
        for i in range(len(CU_X)):
            for j in range(95):
                if OPTIMAL_FM[j] == 0:
                    CU_X[i][j] = 0.0 
        # Mask the incoming feature vector being evaluated            
        for k in range(95):
            if OPTIMAL_FM[k] == 0:
                feat_vect[k] = 0.0 
        # Linear Support Vector Classification
        lsvm = svm.LinearSVC()
        
        # Preprocessing initializers
        scaler = StandardScaler()
        tfidf = TfidfTransformer(norm = None)
        dense = DenseTransformer()
        train_labels = Y
    
        # tf-idf (term frequency-inverse document frequency)
        tfidf.fit(CU_X)
        CU_train_data = CU_X
        CU_eval_data = dense.transform(sparse.hstack(feat_vect)) 
        
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
        
        # CONFIDENCE SCORES FOR EACH OF THE AUTHORS
        confidence_scores = lsvm.decision_function(eval_data)[0]
        print('{}\n'.format(confidence_scores))
        return list(confidence_scores).index(max(confidence_scores))
        
    def getVandyAuthorAt(self, keyID):
        options = {
            0: 'Allen',
            1: 'D\'Andrea',
            2: 'Stephenson',
            3: 'Walker',
            4: 'Fiutak',
            5: 'Boclair',
            6: 'Sparks',
            7: 'Marlin',
            8: 'Goodfriend'
        }
        return options.get(keyID)
        

class DenseTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray()
    
#===============================================================#
#================== BEGIN MAIN RUNNING SCRIPT ==================#
#===============================================================#
a1 = AISv1() # Class Instance

#===============================================================================#
#========== SELF-ADVERSARIAL ===================================================#
#===============================================================================#
# PUTS OUTPUT OF TRANSLATED TEXT IN EITHER /MaskedSamples/VANDERBILT or CASIS 
# USING AnchorOfGold_Allen_VanderbiltvsFlo_wk7 For Vanderbilt
# USING 1015_1.txt for CASIS

#==================================================================#
#    Iterative Language Translation (ILT)                          #
#    READING FROM: /DataSets/CASIS-25_Dataset/1015_1.txt           #
#    TRANSLATING TO: /DataSets/MaskedSamples/CASIS/1015_1.txt/     #
#==================================================================#
# a1.iterativeTranslateN(11, 'casis', '1015_1', 1015)
#casMaskedCU15 = a1.getCASISMaskedCU("1015")
#casLabel15 = a1.getAuthLabel('casis', casMaskedCU15)
#print("SELF TESTING: \nThe Mask for CASIS-25 Author 15 after masking file 1015_1.txt classifed >> Author {}\n".format(casLabel15))

#=================================#
#    Paraphrasing Tool Sample     #
#=================================#
#casMaskedCU20 = a1.getCASISMaskedCU("1020")
#casLabel20 = a1.getAuthLabel('casis', casMaskedCU20)
#print("SELF TESTING: \nThe Mask for CASIS-25 Author 20 after masking file 1020_1.txt classifed >> Author {}\n".format(casLabel20))

#============================================================================================================#
# === READING FROM: /DataSets/VANDERBILT_Dataset/AnchorOfGold_Allen_VanderbiltvsFlo_wk7.txt                  #
# === TRANSLATING TO: /DataSets/MaskedSamples/VANDERBILT/AnchorOfGold_Allen_VanderbiltvsFlo_wk7_TRANSLATED/  #
#============================================================================================================#
#a1.iterativeTranslateN(15, 'vanderbilt', 'AnchorOfGold_Allen_VanderbiltvsFlo_wk7', 'AnchorOfGold_Allen_VanderbiltvsFlo_wk7_TRANSLATED')
#vanMaskedCU0 = a1.getVandyMaskedCU('AnchorOfGold_Allen_VanderbiltvsFlo_wk7_TRANSLATED', False)
#vanLabel0 = a1.getAuthLabel('vanderbilt', vanMaskedCU0)
#print("SELF TESTING: \nThe Mask for Vandy Author 0 after masking file AnchorOfGold_Allen_VanderbiltvsFlo_wk7_TRANSLATED.txt classifed >> Author {} {}\n".format(vanLabel0, a1.getVandyAuthorAt(vanLabel0)))

#====================================================#
#============ AISv1 RESULTS OUTGOING ================#
#====================================================#
#a1.iterativeTranslateN(8, None, 'TennTeam10', 'TennTeam10_Out_2') # xRef: /MaskedSamples/OUTGOING_MASKED_TEXT/<file>
#a1.iterativeTranslateN(20, None, 'MissTeam5', 'MissTeam5_Out') # xRef: /MaskedSamples/OUTGOING_MASKED_TEXT/<file>
#a1.iterativeTranslateN(5, None, 'KentuckyTeam8', 'KentuckyTeam8_Out') # xRef: /MaskedSamples/OUTGOING_MASKED_TEXT/<file>
#a1.iterativeTranslateN(7, None, 'AuburnTeam4', 'AuburnTeam4_Out') # xRef: /MaskedSamples/OUTGOING_MASKED_TEXT/<file>
#a1.iterativeTranslateN(7, None, 'SouthCarolinaTeam11', 'SouthCarolinaTeam11_Out')
#a1.iterativeTranslateN(15, None, 'LouisianaTeam2', 'LouisianaTeam2_Out')
#a1.iterativeTranslateN(7, None, 'AlabamaTeam1', 'AlabamaTeam1_Out')

#====================================================#
#========== AISv1 RESULTS FROM INCOMING==============#
#====================================================#
# TEAM 8's (MISSISSIPPI) Masking of Allen #
maskedFV5 = a1.getVandyMaskedCU("TEAM5", True)
label5 = a1.getAuthLabel('vanderbilt', maskedFV5)
print("Team 5's Mask of Vandy Author 0: Allen classifies to Author {}: {}\n".format(label5, a1.getVandyAuthorAt(label5)))

# TEAM 8's (KENTUCKY) Masking of Allen #
maskedFV8 = a1.getVandyMaskedCU("TEAM8", True)
label8 = a1.getAuthLabel('vanderbilt', maskedFV8)
print("Team 8's Mask of Vandy Author 0: Allen classifies to Author {}: {}\n".format(label8, a1.getVandyAuthorAt(label8)))

# TEAM 10's (TENNESSEE) Masking of Allen #
maskedFV10 = a1.getVandyMaskedCU("TEAM10", True)
label10 = a1.getAuthLabel('vanderbilt', maskedFV10)
print("Team 10's Mask of Vandy Author 0: Allen classifies to Author {}: {}\n".format(label10, a1.getVandyAuthorAt(label10)))










