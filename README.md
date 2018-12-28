# Machine Learning Projects (Implemented in Python)

This repository is composed of my contributions to a series of group projects for my Computational Intelligence and Adversarial Machine Learning class taken Fall 2018 at Auburn University. A link to the full repository will be provided in the event that it is made public. 

## Assignment 1: Data Collection & Feature Extraction

### Data Collection

We were to use 2 datasets for training and testing our ML algorithms. One being static and the other dynamic. 


#### Static Dataset

Our static dataset was the CASIS-25 dataset, which consisted of 25 authors with 4 samples per author totaling at 100 samples. This dataset was provided to us at the beginning of the semester to serve as a reliable means of benchmarking our algorithms. 


#### Dynamic Dataset

Each group was assigned a SEC football team from which they will find a series of sportswriters that publish articles on that team on a weekly basis. A single article from each sportswriter is collected every week. My team was assigned to Vanderbilt. 

Our dynamic dataset consisted of articles published by Vanderbilt sportswriters. A single sample was collected from each sportswriter each week for the entirety of the football season, the subject of which being the Vanderbilt football game from that particular week. We were able to find 9 sportswriters that would reliably publish relevant articles on a weekly basis, thus resulting in an increase of the dataset every week by 9 samples. 

## Feature Extraction

The feature extractor in use for the entirety of the semester was a 95-feature set character unigram extractor. The character unigram feature extractor steps through each author sample and counts the number of occurrences of individual characters, including the alphabet, digits 0-9, and special characters, totaling in a feature vector size of 95 for each sample. 

#### Alternative Approach

An alternative approach to feature extraction, a word feature extractor, was considered in the last phase of projects as a countermeasure to identifying masked author samples from our dataset. The word feature extractor was implemented using python's Natural Language Toolkit (NLTK). It is fully functional and can be found with its relevant source code [here](https://github.com/ThaddeusHatcher/Machine-Learning-Projects/tree/master/Authorship-Identification-Systems/World-Feature_Extraction). We eventually determined it to be an insufficient means of feature extraction due to the inevitably massive increase in feature vector size, which would have been upwards of 230,000 features (as opposed to 95), and the effects of such in performance when processing the dataset. The reason for the substantial feature vector size is primarily due to the way the extractor was developed/implemented, which was by using an NLTK general corpus to serve as the feature dictionary that was utilized to count word occurrences. So, feature vector size was simply the size of the corpus, that being upwards of 230,000 words. Additionally, words found in any sample in the dataset that were not already in the corpus were appended to the feature dictionary, causing feature vector size to increase by 1 each time a new word could not be found in the corpus.

## Assignment 2: Implementing an Instance-Based Learner for Authorship Attribution

For this assignment we were to implement a series of Instance-based and Kernel-based learners to perform authorship attribution using each of our 2 datasets. My role was to implement the Instance-based K-Nearest Neighbor Distance Weighted (KNN-DW) Machine Learner. 

### Testing 

Testing was performed using k-fold Cross Validation were the number of folds may be specified in a function call. The Learner is tested for k values of 1, 3, 5, and n, where n = total number of samples in the training set.

### Optimization

After implementing the Machine Learner we were to develop and implement an optimization that would increase its performance. The goal of mine was to mitigate the degree of influence that outlier samples have on target classification in the event that one ends up being extremely close in distance to the target. 

An outlier sample is simply an author sample that is more similar to another author in the dataset than it is to the true author of said sample. More abstractly, it is a case in which one of the authors in the dataset has a writing sample that more closely resembles the writing style of another author that it does their own. 
