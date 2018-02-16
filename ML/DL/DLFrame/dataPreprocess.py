#######################################################################
'''
dataPreprocessing code

You can implement your own data here
'''



import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from deeplearning_frame import *

digits = datasets.load_digits()
images_and_labels = list(zip(digits.images, digits.target))

def train_test_genetate(images_and_labels, split_rate = 0.7):
    """
    generate training set and test set
    """
    training_set = [];
    test_set = []
    training_label = []
    test_label = []
    test_label2 = []

    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder()
    enc.fit([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]])
    enc.transform([[0]]).toarray().T
    
    def normalize(vector):
        vector = vector - vector.mean()
        vector_norm = vector / np.var(vector)
        return vector
    
    def image2vector(image):
        data = np.asarray(image, dtype = "int32") #convert image to np array
        v = image.reshape((image.shape[0] * image.shape[1] , 1)) #convert array to 1D
        v = normalize(v)
        return v

    def combine(vector1, vector2):
        if vector1 == []:
            return vector2
        else:
            return np.concatenate((vector1, vector2), axis = 1)

    for image, label in images_and_labels:

        vector = image2vector(image)
        if (np.random.rand() < split_rate):                     #split the raw dataset into training set and test set
            training_set = combine(training_set, vector)
            new_label = enc.transform([[label]]).toarray().T
            training_label = combine(training_label, new_label)

        else:
            test_set = combine(test_set, vector)
            new_label = enc.transform([[label]]).toarray().T
            test_label = combine(test_label, new_label)
            test_label2.append(label)
            
    return training_set, test_set, training_label, test_label, test_label2
