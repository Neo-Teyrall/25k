
###############################################################################
################################ import   #####################################
###############################################################################
import copy
import data
import resnet
import numpy
import os
import json

import datetime
import time

import simple
import janus

import matplotlib
from matplotlib import pyplot

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import wrappers
from tensorflow.keras.wrappers import scikit_learn
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import keras
from keras.utils.vis_utils import plot_model

import sklearn
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight

###############################################################################
################################ Reseaux  #####################################
###############################################################################




if __name__ == "__main__" :
    jsons_train = []

    with open("../data/train.json") as fil :
        for i in fil :
            a = json.loads(i)
            if a["SN_filter"] == 0:
                continue
            jsons_train.append(a)
 
            jsons_test = []

    with open("../data/test.json") as fil :
        for i in fil :
            jsons_test.append(json.loads(i))


    mat_intput_train,mat_input2 = data.mat_input(jsons_train)

    Y_train = data.mat_output(jsons_train)

    l_rates = [1e-1,1e-2,1e-3,
               1e-4,1e-5,1e-6,
               1e-7,1e-8,1e-9]

    list_resnet = [resnet.classic,
                   resnet.original,
                   resnet.pre_act,
                   resnet.pre_act_mod]


    list_resnet = [resnet.classic,
                   resnet.pre_act_mod]

    simple.compare_model(mat_input = mat_intput_train,
   

                  mat_output = Y_train,
                  list_resnet = list_resnet,
                  l_rates = l_rates,
                  nb_resnet = 30 ,
                  epochs = 50,
                  verbose = 0,sleep = 60)

    janus.compare_model_janus(mat_input = (mat_intput_train, mat_input2),
                        mat_output = Y_train,
                        list_resnet = list_resnet,
                        l_rates = l_rates,
                        nb_resnet = [20,20,20] ,
                        epochs = 50,sleep = 60*3,verbose = 0)

