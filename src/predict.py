import os
import datetime
import resnet
import data
import json
from resnet import tail_model
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import wrappers
from tensorflow.keras.wrappers import scikit_learn
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import keras
from keras.utils.vis_utils import plot_model


from keras import backend as K


from save import *
import getpass
import time

if __name__ == "__main__" :
    jsons_test = [] 
    with open("../data/test.json") as fil :
        for i in fil :
            jsons_test.append(json.loads(i))

    test = data.mat_input_test(jsons_test)

    model = keras.models.load_model("../results/modeles/2020-10-25-21-36-39-neo-janus/pre_act_mod/_1e-06_.model")
    chars = ["\\","|","/","-"]
    c=0
    with open("../results/results.csv","w") as fillout:
        fillout.write("id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C\n")

        for i in test:
            print(chars[c%4],end="\r")
            c +=1
            pred1 = model.predict(x = [test[i]["pos"], test[i]["mat"]])
            start2 = 68 - (test[i]["len"]-68)
            pred = numpy.concatenate((pred1[0], pred1[1][start2:]))
            for j,val in enumerate(pred):
                fillout.write("{}_{},{},{},{},{},{}\n".format(i,j,
                                                            val[0],
                                                            val[1],
                                                            val[2],
                                                            val[3],
                                                            val[4]))
            
