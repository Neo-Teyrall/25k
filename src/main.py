import os
import json
import pandas
import numpy
import copy
import math
import statistics
import datetime

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

#TF
###############################################################################
####################         Functions                     ####################
###############################################################################

def one_hot_AA_seq(seq: str):
    out = []
    keys = {"G": [1,0,0,0],
            "C": [0,1,0,0],
            "A": [0,0,1,0],
            "U": [0,0,0,1]}
    for i in seq:
        out.append(copy.deepcopy(keys[i]))
    return numpy.matrix(out)

def one_hot_struct_seq(seq: str):
    out = []
    keys = {"S": [1,0,0,0,0,0,0],
            "M": [0,1,0,0,0,0,0],
            "I": [0,0,1,0,0,0,0],
            "B": [0,0,0,1,0,0,0],
            "H": [0,0,0,0,1,0,0],
            "E": [0,0,0,0,0,1,0],
            "X": [0,0,0,0,0,0,1]}
    for i in seq :
        out.append(copy.deepcopy(keys[i]))
    return numpy.matrix(out)

def mat_struct(seq : str):
    taille = len(seq)
    out = []
    line = [0]*taille

    for i in range(taille):
        out.append(copy.deepcopy(line))

    for i in range(len(seq)):
        if seq[i] == "(" :
            delta = 0
            for j in range(len(seq)-i):
                if seq[i+j] == "(" :
                    delta + 1
                if seq[i+j] == ")" :
                    if delta != 0 :
                        delta -= 1
                    else:
                        out[i][i+j] = 1
                        out[i+j][i] = 1
                        break
    return numpy.matrix(out)

def extract_key(clef : str, jsons) -> pandas.DataFrame:
    Y = []
    for i, val  in enumerate(jsons):
        Y.append(val[clef])
    return Y

def extract_Y(jsons):
    Y = pandas.concat([pandas.DataFrame(extract_key("deg_50C",jsons)),
                       pandas.DataFrame(extract_key("deg_Mg_50C",jsons)),
                       pandas.DataFrame(extract_key("deg_pH10",jsons)),
                       pandas.DataFrame(extract_key("deg_Mg_pH10",jsons)),
                       pandas.DataFrame(extract_key("reactivity",jsons))],axis = 1)

    return Y

def one_hot_structure(seq : str):
    out = []
    key = {"." : [1,0,0],
           "(" : [0,1,0],
           ")" : [0,0,1]}
    for i in seq:
        out.append(key[i])
    return numpy.matrix(out)

def extract_X2(jsons):
    X2 = pandas.concat([pandas.DataFrame(normer(extract_key("signal_to_noise",jsons))),
                        pandas.DataFrame(normer(extract_key("SN_filter",jsons))),
                        pandas.DataFrame(normer(extract_key("seq_length",jsons))),
                        pandas.DataFrame(normer(extract_key("seq_scored",jsons))),
                        pandas.DataFrame(normer(extract_key("reactivity_error",jsons))),
                        pandas.DataFrame(normer(extract_key("deg_error_Mg_pH10",jsons))),
                        pandas.DataFrame(normer(extract_key("deg_error_pH10",jsons))),
                        pandas.DataFrame(normer(extract_key("deg_error_Mg_50C",jsons))),
                        pandas.DataFrame(normer(extract_key("deg_error_50C",jsons)))],
                       axis=1)
    return X2

def normer(vec):
    if isinstance(vec[0],list):
        out = []
        for i in vec:
            out.append(norm(i))
        return(out)
    else :
        return norm(vec)

def norm(vec):
    #print(vec[:5],"\n")
    m = statistics.mean(vec)
    std = statistics.stdev(vec)
    for i in range(len(vec)):
        if std != 0:
            vec[i] = (vec[i]-m)/std
        else:
            vec[i]=1
    return vec

def mat_input(jsons):
    data_out  = []
    for i in jsons :
        ohaa  = one_hot_AA_seq(i["sequence"])
        ohsty = one_hot_struct_seq(i["predicted_loop_type"])
        ohst = one_hot_structure(i["structure"])
        data = numpy.concatenate((ohaa ,ohsty , ohst), axis= 1)
        data_out.append(data)
    mat_out = numpy.array(data_out)
    return(mat_out)


def mat_output(jsons):
    data_out = []
    for i in jsons :
        reactivity = i["reactivity"]
        deg_Mg_pH10 = i["deg_Mg_pH10"]
        deg_pH10 = i["deg_pH10"]
        deg_Mg_50C = i["deg_Mg_50C"]
        deg_50C = i["deg_50C"]
        out = []
        for j in range(len(reactivity)):
             out.append([reactivity[j],
                         deg_Mg_pH10[j],
                         deg_pH10[j],
                         deg_Mg_50C[j],
                         deg_50C[j]])
        data_out.append(out)
    data_out = numpy.array(data_out)
    return data_out

###############################################################################
####################         load data                     ####################
###############################################################################

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


mat_intput_train = mat_input(jsons_train)

for i in jsons_train:
    if len(i["structure"]) != 107:
        print("error")
Y_train = extract_Y(jsons_train)

Y_train = mat_output(jsons_train)

# to predict
    # 'reactivity',
    # 'deg_Mg_pH10',
    # 'deg_pH10',
    # 'deg_Mg_50C',
    # 'deg_50C'


## creation du modele


# def resnet(modelI):
#     model = layers.Dense(units = 1024,
#                             activation= "relu")(modelI)

#     model = layers.Dense(units = 1024,
#                             activation= "relu")(model)
#     model = layers.Add()([model, modelI])
#     return(model)


def conv_resnet(modelI):
    model = layers.Conv1D(filters= 30,
                          kernel_size=(3,),
                          activation= "relu",
                          padding="same")(modelI)

    model = layers.Conv1D(filters= 30,
                          kernel_size=(3,),
                          activation= "relu",
                          padding="same")(model)
    model = layers.Add()([model, modelI])
    return(model)


def tail_model(model1):
    model1 = layers.Conv1D(filters= 30,
                           kernel_size=(14,),
                           activation="relu")(model1)
    model1 = layers.Conv1D(filters= 30,
                           kernel_size=(14,),
                           activation="relu")(model1)
    model1 = layers.Conv1D(filters= 5,
                           kernel_size=(14,),
                           activation="softmax")(model1)
    return model1

def res_original(model):
    model  = layers.Conv1D(filters = 30,
                           kernel_size=(3,),
                           activation = "relu",
                           padding="same")(model)
    model = layers.BatchNormalization()(model)
    model = layers.Activation("relu")(model)
    model = layers.Conv1D(filters = 30,
                           kernel_size=(3,),
                           activation = "relu",
                           padding="same")(model)
    model = layers.BatchNormalization()(model)
    return model

def res_pre_act(model):
    model = layers.BatchNormalization()(model)
    model = layers.Activation("relu")(model)
    model = layers.Conv1D(filters = 30,
                           kernel_size=(3,),
                           activation = "relu",
                           padding="same")(model)
    model = layers.BatchNormalization()(model)
    model = layers.Activation("relu")(model)
    model = layers.Conv1D(filters = 30,
                           kernel_size=(3,),
                           activation = "relu",
                           padding="same")(model)
    return model


def creat_model():
    i = layers.Input(shape =(107,14))
    model1 = layers.Conv1D(filters= 30,
                           kernel_size=(3,),
                           activation = "relu",
                           padding="same")(i)
    model1 = conv_resnet(model1)
    model1 = layers.Conv1D(filters= 30,
                             kernel_size=(2,),
                             activation="relu",
                           padding="same")(model1)
    model1 = conv_resnet(model1)
    model1 = layers.Conv1D(filters= 30,
                             kernel_size=(2,),
                             activation="relu",
                           padding="same")(model1)
    model1 = layers.Conv1D(filters= 30,
                           kernel_size=(2,),
                           activation="relu",
                           padding="same")(model1)
    #
    model1 = conv_resnet(model1)
    #
    model1 = tail_model(model1)
    #
    model1 = tf.keras.Model(inputs=i, outputs=model1)
    model1.compile(optimizer="rmsprop",
                    loss = "mse")
    return model1

    # model1 = layers.Flatten()(model1)

    # model1 = layers.Dense(units = 1024,
    #                       activation= "relu")(model1)
    # model1 = resnet(model1)
    # model1 = layers.Dense(units = 1024,
    #                       activation= "relu")(model1)
    # model1 = resnet(model1)
    # model1 = layers.Dense(units = 1024,
    #                       activation= "relu")(model1)
    # model1 = resnet(model1)
    # model1 = layers.Dense(units = 1024,
    #                       activation= "relu")(model1)



    # model1 = layers.Dense(units = 512,
    #                       activation= "relu")(model1)

    # model1 = layers.Dense(units=340,
    #                       activation="relu")(model1)

out = []
i = 0.01
l_rate = [ 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
l_rate = [ 1e-1, 1e-2, 1e-3]
for i in l_rate:
    model1 = creat_model()
    fit_out = model1.fit(x = mat_intput_train,
                         y = Y_train,
                         batch_size=32,
                         #steps_per_epoch= 5,
                         epochs =5,
                         validation_split= i,
                         verbose = 1)
    out.append({"res" :fit_out,"l_rate":i})

###############################################################################
############################## plot / Save ####################################
###############################################################################

now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
save = "../results/{}".format(now)

os.mkdir(save)
plot_model(model1,
           to_file = save+"/reseau.png")

for i in out:
    pyplot.plot(i["res"].history["loss"][1:])
pyplot.savefig(save + "/loss.png")

for i in out :
    pyplot.plot(i["res"].history["val_loss"])
pyplot.savefig(save + "/val_loss.png")

for i in out :
    i["res"].model.save(save + "/{}-.model".format(i["l_rate"]))



# training = KerasClassifier(build_fn = creat_model,
#                            epochs = 1,
#                            batch_size = 3)
# cross_validation = KFold(n_splits = 3,
#                          shuffle = True)
# cv_results = sklearn.model_selection.cross_val_score(training,
#                                                      mat_intput_train,
#                                                      Y_train,
#                                                      cv = cross_validation)
# model.save('{0}{}',format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
#                           cv_results))


# print(cv_results)

