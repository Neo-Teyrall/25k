
###############################################################################
################################ import   #####################################
###############################################################################
import data
import resnet

import os
import json
import getpass
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

###############################################################################
################################ Reseaux  #####################################
###############################################################################

def tail_model(model_in):
    model = layers.Conv1D(filters= 30,
                           kernel_size=(14,),
                           activation="relu")(model_in)
    model = layers.Conv1D(filters= 30,
                           kernel_size=(14,),
                           activation="relu")(model)
    model = layers.Conv1D(filters= 5,
                           kernel_size=(14,),
                           activation="softmax")(model)
    return model


def creat_model(resnet_part,rep = 5):
    model_input = tf.keras.Input(shape =(107,14))
    model = layers.Conv1D(filters= 30,
                           kernel_size=(3,),
                           activation = "relu",
                           padding="same")(model_input)
    for i in range(rep):
        model = resnet_part(model)
    model = tail_model(model)
    model = tf.keras.Model(inputs=model_input, outputs=model)
    model.compile(optimizer="rmsprop",
                    loss = "mse")
    return model


def compare_model(mat_input, mat_output, list_resnet, l_rates, nb_resnet = 20,
                  epochs = 5, batch_size = 16,verbose = 1):
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = "../results/{}-{}".format(now,getpass.getuser())

    os.mkdir(save_dir)
    for i_resnet,resnet_part in enumerate(list_resnet):

        res_name = str(resnet_part).split()[1]
        res_name_dir = save_dir + "/{}".format(res_name)
        os.mkdir(res_name_dir)

        model = creat_model(resnet_part,nb_resnet)
        plot_model(model,to_file=res_name_dir + "/model.png")

        fig_loss = pyplot.figure()
        ax_loss = fig_loss.add_subplot(1,1,1)
        fig_val_loss = pyplot.figure()
        ax_val_loss = fig_val_loss.add_subplot(1,1,1)

        for i_l_rate ,l_rate in enumerate(l_rates):
            print("{}:{}/{} || {}:{}/{}".format(res_name,i_resnet+1,
                                                len(list_resnet),
                                                str(l_rate), i_l_rate+1,
                                                len(l_rates)))
            model = creat_model(resnet_part,nb_resnet)
            out_fit = model.fit(x = mat_input,
                             y = mat_output,
                             batch_size=batch_size,
                             #steps_per_epoch= 5,
                             epochs =epochs,
                             validation_split= l_rate,
                                verbose = verbose)
            ax_loss.plot(out_fit.history["loss"])
            ax_val_loss.plot(out_fit.history["val_loss"])
            model.save(res_name_dir+"/_{}_.model".format(l_rate))


        fig_loss.savefig(res_name_dir+"/loss.png")
        fig_val_loss.savefig(res_name_dir+"/val_loss.png")

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


    mat_intput_train = data.mat_input(jsons_train)

    Y_train = data.mat_output(jsons_train)

    l_rates = [1e-1,1e-2,1e-3,
               1e-4,1e-5,1e-6,
               1e-7,1e-8,1e-9]

    list_resnet = [resnet.classic,
                   resnet.original,
                   resnet.pre_act,
                   resnet.pre_act_mod]

    compare_model(mat_input = mat_intput_train,
                  mat_output = Y_train,
                  list_resnet = list_resnet,
                  l_rates = l_rates,
                  nb_resnet = 2 ,
                  epochs = 2)
