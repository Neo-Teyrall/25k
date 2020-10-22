
###############################################################################
################################ import   #####################################
###############################################################################
import copy
import data
import resnet
import numpy
import os
import json
import getpass
import datetime
import time

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
                          kernel_size=(5,),
                          activation="relu",
                          padding = "same")(model_in)

    model = layers.Conv1D(filters= 20,
                          kernel_size=(5,),
                          activation="relu",
                          padding = "same")(model)
    model = layers.Conv1D(filters= 10,
                          kernel_size=(5,),
                          activation="relu",
                          padding = "same")(model)
    model = layers.Conv1D(filters= 4,
                          kernel_size=(5,),
                          activation="softmax",
                          padding = "same")(model)
    return model


def creat_model(resnet_part,rep = 5,optimizer = "rmsprop"):
    model_input = tf.keras.Input(shape =(68,14))
    model = layers.Conv1D(filters= 30,
                           kernel_size=(3,),
                           activation = "relu",
                           padding="same")(model_input)
    for i in range(rep):
        model = resnet_part(model)
    model = tail_model(model)
    model = tf.keras.Model(inputs=model_input, outputs=model)
    model.compile(optimizer=optimizer,
                  loss = "mse",
                  metrics = ["accuracy"])
    return model

def creat_model_janus(resnet_part,
                      optimizer = "rmsprop",
                      rep_head_1 = 2,
                      rep_head_2 = 2,
                      rep_merged = 2):
    ################### model 2 ###############################################
    model_input_1 = tf.keras.Input(shape =(68,14))
    model_1 = layers.Conv1D(filters= 30,
                           kernel_size=(3,),
                           activation = "relu",
                           padding="same")(model_input_1)
    for i in range(rep_head_1):
        model_1 = resnet_part(model_1)
    ################### model 2 ###############################################
    model_input_2 = tf.keras.Input(shape =(68,68))
    model_2 = layers.Conv1D(filters= 50,
                           kernel_size=(3,),
                           activation = "relu",
                           padding="same")(model_input_2)
    for i in range(rep_head_2):
        model_2 = resnet_part(model_2,50)
    ###################### merge model ########################################
    model = layers.Concatenate()([model_1, model_2])
    ###################### tail model ########################################
    for i in range(rep_merged):
        model = resnet_part(model,80)
    model = tail_model(model)
    ##################### Compile Model #####################################
    model = tf.keras.Model(inputs=[model_input_1,model_input_2], outputs=model)
    model.compile(optimizer=optimizer,
                  loss = "mse",
                  metrics = ["accuracy"])
    return model


def compare_model(mat_input, mat_output, list_resnet, l_rates, nb_resnet = 20,
                  epochs = 5, batch_size = 16,verbose = 0,sleep = 0):
    ## creat save repersitory 
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = "../results/modeles/{}-{}".format(now,getpass.getuser())
    os.mkdir(save_dir)
    head_csv(save_dir,epochs)
    ## pour chaque resnet
    for i_resnet,resnet_part in enumerate(list_resnet):
        ## creat type resnet repersitory
        res_name = str(resnet_part).split()[1]
        res_name_dir = save_dir + "/{}".format(res_name)
        os.mkdir(res_name_dir)
        ## sauvegarde du shéma du model
        model = creat_model(resnet_part,rep =nb_resnet )
        plot_model(model,to_file=res_name_dir + "/model.png")
        del(model)
        ## preparationd des plot
        #### loss
        fig_loss = pyplot.figure()
        ax_loss = fig_loss.add_subplot(1,1,1)
        fig_val_loss = pyplot.figure()
        ax_val_loss = fig_val_loss.add_subplot(1,1,1)
        #### acc
        fig_acc = pyplot.figure()
        ax_acc = fig_acc.add_subplot(1,1,1)
        fig_val_acc = pyplot.figure()
        ax_val_acc = fig_val_acc.add_subplot(1,1,1)
        dic_fig = {"loss": [fig_loss, ax_loss],
                   "val_loss": [fig_val_loss, ax_val_loss],
                   "accuracy": [fig_acc, ax_acc],
                   "val_accuracy": [fig_val_acc, ax_val_acc]}
        ## pour chaque learning rate
        for i_l_rate ,l_rate in enumerate(l_rates):
            ## optimizer
            tf.keras.backend.clear_session()
            opt = keras.optimizers.Adam(learning_rate=l_rate)
            print("{}:{}/{} || {}:{}/{}".format(res_name,i_resnet+1,
                                                len(list_resnet),
                                                str(l_rate), i_l_rate+1,
                                                len(l_rates)),flush  = True)
            ## creat_model
            model = creat_model(resnet_part,optimizer=opt,rep=nb_resnet)
            ## fit model 
            out_fit = model.fit(x = mat_input,
                                y = mat_output,
                                batch_size=batch_size,
                                #steps_per_epoch= 5,
                                epochs =epochs,
                                validation_split= 0.2,
                                verbose = verbose)
            ## add to plot data
            #### loss
            add_plot_history(dic_fig,out_fit.history,l_rate)
            ## sauvegarde du model 
            model.save(res_name_dir+"/_{}_.model".format(l_rate))
            save_csv(save_dir,batch_size,res_name,out_fit.history,l_rate)

            tf.keras.backend.clear_session()
            del(out_fit)
            del(model)
            ## sleep between two fit to fresh gpu
            print("begin Waiting at {} during {} second".format(datetime.datetime.now().strftime("%H-%M-%S"),sleep),flush  = True)
            time.sleep(sleep)
        ## save plot
        #### Xlabel
        save_fig(dic_fig,res_name_dir)
        del_fig([fig_acc,fig_val_acc,fig_val_loss,fig_loss])
        tf.keras.backend.clear_session()
    tf.keras.backend.clear_session()


def compare_model_janus(mat_input, mat_output,
                        list_resnet,
                        l_rates,
                        nb_resnet = [20,20,20],
                        epochs = 5,
                        batch_size = 16,
                        verbose = 0,sleep = 0 ):
    ## creat comparaison repersitory
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = "../results/modeles/{}-{}-janus".format(now,getpass.getuser())
    os.mkdir(save_dir)
    head_csv(save_dir,epochs)
    for i_resnet,resnet_part in enumerate(list_resnet):
        ## creat renset repersitory
        res_name = str(resnet_part).split()[1]
        res_name_dir = save_dir + "/{}".format(res_name)
        os.mkdir(res_name_dir)
        ## sauvegarde du shema du model
        model = creat_model_janus(resnet_part,
                                  rep_head_1= nb_resnet[0],
                                  rep_head_2= nb_resnet[1],
                                  rep_merged= nb_resnet[2])
        plot_model(model,to_file=res_name_dir + "/model.png")
        del(model)
        ## preparation des plot
        #### loss
        fig_loss = pyplot.figure()
        ax_loss = fig_loss.add_subplot(1,1,1)
        fig_val_loss = pyplot.figure()
        ax_val_loss = fig_val_loss.add_subplot(1,1,1)
        #### acc
        fig_acc = pyplot.figure()
        ax_acc = fig_acc.add_subplot(1,1,1)
        fig_val_acc = pyplot.figure()
        ax_val_acc = fig_val_acc.add_subplot(1,1,1)
        dic_fig = {"loss": [fig_loss, ax_loss],
                   "val_loss": [fig_val_loss, ax_val_loss],
                   "accuracy": [fig_acc, ax_acc],
                   "val_accuracy": [fig_val_acc, ax_val_acc]}
        ## pour chaque learning rate
        for i_l_rate ,l_rate in enumerate(l_rates):
            # optimiszer
            tf.keras.backend.clear_session()
            opt = keras.optimizers.Adam(learning_rate=l_rate)
            print("{}:{}/{} || {}:{}/{} | JANUS".format(res_name,i_resnet+1,
                                                len(list_resnet),
                                                str(l_rate), i_l_rate+1,
                                                len(l_rates)),flush  = True)
            model = creat_model_janus(resnet_part,optimizer=opt,
                                  rep_head_1= nb_resnet[0],
                                  rep_head_2= nb_resnet[1],
                                  rep_merged= nb_resnet[2])
            #fit model
            out_fit = model.fit(x = mat_input,
                             y = mat_output,
                             batch_size=batch_size,
                             #steps_per_epoch= 5,
                             epochs =epochs,
                             validation_split= 0.2,
                                verbose = verbose)
            ## add data to plot
            add_plot_history(dic_fig,out_fit.history,l_rate)
            ## save model
            model.save(res_name_dir+"/_{}_.model".format(l_rate))
            ## save data_to csv
            save_csv(save_dir,batch_size,res_name,out_fit.history,l_rate)
            
            tf.keras.backend.clear_session()
            del(model)
            del(out_fit)
            ## sleep between two fit to fresh gpu
            print("begin Waiting at {} during {} second".format(datetime.datetime.now().strftime("%H-%M-%S"),sleep),flush  = True)
            time.sleep(sleep)
        ## save fig 
        save_fig(dic_fig,res_name_dir)
        del_fig([fig_acc,fig_val_acc,fig_val_loss,fig_loss])
        tf.keras.backend.clear_session()


def head_csv(save_dir,epoch):
    head = "renset,learningRate,batchsize,"
    for i in range(epoch):
        head += ",epoch_{}".format(i+1)
        pass
    with open(save_dir+"/learning.csv","w") as filout:
        filout.write(head+"\n")

def save_csv(save_dir,batchsize,res_name,history,l_rate):
    line = "{},{},{}".format(res_name,l_rate,batchsize)
    out = []
    for i in history:
        app_line = copy.copy(line)
        app_line +=","+i
        for j in history[i]:
            app_line += ",{:.6f}".format(j)
        out.append(app_line)
    with open(save_dir+"/learning.csv","a") as filout:
        for i in out:
            filout.write(i+"\n")

def add_plot_history(dic_fig, history,l_rate):
    for i in history:
        dic_fig[i][1].plot(history[i],label = str(l_rate))

def save_fig(dic_fig,res_name_dir):
    for i in dic_fig:
        dic_fig[i][1].set_xlabel("epochs")
        dic_fig[i][1].set_ylabel(i)
        dic_fig[i][1].set_title(i + " en fonction des epochs")
        dic_fig[i][0].legend()
        dic_fig[i][0].savefig(res_name_dir + "/" + i + ".png")
        numpy.save(res_name_dir + "/" + i +".npy",
                   [dic_fig[i][0]])

def del_fig(args):
    for i in args:
        pyplot.close(i)
        del(i)


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

    compare_model(mat_input = mat_intput_train,
                  mat_output = Y_train,
                  list_resnet = list_resnet,
                  l_rates = l_rates,
                  nb_resnet = 30 ,
                  epochs = 50,
                  verbose = 0,sleep = 60)

    compare_model_janus(mat_input = (mat_intput_train, mat_input2),
                        mat_output = Y_train,
                        list_resnet = list_resnet,
                        l_rates = l_rates,
                        nb_resnet = [20,20,20] ,
                        epochs = 50,sleep = 60*3,verbose = 0)

