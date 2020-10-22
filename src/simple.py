import os
import datetime
import resnet
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

from save import *
import getpass
import time

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
                                                len(l_rates)),flush = True)
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
            print("begin Waiting at {} during {} second".format(datetime.datetime.now().strftime("%H-%M-%S"),sleep),flush = True)
            time.sleep(sleep)
        ## save plot
        #### Xlabel
        save_fig(dic_fig,res_name_dir)
        del_fig([fig_acc,fig_val_acc,fig_val_loss,fig_loss])
        tf.keras.backend.clear_session()
    tf.keras.backend.clear_session()

