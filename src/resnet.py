import tensorflow as tf
from tensorflow.keras import layers

def classic(model_in,filters = 30):
    model = layers.Conv1D(filters= filters,
                          kernel_size=(3,),
                          activation= "relu",
                          padding="same")(model_in)
    model = layers.Conv1D(filters= filters,
                          kernel_size=(3,),
                          activation= "relu",
                          padding="same")(model)
    model = layers.Conv1D(filters= filters,
                          kernel_size=(3,),
                          activation= "relu",
                          padding="same")(model)
    model = layers.Add()([model, model_in])
    return model



def original(model_in,filters = 30):
    model  = layers.Conv1D(filters = filters,
                           kernel_size=(3,),
                           activation = "relu",
                           padding="same")(model_in)
    model = layers.BatchNormalization()(model)
    model = layers.Activation("relu")(model)
    model = layers.Conv1D(filters = filters,
                           kernel_size=(3,),
                           activation = "relu",
                           padding="same")(model)
    model = layers.BatchNormalization()(model)
    model = layers.Add()([model, model_in])
    return model


def pre_act(model_in,filters = 30):
    model = layers.BatchNormalization()(model_in)
    model = layers.Activation(activation="relu")(model)
    model = layers.Conv1D(filters = filters,
                           kernel_size=(3,),
                           activation = "relu",
                           padding="same")(model)
    model = layers.BatchNormalization()(model)
    model = layers.Activation(activation="relu")(model)
    model = layers.Conv1D(filters = filters,
                           kernel_size=(3,),
                           activation = "relu",
                           padding="same")(model)
    model = layers.Add()([model, model_in])
    return model


def pre_act_mod(model_in,filters = 30):
    model = layers.BatchNormalization()(model_in)
    model = layers.Conv1D(filters = filters,
                           kernel_size=(3,),
                           activation = "relu",
                           padding="same")(model)
    model = layers.BatchNormalization()(model)
    model = layers.Activation(activation="relu")(model)
    model = layers.Conv1D(filters = filters,
                           kernel_size=(3,),
                           activation = "relu",
                           padding="same")(model)
    model = layers.Add()([model, model_in])
    return model

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
    model = layers.Conv1D(filters= 5,
                          kernel_size=(5,),
                          activation="softmax",
                          padding = "same")(model)
    return model
