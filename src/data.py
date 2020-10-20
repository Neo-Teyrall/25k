import pandas
import numpy
import copy
import math
import statistics
import datetime
import json

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
    list_mat = []
    for i,json in enumerate(jsons) :
        print("loading input : {:.2f} %".format((i+1)/len(jsons)*100),end = "\r")
        mat = numpy.load("../data/bpps/"+ json["id"]+".npy")
        ohaa  = one_hot_AA_seq(json["sequence"])
        ohsty = one_hot_struct_seq(json["predicted_loop_type"])
        ohst = one_hot_structure(json["structure"])
        data = numpy.concatenate((ohaa ,ohsty , ohst), axis= 1)
        data_out.append(data)
        list_mat.append(mat)
    mat_out = numpy.array(data_out)
    list_mat = numpy.array(list_mat)
    print()
    return(mat_out,list_mat)


def mat_output(jsons):
    data_out = []
    for i,json in enumerate(jsons) :
        print("loading output : {:.2f} %".format((i+1)/len(jsons)*100),end = "\r")
        reactivity = json["reactivity"]
        deg_Mg_pH10 = json["deg_Mg_pH10"]
        deg_pH10 = json["deg_pH10"]
        deg_Mg_50C = json["deg_Mg_50C"]
        deg_50C = json["deg_50C"]
        out = []
        for j in range(len(reactivity)):
             out.append([reactivity[j],
                         deg_Mg_pH10[j],
                         deg_pH10[j],
                         deg_Mg_50C[j],
                         deg_50C[j]])
        data_out.append(out)
    print()
    data_out = numpy.array(data_out)
    return data_out

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


    mat_intput_train,mat_input2 = mat_input(jsons_train)

    Y_train = mat_output(jsons_train)
    pass
