import pandas
import numpy
import copy
import math
import statistics
import datetime
import json
import random

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

def mat_input(jsons,decalage = 0, taille = 68,):
    datas = []
    for i,json in enumerate(jsons) :
        print("loading input : {:.2f} %".format((i+1)/len(jsons)*100),end = "\r")
        mat = numpy.load("../data/bpps/"+ json["id"]+".npy")
        mat = reshape_mat(mat,start= decalage, size  = taille)
        ohaa  = one_hot_AA_seq(json["sequence"][decalage:taille])
        ohsty = one_hot_struct_seq(json["predicted_loop_type"][decalage:taille])
        ohst = one_hot_structure(json["structure"][decalage:taille])
        data = numpy.concatenate((ohaa ,ohsty , ohst), axis= 1)
        data = numpy.array(data)
        datas.append([data,mat])
    return(datas)

def mat_input_test(jsons,decalage = 0, taille = 68,):
    datas = {}
    for i,json in enumerate(jsons) :
        print("loading input : {:.2f} %".format((i+1)/len(jsons)*100),end = "\r")
        id_ = json["id"]
        length = json["seq_length"]

        data, mat = get_data_decalage(json,decalage,taille) 
        data1, mat1 = get_data_decalage(json,length-taille,taille) 
        datas.setdefault(id_,{"pos" : numpy.array([data,data1]), "mat" : numpy.array([mat,mat1]),"len" : length})
    return(datas)

def get_data_decalage(json, decalage ,taille):
    mat = numpy.load("../data/bpps/"+ json["id"]+".npy")
    mat = reshape_mat(mat,start= decalage, size  = taille)
    ohaa  = one_hot_AA_seq(json["sequence"][decalage:decalage + taille])
    ohsty = one_hot_struct_seq(json["predicted_loop_type"][decalage:decalage + taille])
    ohst = one_hot_structure(json["structure"][decalage:decalage + taille])
    data = numpy.concatenate((ohaa ,ohsty , ohst), axis= 1)
    data = numpy.array(data)
    return data, mat
    pass

def split_datas(datas):
    colision = []
    vec_pos = []
    out = []
    for i in datas:
        colision.append(i[1])
        vec_pos.append(i[0])
        out.append(i[2])
    colision = numpy.array(colision)
    vec_pos = numpy.array(vec_pos)
    out = numpy.array(out)
    return vec_pos,colision,out

def get_cross_val_ready(cv_datas):
    cv_ready = []
    for i in cv_datas:
        cv_ready.append(list(split_datas(i)))
    return cv_ready

def C_V(datas,k):
    datas =cross_val(datas,k)
    datas_cv  = get_cross_val_ready(datas)
    return datas_cv

def reshape_mat(mat,start = 0 ,size = 68):
    mat = mat[start:start+size,start:start+size]
    return mat

def merge_data(datas,output):
    all_data = []
    for i in range(len(datas)):
        idvd = copy.deepcopy(datas[i])
        idvd.append(output[i])
        all_data.append(idvd)
    
    return all_data


def mat_output(jsons):
    data_out = []
    for i ,json in  enumerate(jsons):
        reactivity = json["reactivity"]
        deg_Mg_pH10 = json["deg_Mg_pH10"]
        deg_pH10 = json["deg_pH10"]
        deg_Mg_50C = json["deg_Mg_50C"]
        deg_50C = json["deg_50C"]
        out = []
        for j in range(len(deg_50C)):
            out.append([reactivity[j],
                        deg_Mg_pH10[j],
                        deg_pH10[j],
                        deg_Mg_50C[j],
                        deg_50C[j]])
        data_out.append(out)
    return numpy.array(data_out)

def cross_val(data,k):
    out = []
    data = copy.deepcopy(data)
    random.shuffle(data)
    frac = int(len(data)/k)
    for i in range(k-1) :
        out.append(data[i*frac:i*frac+frac])
    out.append(data[(i+1)*frac:])
    return out

def merge_cross_val_exept(CV_data,excepte):
    exp = {"pos": [], "mat": [], "out" : []}
    un_exp = {"pos": [], "mat": [], "out": []}
    for i ,val  in enumerate(CV_data) :
        if i == excepte: 
            exp["pos"] = val[0]
            exp["mat"] = val[1]
            exp["out"] = val[2]
            continue
        un_exp["pos"].extend(val[0])
        un_exp["mat"].extend(val[1])
        un_exp["out"].extend(val[2])

    un_exp["pos"] = numpy.array(un_exp["pos"])
    un_exp["mat"] = numpy.array(un_exp["mat"])
    un_exp["out"] = numpy.array(un_exp["out"])
    exp["pos"] = numpy.array(exp["pos"])
    exp["mat"] = numpy.array(exp["mat"])
    exp["out"] = numpy.array(exp["out"])
    return un_exp , exp

def get_window_data(jsons,window = 3):
    out1 = []
    out2 = []
    out3 = []
    out123 = []
    out_index  = numpy.array([0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0])
    out_mat = numpy.matrix([-1]*window*window).reshape(window,window)
    half = int(window/2)
    for i,json in enumerate(jsons):
        print("loading input : {:.2f} %".format((i+1)/len(jsons)*100),end = "\r")
        mat = numpy.load("../data/bpps/"+ json["id"]+".npy")
        ohaa  = one_hot_AA_seq(json["sequence"])
        ohsty = one_hot_struct_seq(json["predicted_loop_type"])
        ohst = one_hot_structure(json["structure"])
        #
        data = numpy.concatenate((ohaa ,ohsty , ohst), axis= 1)
        data = numpy.array(data)
        #
        reactivity = json["reactivity"]
        deg_Mg_pH10 = json["deg_Mg_pH10"]
        deg_pH10 = json["deg_pH10"]
        deg_Mg_50C = json["deg_Mg_50C"]
        deg_50C = json["deg_50C"]
        output = []
        for j in range(len(deg_50C)):
            output.append([reactivity[j],
                        deg_Mg_pH10[j],
                        deg_pH10[j],
                        deg_Mg_50C[j],
                        deg_50C[j]])
        #
        for i in range(len(output)):
            m = copy.deepcopy(out_mat)
            if i in range(0,half) :
                frag  = []
                m[half:,half:] = mat[i:i+half+1,i:i+half+1]
                for h in range(-half,0):
                    if i+h < 0 :
                        frag.append((copy.deepcopy(out_index)))
                    else :
                        frag.append(data[i+h])
                frag.append(data[i])
                for h in range(1,half+1):
                    frag.append(data[i+h])
            elif i in range(len(output)-half,len(output)):
                
                m[:half+1,:half+1] =  mat[i-half:i+1,i-half:i+1]
                frag = []
                for h in range(1,half+1):
                    frag.append(data[i-h])
                frag.append(data[i])
                for h in range(1,half+1):
                    if i+h < len(output):
                        frag.append(data[i+h])
                    else:
                        frag.append(copy.deepcopy(out_index))
            else :
                m = mat[i-half:i+half+1,i-half:i+half+1]
                frag = data[i-half:i+half+1]
            #
            out1.append(numpy.array(frag))
            out2.append(numpy.array(m))
            out3.append(numpy.array(output[i]))
            out123.append([frag,m,output[i]])
    return( {"pos": numpy.array(out1),
            "mat": numpy.array(out2),
            "out":numpy.array(out3),
            "to_cv": out123})



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


    datas_input = mat_input(jsons_train)
    
    Y_train = mat_output(jsons_train)
    data_all  = merge_data(datas_input,Y_train)

    CV_data = C_V(data_all,k = 5)
    learning, cv =  merge_cross_val_exept(CV_data,1)

    ####
    a = get_window_data(jsons_train,window = 5)


