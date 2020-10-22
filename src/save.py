import copy
import numpy
import matplotlib
from matplotlib import pyplot

def head_csv(save_dir,epoch):
    head = "renset,learningRate,batchsize,"
    for i in range(epoch):
        head += ",epoch_{}".format(i+1)
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
