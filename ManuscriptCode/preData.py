import re, os, glob, subprocess
import pandas as pd, numpy as np
from itertools import product
from util import sigid2iname, drug2MOA
from util import RunMultiProcess, getLmGenes, convertDrugName
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


level = 'L4'
def myfun(x, y):
    allSize = 2   ### 一个MOA有几个药物
    selected = [True if list(y).count(i) >= allSize else False for i in y]
    x_ = np.array(x)[selected]; y_ = np.array(y)[selected]
    testSize_per = round((len(np.unique(y_))) / len(y_), 2) + 0.01
    testSize_per = .5
    print (testSize_per)
    Xtr, Xte, ytr, yte = train_test_split(x_, y_, random_state = 2020, 
                    test_size = testSize_per, stratify = y_)
    return Xtr, Xte, ytr, yte

def preData1(X):
    minSize, maxSize  = 5, 10000      ###  单个药物的重复数
    basepath = '/home/wzt/project/Metric_learning'; os.chdir(basepath)
    GSE, cell_line, trTime = X
    input_file = 'ZScore/{}/{}_{}/zscore{}.h5'.format(cell_line, GSE, trTime, level)
    if not os.path.isfile(input_file): return ''
    dat = pd.read_hdf(input_file, key='dat')
    sig_id2pert_iname = sigid2iname('')
    sig_id2MOA = sigid2iname('MOA')
    drugtoMOA = drug2MOA('MOA')
    a = [i for i in dat.index if i in sig_id2MOA]    #### 会有药物不存在MOA
    b = list(map(lambda x: sig_id2pert_iname[x].split('_')[0], a))  ### 只看药物标签
    selected = [True if  maxSize >= b.count(i) >= minSize else False for i in b]
    a_ = np.array(a)[selected]; b_ = np.array(b)[selected]

    drugs = np.unique(b_); MOAs = [drugtoMOA[i] for i in drugs]
    drugs_Xtr, drugs_Xte, _, _ = myfun(drugs, MOAs)
    file_Xtr = 'ZScore/{}/{}_{}/Xtr.h5'.format(cell_line, GSE, trTime)
    file_Xte = 'ZScore/{}/{}_{}/Xte.h5'.format(cell_line, GSE, trTime)
    if os.path.isfile(file_Xtr): os.remove(file_Xtr)
    if os.path.isfile(file_Xte): os.remove(file_Xte)
    selected_Xtr = [True if i in drugs_Xtr  else False for i in b_]
    selected_Xte = [True if i in drugs_Xte  else False for i in b_]
    Xtr = a_[selected_Xtr]; Xte = a_[selected_Xte]
    Xtr = dat.loc[Xtr, :]; Xte = dat.loc[Xte, :]
    Xtr.to_hdf(file_Xtr, key = 'dat'); Xte.to_hdf(file_Xte, key = 'dat')


def f_preData(fun):
    doMultiProcess = RunMultiProcess()
    doMultiProcess.myPool(fun, doMultiProcess.mylist, 6)

if __name__ == "__main__":
    print ('hello, world')
    f_preData(preData1)