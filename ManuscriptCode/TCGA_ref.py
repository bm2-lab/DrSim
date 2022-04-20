import re, os, glob, subprocess, string
import pandas as pd, numpy as np
from itertools import product
from util import sigid2iname
from util import RunMultiProcess, getLmGenes, convertDrugName
from sklearn.utils import shuffle
sig_id2pert_iname = sigid2iname('')
level = RunMultiProcess().level
filterDrugs = True      ###过滤掉一些没有注释的小分子药物


## generate reference signature using LINCS data
def fun3(cell_line):
    filein = '/home/wzt/project/Metric_learning/TCGA/{}/FDAapproved_raw.txt'.format(cell_line)
    fileout = '/home/wzt/project/Metric_learning/TCGA/{}/FDAapproved.txt'.format(cell_line)
    with open(filein, 'r') as fin, open(fileout, 'w') as fout:
        for line in fin:
            lines = line.strip().split(' ')
            for i in lines:
                i = i.strip(' ,()'); i = convertDrugName(i); i = i.lower()
                fout.write('{}\n'.format(i))

def fun1(x):
    def foo(x):
        if x in string.ascii_letters: return True
        else: return False
    return all((map(foo, list(x))))

def fun2():
    filein = '/home/wzt/database/CMap/CMap_FDADrugs.tsv'
    dat = pd.read_csv(filein, sep='\t', header=0)
    dat = dat[dat['Phase'] == 'Launched']
    return dat['Name'].apply(convertDrugName).values


def preData(X):
    GSE, cell_line, trTime = X
    allSize = 3   ####
    FDA_Approved = fun2()
    basepath = '/home/wzt/project/Metric_learning'; os.chdir(basepath)
    input_file = 'ZScore/{}/{}_{}/zscore{}.h5'.format(cell_line, GSE, trTime, level)
    if not os.path.isfile(input_file): return ''
    dat = pd.read_hdf(input_file, key='dat')
    a = dat.index
    a = [i for i in a if i in sig_id2pert_iname]    #### MOA会有药物不存在
    b = list(map(lambda x: sig_id2pert_iname[x].split('_')[0], a))  ### 只看药物标签
    if filterDrugs:
        selected = [True if i in FDA_Approved and b.count(i) >= allSize else False for i in b]
    else:
        selected = [True if b.count(i) >= allSize else False for i in b]
    a_ = np.array(a)[selected]; b_ = np.array(b)[selected]
    mydir = 'TCGA/{}/{}_{}/{}'.format(cell_line, GSE, trTime, level)
    if not os.path.isdir(mydir): os.makedirs(mydir)
    file_Xtr = 'TCGA/{}/{}_{}/{}/Xtr.h5'.format(cell_line, GSE, trTime, level)
    if os.path.isfile(file_Xtr): os.remove(file_Xtr)
    if len(np.unique(b_)) <= 10: return
    Xtr = dat.loc[a_, :]
    Xtr.to_hdf(file_Xtr, key = 'dat')


cell_lines = ['MCF7', 'A549', 'HCC515', 'PC3', 'VCAP']
cell_lines = ['A375', 'HT29', 'BT20', 'HEPG2']
#cell_lines = ['A549']
trTimes = ['24H', '6H']
def f_preData():
    for cell_line in cell_lines:
        for trTime in trTimes:
            preData(('GSE92742', cell_line, trTime))


if __name__ == "__main__":
    print ('hello, world')
    f_preData()
