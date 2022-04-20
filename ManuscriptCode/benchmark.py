import os, sys, re, glob, subprocess, glob, string
from collections import defaultdict
import numpy as np, pandas as pd
from itertools import product
from util import calCosine, calPearson, calSpearman, RunMultiProcess
from util import precision, getLmGenes, sigid2iname, drug2MOA
from CMapKS import runKS
from CMapGSEA import runGSEA
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

### drug annotation benchmark
sig_id2pert_iname = sigid2iname('')
sig_id2MOA = sigid2iname('MOA')
### LDA  PCA  Cosine   KS  GSEA   XSum  XCos  sscMap

def Pearson(query):
    ref = Xtr
    Nums = 10 if ref.shape[0] >=10 else ref.shape[0]
    query = query.sort_values(by=query.index[0], axis=1)   ####  基因排序
    genes = query.columns[:100].tolist() + query.columns[-100:].tolist()  ### 改变基因的数量
    query = query[genes]; ref = ref[genes]
    dat_cor = method(Xtr = ref, Xte = query)
    for i in dat_cor.index:
        tmp = dat_cor.loc[i,:]
        positive = tmp.sort_values(ascending=False)[:Nums].index.tolist()
        values = tmp.sort_values(ascending=False)[:Nums].values.tolist()
        values = [str(round(i,4)) for i in values]
    return (query.index[0], positive, values)

### 对每个query做循环
def f_Pearson(threads):
    global Xtr, method
    method = calCosine
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    doMultiProcess = RunMultiProcess()
    for GSE, cell_line, trTime in doMultiProcess.mylist:
        output_file  = 'ZScore/{}/{}_{}/cosine.tsv'.format(cell_line, GSE, trTime)
        file_Xtr = 'ZScore/{}/{}_{}/Xtr.h5'.format(cell_line, GSE, trTime)
        file_Xte = 'ZScore/{}/{}_{}/Xte.h5'.format(cell_line, GSE, trTime)
        if not os.path.isfile(file_Xtr) or not os.path.isfile(file_Xte):
            if os.path.isfile(output_file): os.remove(output_file)
            print ('{}\t{}\t{}\tfile not exist\t'.format(GSE, cell_line, trTime))
            continue
        Xtr = pd.read_hdf(file_Xtr); Xte = pd.read_hdf(file_Xte)
        pert_iname = [sig_id2pert_iname[i].split('_')[0] for i in Xtr.index]
        Xtr = Xtr.groupby(pert_iname, axis=0).median()
        tmp = [i for i in Xtr.columns if i in Xte.columns]
        Xtr = Xtr.loc[:, tmp]; Xte = Xte.loc[:, tmp]
        Xte_list = [Xte.iloc[i:i+1,:] for i in range(Xte.shape[0])]
        results = doMultiProcess.myPool(Pearson, Xte_list, threads)        
        with open(output_file, 'w') as fout:
            for index, positive, values in results:
                fout.write('{}\t{}\t{}\n'.format(index, '\t'.join(positive), '\t'.join(values)))
        print ('{}\t{}\t{}\tcosine\tcompleted\n'.format(GSE, cell_line, trTime))

def XSum(query):
    ref = Xtr
    Nums = 10 if ref.shape[0] >=10 else ref.shape[0]
    query = query.sort_values(by=query.index[0], axis=1, ascending = False)   ####  基因排序
    upGenes = query.columns[:100]; dnGenes = query.columns[-100:]   ### 改变基因的数量
    score = np.sum(Xtr.loc[:, upGenes], axis=1) - np.sum(Xtr.loc[:, dnGenes], axis=1)
    score.sort_values(inplace=True, ascending=False)
    positive = score.index[:Nums]; values = score.values[:10]; values = [str(round(i,4)) for i in values]
    return (query.index[0], positive, values)

### 把Xtr不显著的基因score设置为0, 计算XSum的score
def f_XSum(threads):
    global Xtr
    n = 500
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    doMultiProcess = RunMultiProcess()
    for GSE, cell_line, trTime in doMultiProcess.mylist:
        output_file  = 'ZScore/{}/{}_{}/xSum.tsv'.format(cell_line, GSE, trTime)
        file_Xtr = 'ZScore/{}/{}_{}/Xtr.h5'.format(cell_line, GSE, trTime)
        file_Xte = 'ZScore/{}/{}_{}/Xte.h5'.format(cell_line, GSE, trTime)
        if not os.path.isfile(file_Xtr) or not os.path.isfile(file_Xte):
            if os.path.isfile(output_file): os.remove(output_file)
            print ('{}\t{}\t{}\tfile not exist\t'.format(GSE, cell_line, trTime))
            continue
        Xtr = pd.read_hdf(file_Xtr); Xte = pd.read_hdf(file_Xte)
        pert_iname = [sig_id2pert_iname[i].split('_')[0] for i in Xtr.index]
        Xtr = Xtr.groupby(pert_iname, axis=0).median()
        tmp = [i for i in Xtr.columns if i in Xte.columns]
        Xtr = Xtr.loc[:, tmp]; Xte = Xte.loc[:, tmp]
        for i in Xtr.index:
            tmp = Xtr.loc[i,:]; tmp.sort_values(inplace=True); tmp.iloc[n:-n] = 0
            Xtr.loc[i, :] = tmp.loc[Xtr.columns]        
        Xte_list = [Xte.iloc[i:i+1,:] for i in range(Xte.shape[0])]
        results = doMultiProcess.myPool(XSum, Xte_list, threads)        
        with open(output_file, 'w') as fout:
            for index, positive, values in results:
                fout.write('{}\t{}\t{}\n'.format(index, '\t'.join(positive), '\t'.join(values)))
        print ('{}\t{}\t{}\txSum\tcompleted\n'.format(GSE, cell_line, trTime))

### 把Xtr不显著的基因score设置为0
def f_XPearson(threads):
    global Xtr, method
    n = 500
    method = calCosine
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    doMultiProcess = RunMultiProcess()
    for GSE, cell_line, trTime in doMultiProcess.mylist:
        output_file  = 'ZScore/{}/{}_{}/xcosine.tsv'.format(cell_line, GSE, trTime)
        file_Xtr = 'ZScore/{}/{}_{}/Xtr.h5'.format(cell_line, GSE, trTime)
        file_Xte = 'ZScore/{}/{}_{}/Xte.h5'.format(cell_line, GSE, trTime)
        if not os.path.isfile(file_Xtr) or not os.path.isfile(file_Xte):
            if os.path.isfile(output_file): os.remove(output_file)
            print ('{}\t{}\t{}\tfile not exist\t'.format(GSE, cell_line, trTime))
            continue
        Xtr = pd.read_hdf(file_Xtr); Xte = pd.read_hdf(file_Xte)
        pert_iname = [sig_id2pert_iname[i].split('_')[0] for i in Xtr.index]
        Xtr = Xtr.groupby(pert_iname, axis=0).median()
        tmp = [i for i in Xtr.columns if i in Xte.columns]
        Xtr = Xtr.loc[:, tmp]; Xte = Xte.loc[:, tmp]
        for i in Xtr.index:
            tmp = Xtr.loc[i,:]; tmp.sort_values(inplace=True); tmp.iloc[n:-n] = 0
            Xtr.loc[i, :] = tmp.loc[Xtr.columns]        
        Xte_list = [Xte.iloc[i:i+1,:] for i in range(Xte.shape[0])]
        results = doMultiProcess.myPool(Pearson, Xte_list, threads)        
        with open(output_file, 'w') as fout:
            for index, positive, values in results:
                fout.write('{}\t{}\t{}\n'.format(index, '\t'.join(positive), '\t'.join(values)))
        print ('{}\t{}\t{}\txcosine\tcompleted\n'.format(GSE, cell_line, trTime))


###  A simple and robust method for connecting small-molecule drugs using gene-expression signatures
def rRank(query):
    ref = Xtr
    Nums = 10 if ref.shape[0] >=10 else ref.shape[0]
    query = query.sort_values(by=query.index[0], axis=1)   ####  基因排序
    genes = query.columns[:100].tolist() + query.columns[-100:].tolist()  ### 改变基因的数量
    query = query[genes]; ref = ref[genes]
    query.iloc[0, :] = [i for i in range(-100, 100)]  ### 变成排序
    a = query.values; b = ref.values
    a = a.reshape(200); b = b.T; c = a.dot(b); max_c = np.abs(a).dot(np.abs(b))
    c = c / max_c
    dat_cor = pd.DataFrame(c.reshape(1, -1), columns=ref.index, index= query.index)
    for i in dat_cor.index:
        tmp = dat_cor.loc[i,:]
        positive = tmp.sort_values(ascending=False)[:Nums].index.tolist()
        values = tmp.sort_values(ascending=False)[:Nums].values.tolist()
        values = [str(round(i,4)) for i in values]
    return (query.index[0], positive, values)

def f_rRank(threads):
    global Xtr
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    doMultiProcess = RunMultiProcess()
    for GSE, cell_line, trTime in doMultiProcess.mylist:
        output_file  = 'ZScore/{}/{}_{}/rRank.tsv'.format(cell_line, GSE, trTime)
        file_Xtr = 'ZScore/{}/{}_{}/Xtr.h5'.format(cell_line, GSE, trTime)
        file_Xte = 'ZScore/{}/{}_{}/Xte.h5'.format(cell_line, GSE, trTime)
        if not os.path.isfile(file_Xtr) or not os.path.isfile(file_Xte):
            if os.path.isfile(output_file):os.remove(output_file)
            continue
        Xtr = pd.read_hdf(file_Xtr); Xte = pd.read_hdf(file_Xte)
        pert_iname = [sig_id2pert_iname[i].split('_')[0] for i in Xtr.index]
        Xtr = Xtr.groupby(pert_iname, axis=0).median()
        tmp = [i for i in Xtr.columns if i in Xte.columns]
        Xtr = Xtr.loc[:, tmp]; Xte = Xte.loc[:, tmp]
        for i in Xtr.index:
            tmp = Xtr.loc[i,:]; tmp.sort_values(inplace=True)
            a = sum(tmp <0); b = sum(tmp >=0)
            tmp.iloc[:] = [i for i in range(-a, b)]; Xtr.loc[i, :] = tmp.loc[Xtr.columns]
        Xte_list = [Xte.iloc[i:i+1,:] for i in range(Xte.shape[0])]
        results = doMultiProcess.myPool(rRank, Xte_list, threads)
        with open(output_file, 'w') as fout:
            for index, positive, values in results:
                fout.write('{}\t{}\t{}\n'.format(index, '\t'.join(positive), '\t'.join(values)))
        print ('{}\t{}\t{}\trRank\tcompleted\n'.format(GSE, cell_line, trTime))


def KSAndGSEA(threads):
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    doMultiProcess = RunMultiProcess(methods=[runKS, runGSEA])
    for GSE, cell_line, trTime, method in doMultiProcess.mylist:
        method_name = method.__name__[3:].lower()
        output_file  = 'ZScore/{}/{}_{}/{}.tsv'.format(cell_line, GSE, trTime, method_name)
        file_Xtr = 'ZScore/{}/{}_{}/Xtr.h5'.format(cell_line, GSE, trTime)
        file_Xte = 'ZScore/{}/{}_{}/Xte.h5'.format(cell_line, GSE, trTime)
        if not os.path.isfile(file_Xtr) or not os.path.isfile(file_Xte):
            if os.path.isfile(output_file): os.remove(output_file)
            print ('{}\t{}\t{}\tfile not exist\t'.format(GSE, cell_line, trTime))
            continue
        Xtr = pd.read_hdf(file_Xtr); Xte = pd.read_hdf(file_Xte)
        pert_iname = [sig_id2pert_iname[i].split('_')[0] for i in Xtr.index]
        Xtr = Xtr.groupby(pert_iname, axis=0).median()
        tmp = [i for i in Xtr.columns if i in Xte.columns]
        Xtr = Xtr.loc[:, tmp]; Xte = Xte.loc[:, tmp]
        if Xte.shape[0] >= 2000: Xte = Xte.iloc[:2000, :]
        Nums = 10 if Xtr.shape[0] >=10 else Xte.shape[0]
        dat_cor = method(Xtr = Xtr, Xte = Xte, processes = threads)
        with open(output_file, 'w') as fout:
            for i in dat_cor.index:
                tmp = dat_cor.loc[i,:]
                positive = tmp.sort_values(ascending=False)[:Nums].index.tolist() ## 相似是False
                values = tmp.sort_values(ascending=False)[:Nums].values.tolist()
                values = [str(round(i,4)) for i in values]
                fout.write('{}\t{}\t{}\n'.format(i, '\t'.join(positive), '\t'.join(values)))
        print ('{}\t{}\t{}\t{}\tcompleted\n'.format(GSE, cell_line, trTime, method_name))


#### 计算各个算法的准确性
def runPrecision(X):
    GSE, cell_line, trTime, methods = X
    results1 = []; results2 = []
    for method in methods:
        p, n = precision(GSE, cell_line, trTime, method)
        results1.append(p); results2.append(n)
    return [results1, results2, GSE, cell_line, trTime]


def f_runPrecision(threads):
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    methods = ['LMNN', 'pca', 'cosine', 'ks', 'gsea', 'xSum', 'rRank', 'xcosine']
    #methods = ['LMNN']
    fileout  = 'ZScore/AllGenes/MOA_benchmarkDrug.tsv'
    with open(fileout, 'w') as fout:
        fout.write('GSE\tcell_line\ttrTime\t{}\tNumsSamples\n'.format('\t'.join(methods)))
        doMultiProcess = RunMultiProcess(methods= [methods])
        results = doMultiProcess.myPool(runPrecision, doMultiProcess.mylist, threads)
        for results1, results2, GSE, cell_line, trTime in results:
            if any(results1):
                fout.write('{}\t{}\t{}\t{}\t{}\n'.format(GSE,cell_line,trTime, '\t'.join(results1), results2[0]))
    tmp = pd.read_csv(fileout, header=0, sep='\t')  ## 处理输出的结果
    tmp.sort_values(by='NumsSamples', ascending=False, inplace=True)
    tmp = tmp[tmp['NumsSamples'] >= 10]
    tmp['rs'] =  round(1 / tmp['NumsSamples'], 6)
    tmp['Rank_LMNN']  = list(map(int, tmp[methods].rank(axis=1, ascending=False)['LMNN'].values))
    tmp.to_csv(fileout, sep='\t', header=True, index=False, float_format='%.3f')



def precision_2(GSE, cell_line, trTime, method):
    Nums_right = 0; Nums_sample = 0
    drugtoMOA = drug2MOA('MOA')
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    filein  = 'ZScore/{}/{}_{}/{}{}.tsv'.format(cell_line, GSE, trTime, method, '')
    if not os.path.isfile(filein): return '', ''
    dat = pd.read_csv(filein, sep='\t', header=None)
    Nums_ypds = int((dat.shape[1] - 1) / 2)  ## 结果的个数
    dat.columns = ['pert_iname'] + ['pert_iname_' + str(i) for i in range(Nums_ypds)] + ['cor_' + str(i) for i in range(Nums_ypds)]
    for i in dat.columns[:1]:
        dat[i] = dat[i].apply(lambda x: sig_id2MOA.get(x, 'None_None').split('_')[0])
    for i in dat.columns[1:Nums_ypds+1]:
        dat[i] = dat[i].apply(lambda x: drugtoMOA.get(x, 'None_None').split('_')[0])
    for i in dat.index:
        yte = dat.loc[i, 'pert_iname']
        ypd_list = dat.loc[i, ['pert_iname_' + str(k) for k in range(Nums_ypds)]].tolist()
        ypd_list = [k for k in ypd_list if k != 'None'][:2]
        if yte != 'None':
            Nums_sample += 1
            if yte in ypd_list:  Nums_right += 1
    result = round(Nums_right / Nums_sample, 3)
    return str(result), str(len(np.unique(dat['pert_iname'])))

precision = precision_2

if __name__ == '__main__':
    print ('hello, world')
    threads = 60
    #f_Pearson(threads)
    #KSAndGSEA(threads)
    #f_XSum(threads)
    #f_rRank(threads)
    #f_XPearson(threads)
    f_runPrecision(threads)
