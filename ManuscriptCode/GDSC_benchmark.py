import os, sys, re, glob, subprocess, pickle, time, torch, warnings
from collections import defaultdict, Counter
from itertools import product, chain
import numpy as np, pandas as pd
from scipy import stats
from util import sigid2iname, convertDrugName, getLmGenes
from util import calCosine, calPearson, calSpearman, RunMultiProcess
import metric_learn, dml
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import manifold
from sklearn.metrics import pairwise_distances, pairwise, roc_curve,auc, roc_auc_score
from CMapKS import runKS
from CMapGSEA import runGSEA
from CMapPvalue import calPvalue, calPvalue1
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
warnings.filterwarnings('ignore')
basepath = '/home//project/Metric_learning'; os.chdir(basepath)
query_set = RunMultiProcess().query_set
ref_set = RunMultiProcess().ref_set
level = RunMultiProcess().level
singleLabel = RunMultiProcess().singleLabel
sig_id2pert_iname = sigid2iname('')
doPCA = True; rePCA = True; savaPCA = True; reverseExp = True

### 求到质心的相关性
sig_pValue = 0.01

def writeResult(dat_cor, output_file):
    dat_cor = dat_cor.T
    dat_cor.sort_values(by=dat_cor.columns[0], ascending=True, inplace=True)
    if dat_cor.shape[0] >= 10:  dat_cor = dat_cor.iloc[:10, :]
    dat_cor.to_csv(output_file, sep='\t', header=False, index= True)

def lfda(X):
    GSE, cell_line, trTime, metric = X
    path = 'msViper/{}/{}_{}/{}'.format(cell_line, GSE, trTime, level)
    file_Xtr = '{}/{}_Xtr_{}.h5'.format(path, ref_set, metric)
    file_PCA = '{}/{}_PCA_{}.pkl'.format(path, ref_set, metric)
    output_file1  = '{}/{}_LMNN_{}.tsv'.format(path, ref_set, metric)
    output_file2  = '{}/{}_PCA_{}.tsv'.format(path, ref_set, metric)
    file_Xte = '/home//project/Personal_Drug/{}/{}/Xte_{}.tsv'.format(query_set, cell_line, metric)
    if not os.path.isfile(file_Xtr) or not os.path.isfile(file_Xte):
        if os.path.isfile(output_file1): os.remove(output_file1)
        if os.path.isfile(output_file2): os.remove(output_file2)
        return ''
    Xtr = pd.read_hdf(file_Xtr)
    Xte = pd.read_csv(file_Xte, sep='\t', index_col=0)
    Xte.sort_values(by= Xte.index[0], ascending=True, inplace=True, axis=1)
    tmp = [i for i in Xtr.columns if i in Xte.columns]
    Xtr = Xtr.loc[:, tmp]; Xte = Xte.loc[:, tmp]
    experiment = 'nagative'
    if reverseExp:  Xtr = -Xtr; experiment = 'positive'
    if singleLabel:
        pert_iname = [sig_id2pert_iname[i].split('_')[0] for i in Xtr.index]
    else:
        pert_iname = [sig_id2pert_iname[i] for i in Xtr.index]
    if len(np.unique(pert_iname)) <= 10: return ''
    if doPCA and rePCA:
        n_components = .98
        pca = PCA(random_state=0, n_components = n_components)
        Xtr_pca = pca.fit_transform(Xtr); Xte_pca = pca.transform(Xte)
        if savaPCA:
            with open(file_PCA, 'wb') as fout:
                pickle.dump(pca, fout)
    elif doPCA and os.path.isfile(file_PCA):
        with open(file_PCA, 'rb') as fin:
            pca = pickle.load(fin)
            Xtr_pca = pca.transform(Xtr); Xte_pca = pca.transform(Xte)
    else:
        Xtr_pca = Xtr.values; Xte_pca = Xte.values
    labelencoder = LabelEncoder()
    ytr = labelencoder.fit_transform(pert_iname)
    ml = LinearDiscriminantAnalysis(solver='svd', n_components = 50)
    Xtr_pca_lmnn = ml.fit_transform(Xtr_pca, ytr); Xte_pca_lmnn = ml.transform(Xte_pca)
    Xtr_pca_lmnn = Xtr_pca_lmnn[:, ~np.isnan(Xtr_pca_lmnn)[0, :]]
    Xte_pca_lmnn = Xte_pca_lmnn[:, ~np.isnan(Xte_pca_lmnn)[0, :]]

    a = pd.DataFrame(Xtr_pca_lmnn, index=pert_iname)
    ref = a.groupby(pert_iname).median()
    query = pd.DataFrame(data = Xte_pca_lmnn, index = Xte.index)
    dat_cor = calCosine(Xtr = ref, Xte = query)
    if reverseExp: dat_cor = -dat_cor
    pValue = calPvalue(ref = ref, query = query, experiment= experiment, fun= calCosine)
    a = pValue <= sig_pValue; dat_cor = dat_cor.iloc[:, a.values[0]]
    writeResult(dat_cor, output_file1)

    a = pd.DataFrame(Xtr_pca, index = pert_iname)
    ref = a.groupby(pert_iname).median()
    query = pd.DataFrame(data = Xte_pca, index = Xte.index)
    dat_cor = calCosine(Xtr = ref, Xte = query)
    if reverseExp: dat_cor = -dat_cor
    pValue = calPvalue(ref = ref, query = query, experiment= experiment, fun= calCosine)
    a = pValue <= sig_pValue; dat_cor = dat_cor.iloc[:, a.values[0]]
    writeResult(dat_cor, output_file2)
    print ('{}\t{}\t{}\tlfda\tcompleted\n'.format(GSE, cell_line, trTime))


def runlfda(metric):
    doMultiProcess = RunMultiProcess(methods=[metric])
    for GSE, cell_line, trTime, method in doMultiProcess.mylist:
       lfda((GSE, cell_line, trTime, metric))

###################################################
def Pearson(ref, query):
    query = query.sort_values(by=query.index[0], axis=1)   ####  基因排序
    genes = query.columns[:100].tolist() + query.columns[-100:].tolist()  ### 改变基因的数量
    query = query[genes]; ref = ref[genes]
    dat_cor = calCosine(Xtr = ref, Xte = query)
    return dat_cor

def f_Pearson(metric):
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    doMultiProcess = RunMultiProcess()
    for GSE, cell_line, trTime in doMultiProcess.mylist:
        path = 'msViper/{}/{}_{}/{}'.format(cell_line, GSE, trTime, level)
        output_file  = '{}/{}_cosine_{}.tsv'.format(path, ref_set, metric)
        file_Xtr = '{}/{}_Xtr_{}.h5'.format(path, ref_set, metric)
        file_Xte = '/home//project/Personal_Drug/{}/{}/Xte_{}.tsv'.format(query_set, cell_line, metric)
        if not os.path.isfile(file_Xtr) or not os.path.isfile(file_Xte):
            if os.path.isfile(output_file): os.remove(output_file)
            continue
        Xtr = pd.read_hdf(file_Xtr)
        Xte = pd.read_csv(file_Xte, sep='\t', index_col=0)
        pert_iname = [sig_id2pert_iname[i].split('_')[0] for i in Xtr.index]
        tmp = [i for i in Xtr.columns if i in Xte.columns]
        Xtr = Xtr.loc[:, tmp]; Xte = Xte.loc[:, tmp]
        Xtr = Xtr.groupby(pert_iname, axis=0).median()
        dat_cor = Pearson(ref = Xtr, query= Xte)
        pValue = calPvalue1(ref = Xtr, query = Xte, experiment='negative', fun=Pearson)
        a = pValue <= sig_pValue; dat_cor = dat_cor.iloc[:, a.values[0]]
        writeResult(dat_cor, output_file)
        print ('{}\t{}\t{}\tcosine\tcompleted\n'.format(GSE, cell_line, trTime))



def XSum(ref, query):
    query = query.sort_values(by=query.index[0], axis=1, ascending = False)   ####  基因排序
    upGenes = query.columns[:100]; dnGenes = query.columns[-100:]   ### 改变基因的数量
    score = np.sum(ref.loc[:, upGenes], axis=1) - np.sum(ref.loc[:, dnGenes], axis=1)
    score.sort_values(inplace=True, ascending=False)
    dat_cor = pd.DataFrame(score.values.reshape(1, -1), columns=score.index, index=query.index)
    return dat_cor

### 把Xtr不显著的基因score设置为0, 计算XSum的score
def f_XSum(metric):
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    doMultiProcess = RunMultiProcess()
    for GSE, cell_line, trTime in doMultiProcess.mylist:
        path = 'msViper/{}/{}_{}/{}'.format(cell_line, GSE, trTime, level)
        output_file  = '{}/{}_XSum_{}.tsv'.format(path, ref_set, metric)
        file_Xtr = '{}/{}_Xtr_{}.h5'.format(path, ref_set, metric)
        file_Xte = '/home//project/Personal_Drug/{}/{}/Xte_{}.tsv'.format(query_set, cell_line, metric)
        if not os.path.isfile(file_Xtr) or not os.path.isfile(file_Xte):
            if os.path.isfile(output_file): os.remove(output_file)
            continue
        Xtr = pd.read_hdf(file_Xtr)
        pert_iname = [sig_id2pert_iname[i].split('_')[0] for i in Xtr.index]
        Xtr = Xtr.groupby(pert_iname, axis=0).median()
        for i in Xtr.index:
            tmp = Xtr.loc[i,:]; tmp.sort_values(inplace=True); tmp.iloc[500:-500] = 0
            Xtr.loc[i, :] = tmp.loc[Xtr.columns]
        Xte = pd.read_csv(file_Xte, sep='\t', index_col=0)
        tmp = [i for i in Xtr.columns if i in Xte.columns]
        Xtr = Xtr.loc[:, tmp]; Xte = Xte.loc[:, tmp]
        dat_cor = XSum(Xtr, Xte)
        pValue = calPvalue1(ref = Xtr, query = Xte, experiment='negative', fun=XSum)
        a = pValue <= sig_pValue; dat_cor = dat_cor.iloc[:, a.values[0]]
        writeResult(dat_cor, output_file)
        print ('{}\t{}\t{}\txSum\tcompleted\n'.format(GSE, cell_line, trTime))


### 把Xtr不显著的基因score设置为0
def f_XPearson(metric):
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    doMultiProcess = RunMultiProcess()
    for GSE, cell_line, trTime in doMultiProcess.mylist:
        path = 'msViper/{}/{}_{}/{}'.format(cell_line, GSE, trTime, level)
        output_file  = '{}/{}_Xcosine_{}.tsv'.format(path, ref_set, metric)
        file_Xtr = '{}/{}_Xtr_{}.h5'.format(path, ref_set, metric)
        file_Xte = '/home//project/Personal_Drug/{}/{}/Xte_{}.tsv'.format(query_set, cell_line, metric)
        if not os.path.isfile(file_Xtr) or not os.path.isfile(file_Xte):
            if os.path.isfile(output_file): os.remove(output_file)
            continue
        Xtr = pd.read_hdf(file_Xtr)
        pert_iname = [sig_id2pert_iname[i].split('_')[0] for i in Xtr.index]
        Xtr = Xtr.groupby(pert_iname, axis=0).median()
        for i in Xtr.index:
            tmp = Xtr.loc[i,:]; tmp.sort_values(inplace=True); tmp.iloc[500:-500] = 0
            Xtr.loc[i, :] = tmp.loc[Xtr.columns]
        Xte = pd.read_csv(file_Xte, sep='\t', index_col=0)
        tmp = [i for i in Xtr.columns if i in Xte.columns]
        Xtr = Xtr.loc[:, tmp]; Xte = Xte.loc[:, tmp]
        dat_cor = Pearson(Xtr, Xte)
        pValue = calPvalue1(ref = Xtr, query = Xte, experiment='negative', fun=Pearson)
        a = pValue <= sig_pValue; dat_cor = dat_cor.iloc[:, a.values[0]]
        writeResult(dat_cor, output_file)
        print ('{}\t{}\t{}\tXCosine\tcompleted\n'.format(GSE, cell_line, trTime))

###  A simple and robust method for connecting small-molecule drugs using gene-expression signatures
def rRank(ref, query):
    query = query.sort_values(by=query.index[0], axis=1)   ####  基因排序
    genes = query.columns[:100].tolist() + query.columns[-100:].tolist()  ### 改变基因的数量
    query = query[genes]; ref = ref[genes]
    query.iloc[0, :] = [i for i in range(-100, 100)]  ### 变成排序
    a = query.values; b = ref.values
    a = a.reshape(2*100); b = b.T; c = a.dot(b); max_c = np.abs(a).dot(np.abs(b))
    c = c / max_c
    dat_cor = pd.DataFrame(c.reshape(1, -1), columns=ref.index, index= query.index)
    return dat_cor

def f_rRank(metric):
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    doMultiProcess = RunMultiProcess()
    for GSE, cell_line, trTime in doMultiProcess.mylist:
        path = 'msViper/{}/{}_{}/{}'.format(cell_line, GSE, trTime, level)
        output_file  = '{}/{}_rRank_{}.tsv'.format(path, ref_set, metric)
        file_Xtr = '{}/{}_Xtr_{}.h5'.format(path, ref_set, metric)
        file_Xte = '/home//project/Personal_Drug/{}/{}/Xte_{}.tsv'.format(query_set, cell_line, metric)
        if not os.path.isfile(file_Xtr) or not os.path.isfile(file_Xte):
            if os.path.isfile(output_file): os.remove(output_file)
            continue
        Xtr = pd.read_hdf(file_Xtr)
        pert_iname = [sig_id2pert_iname[i].split('_')[0] for i in Xtr.index]
        Xtr = Xtr.groupby(pert_iname, axis=0).median()
        for i in Xtr.index:
            tmp = Xtr.loc[i,:]; tmp.sort_values(inplace=True)
            a = sum(tmp <0); b = sum(tmp >=0)
            tmp.iloc[:] = [i for i in range(-a, b)]; Xtr.loc[i, :] = tmp.loc[Xtr.columns]
        Xte = pd.read_csv(file_Xte, sep='\t', index_col=0)
        tmp = [i for i in Xtr.columns if i in Xte.columns]
        Xtr = Xtr.loc[:, tmp]; Xte = Xte.loc[:, tmp]
        dat_cor = rRank(Xtr, Xte)
        pValue = calPvalue1(ref = Xtr, query = Xte, experiment='negative', fun=rRank)
        a = pValue <= sig_pValue; dat_cor = dat_cor.iloc[:, a.values[0]]
        writeResult(dat_cor, output_file)
        print ('{}\t{}\t{}\trRank\tcompleted\n'.format(GSE, cell_line, trTime))

###########################################################
def KSAndGSEA(metric):
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    doMultiProcess = RunMultiProcess(methods=[runKS, runGSEA])
    for GSE, cell_line, trTime, method in doMultiProcess.mylist:
        method_name = method.__name__[3:].lower()
        path = 'msViper/{}/{}_{}/{}'.format(cell_line, GSE, trTime, level)
        output_file  = '{}/{}_{}_{}.tsv'.format(path, ref_set, method_name, metric)
        file_Xtr = '{}/{}_Xtr_{}.h5'.format(path, ref_set, metric)
        file_Xte = '/home//project/Personal_Drug/{}/{}/Xte_{}.tsv'.format(query_set ,cell_line, metric)
        if not os.path.isfile(file_Xtr) or not os.path.isfile(file_Xte):
            if os.path.isfile(output_file): os.remove(output_file)
            continue
        Xtr = pd.read_hdf(file_Xtr)
        Xte = pd.read_csv(file_Xte, sep='\t', index_col=0)
        tmp = [i for i in Xtr.columns if i in Xte.columns]
        Xtr = Xtr.loc[:, tmp]; Xte = Xte.loc[:, tmp]
        pert_iname = [sig_id2pert_iname[i].split('_')[0] for i in Xtr.index]
        Xtr = Xtr.groupby(pert_iname, axis=0).median()
        dat_cor = method(Xtr = Xtr, Xte = Xte, num_genes=100)
        pValue = calPvalue(ref = Xtr, query = Xte, experiment='negative', fun=method)
        if method_name == 'ks': sig_pValue = 0.05
        a = pValue <= sig_pValue; dat_cor = dat_cor.iloc[:, a.values[0]]
        writeResult(dat_cor, output_file)
        print ('{}\t{}\t{}\t{}\tcompleted\n'.format(GSE, cell_line, trTime, method_name))

#############################################################
def runIneffective(X):
    GSE, cell_line, trTime, method, metric = X
    IC50 = '/home//database/{}/{}/{}.txt'.format(ref_set, p1, cell_line)
    path = 'msViper/{}/{}_{}/{}/'.format(cell_line, GSE, trTime, level)
    filein = '{}/{}_{}_{}.tsv'.format(path, ref_set, method, metric)
    if os.path.isfile(IC50) and os.path.isfile(filein) and os.path.getsize(filein) >0:
        datIC50 = pd.read_csv(IC50, header=0, sep = '\t')
        datIC50.columns = ['drug', 'IC50']
        dat = pd.read_csv(filein, header=None, sep = '\t')
        dat.columns = ['drug', 'similarity']
        dat = pd.merge(left=datIC50, right=dat)
        if ref_set in ['CTRP']:  cutoff = np.median(datIC50['IC50'])
        else:  cutoff = 10000
        dat['group'] = np.where(dat.IC50 >= cutoff, 'ineffective', 'effective')
        dat['lable'] = np.where(dat.IC50 >= cutoff, 0, 1)
        Count = dat['group'].tolist()[:].count('effective')
        ACC = round(Count / dat.shape[0], 2)
        return str(ACC), dat.shape[0], str(Count)
    else:
        return '', '', ''
    
def runIneffective_merge(X):
    GSE, cell_line, trTime, method, metric = X
    IC50 = '/home//database/{}/{}/{}.txt'.format(ref_set, p1, cell_line)
    path = 'msViper/{}/{}_{}/{}/'.format(cell_line, GSE, trTime, level)
    filein = '{}/{}_{}_{}.tsv'.format(path, ref_set, method, metric)
    if os.path.isfile(IC50) and os.path.isfile(filein) and os.path.getsize(filein) >0:
        datIC50 = pd.read_csv(IC50, header=0, sep = '\t')
        dat = pd.read_csv(filein, header=None, sep = '\t')
        dat.columns = ['drug', 'similarity']
        dat = pd.merge(left=datIC50, right=dat)
        dat.sort_values(by='label', ascending=False, inplace=True)
        dat.drop_duplicates(subset= 'drug', keep='first', inplace=True)
        Count = dat['label'].tolist()[:].count('effective')
        ACC = round(Count / dat.shape[0], 2)
        return str(ACC), dat.shape[0], str(Count)
    else:
        return '', '', ''

def f_runIneffective(metric):
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    methods = ['LMNN', 'PCA', 'cosine', 'ks', 'gsea', 'XSum', 'rRank', 'Xcosine']
    doMultiProcess =  RunMultiProcess()
    fileout = 'msViper/{}_{}_{}_ROC_benchmark.tsv'.format(ref_set, level, metric)
    with open(fileout, 'w') as fout:
        fout.write('GSE\tcell_line\ttrTime\t{}\tNumSamples\tRank\n'.format('\t'.join(methods)))
        for GSE, cell_line, trTime in doMultiProcess.mylist:
            results = [runIneffective_merge((GSE, cell_line, trTime, method, metric))[0] for method in methods]
            NumSamples = runIneffective_merge((GSE, cell_line, trTime, methods[0], metric))[1]
            Ranks = [runIneffective_merge((GSE, cell_line, trTime, method, metric))[2] for method in methods]
            if any(results):
                fout.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(GSE,cell_line,trTime, '\t'.join(results), NumSamples, ','.join(Ranks)))
    tmp = pd.read_csv(fileout, header=0, sep='\t')  ## 处理输出的结果
    tmp.to_csv(fileout, sep='\t', header=True, index=False, float_format='%.2f')

if __name__ == '__main__':
    print ('hello, world'); metric = 'ZScore'; p1 = 'IC50'; p2 = 'median'
    runlfda(metric= metric)
    #f_Pearson(metric= metric)
    #KSAndGSEA(metric= metric)
    #f_rRank(metric)
    #f_XSum(metric)
    #f_XPearson(metric)
    f_runIneffective(metric)