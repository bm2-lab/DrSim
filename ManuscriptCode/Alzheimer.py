import re, os, glob, subprocess, pickle, time
import pandas as pd, numpy as np, warnings
from itertools import product, chain
from TCGA_query import getExp
from scipy.stats import stats
from collections import defaultdict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from util import RunMultiProcess, sigid2iname, calCosine, convertDrugName
from GDSC_benchmark import Pearson, rRank, XSum
from CMapPvalue import calPvalue, calPvalue1
from CMapKS import runKS
from CMapGSEA import runGSEA
warnings.filterwarnings('ignore')
doPCA = True; rePCA = False; savaPCA = False; reverseExp = True
sig_id2pert_iname = sigid2iname('')

p_value = 0.01
def writeResult(tmp, output_file, gold_standard_drug):
    tmp.sort_values(by='treat', inplace=True)
    tmp['Drug'] = tmp.index
    if tmp.shape[0] >=10: tmp = tmp.iloc[:10, :]
    dat1 = pd.read_csv(gold_standard_drug, sep='\t', header=None)
    dat1.columns = ['Drug', 'Evidence', 'temp']
    dat1 = dat1[['Drug', 'Evidence']]
    dat1.drop_duplicates(subset=['Drug'], inplace=True)
    dat = pd.merge(tmp, dat1, how='left')
    dat = dat[['Drug', 'treat', 'Evidence']]
    dat.to_csv(output_file, sep='\t', float_format='%.3f', header=False, index= False)


def writeResult1(dat_cor, output_file, gold_standard_drug, sigValue = -0.3):
    dat_cor = dat_cor.T
    dat_cor.columns = ['treat']
    dat_cor.sort_values(by='treat', ascending=True, inplace=True)
    pert_iname = [sig_id2pert_iname[i].split('_')[0] for i in dat_cor.index]
    dat_cor.index = pert_iname
    dat_cor = dat_cor.groupby(dat_cor.index).median()
    if sum(dat_cor['treat'] <= sigValue) ==0:
        dat_cor = dat_cor.iloc[:5, :]
    elif sum(dat_cor['treat'] <= sigValue) >=10:
        dat_cor = dat_cor.iloc[:10, :]
    else:
        dat_cor = dat_cor[dat_cor['treat'] <= sigValue]
    dat_cor['Drug'] = dat_cor.index
    dat1 = pd.read_csv(gold_standard_drug, sep='\t', header=None)
    dat1.columns = ['Drug', 'Evidence', 'temp']
    dat1 = dat1[['Drug', 'Evidence']]
    dat1.drop_duplicates(subset=['Drug'], inplace=True)
    dat = pd.merge(dat_cor, dat1, how='left')
    dat = dat[['Drug', 'treat', 'Evidence']]
    dat.to_csv(output_file, sep='\t', float_format='%.3f', header=False, index= False)

def lfda(X):
    GSE, cell_line, trTime = X
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    output_file1  = 'Alzheimer/{}/{}/{}_LMNN.tsv'.format(GSE, cell_line,  trTime)
    output_file2  = 'Alzheimer/{}/{}/{}_pca.tsv'.format(GSE, cell_line,  trTime)
    file_Xtr = 'TCGA/{}/GSE92742_{}/L4/Xtr.h5'.format(cell_line, trTime)
    file_Xte = 'Alzheimer/{}/Xte_ZScore.tsv'.format(GSE)
    file_PCA = 'Alzheimer/{}/PCA/{}_{}_PCA.pkl'.format(GSE, cell_line,  trTime)
    gold_standard_drug = 'Alzheimer/goldStandardDrug.tsv'
    Xtr = pd.read_hdf(file_Xtr)
    Xte = pd.read_csv(file_Xte, sep= '\t', header=0, index_col = 0)
    Xte.sort_values(by= Xte.index[0], ascending=True, inplace=True, axis=1)
    tmp = [i for i in Xtr.columns if i in Xte.columns]
    Xtr = Xtr.loc[:, tmp]; Xte = Xte.loc[:, tmp]
    pert_iname = [sig_id2pert_iname[i].split('_')[0] for i in Xtr.index]
    experiment = 'negative'
    if reverseExp: Xtr = -Xtr; experiment = 'positive'
    if doPCA and rePCA:
        n_components = .98
        pca = PCA(random_state= 2020, n_components = n_components) ## 改变维度
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
    ml = LinearDiscriminantAnalysis(solver='svd', n_components=50)
    Xtr_pca_lmnn = ml.fit_transform(Xtr_pca, ytr); Xte_pca_lmnn = ml.transform(Xte_pca)
    Xtr_pca_lmnn = Xtr_pca_lmnn[:, ~np.isnan(Xtr_pca_lmnn)[0]]   ### 取出不是Nan的列
    Xte_pca_lmnn = Xte_pca_lmnn[:, ~np.isnan(Xte_pca_lmnn)[0]]   ### 取出不是Nan的列
    
    a = pd.DataFrame(Xtr_pca, index=pert_iname)
    ref = a.groupby(pert_iname).median()
    query = pd.DataFrame(data = Xte_pca, index = Xte.index)
    dat_cor = calCosine(Xtr = ref, Xte = query)
    if reverseExp: dat_cor = -dat_cor
    pValue = calPvalue(ref, query, experiment = experiment,fun=calCosine)
    a = pValue <= p_value; tmp = dat_cor.iloc[:, a.values[0]].T
    writeResult(tmp, output_file2, gold_standard_drug)  
    
    a = pd.DataFrame(Xtr_pca_lmnn, index=pert_iname)
    ref = a.groupby(pert_iname).median()
    query = pd.DataFrame(data = Xte_pca_lmnn, index = Xte.index)
    dat_cor = calCosine(Xtr = ref, Xte = query)
    if reverseExp: dat_cor = -dat_cor
    pValue = calPvalue(ref, query, experiment=experiment, fun=calCosine)
    a = pValue <= p_value; tmp = dat_cor.iloc[:, a.values[0]].T
    writeResult(tmp, output_file1, gold_standard_drug)    

def runlfda():
    for GSE, cell_line, trTime in mylist:
        lfda((GSE, cell_line, trTime))

###################################################
def f_Pearson():
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    for GSE, cell_line, trTime in mylist:
        method_name = 'cosine'
        gold_standard_drug = 'Alzheimer/goldStandardDrug.tsv'
        output_file  = 'Alzheimer/{}/{}/{}_{}.tsv'.format(GSE, cell_line,  trTime, method_name)
        file_Xtr = 'TCGA/{}/GSE92742_{}/L4/Xtr.h5'.format(cell_line, trTime)
        file_Xte = 'Alzheimer/{}/Xte_ZScore.tsv'.format(GSE)
        Xtr = pd.read_hdf(file_Xtr)
        Xte = pd.read_csv(file_Xte, sep='\t', header=0, index_col=0)
        tmp = [i for i in Xtr.columns if i in Xte.columns]
        Xtr = Xtr.loc[:, tmp]; Xte = Xte.loc[:, tmp]
        dat_cor = Pearson(ref= Xtr, query= Xte)
        pValue = calPvalue1(ref = Xtr, query = Xte, experiment='negative', fun=Pearson)
        a = pValue <= p_value; tmp = dat_cor.iloc[:, a.values[0]]
        writeResult1(dat_cor, output_file, gold_standard_drug, tmp.median(axis=1)[0])

### 把Xtr不显著的基因score设置为0
def f_XPearson():
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    for GSE, cell_line, trTime in mylist:
        method_name = 'cosine'
        gold_standard_drug = 'Alzheimer/goldStandardDrug.tsv'
        output_file  = 'Alzheimer/{}/{}/{}_{}.tsv'.format(GSE, cell_line,  trTime, method_name)
        file_Xtr = 'TCGA/{}/GSE92742_{}/L4/Xtr.h5'.format(cell_line, trTime)
        file_Xte = 'Alzheimer/{}/Xte_ZScore.tsv'.format(GSE)
        Xtr = pd.read_hdf(file_Xtr)
        for i in Xtr.index:
            tmp = Xtr.loc[i,:]; tmp.sort_values(inplace=True); tmp.iloc[500:-500] = 0
            Xtr.loc[i, :] = tmp.loc[Xtr.columns]
        Xte = pd.read_csv(file_Xte, sep='\t', index_col=0, header=0)
        tmp = [i for i in Xtr.columns if i in Xte.columns]
        Xtr = Xtr.loc[:, tmp]; Xte = Xte.loc[:, tmp]
        dat_cor = Pearson(Xtr, Xte)
        pValue = calPvalue1(ref = Xtr, query = Xte, experiment='negative', fun=Pearson)
        a = pValue <= p_value; tmp = dat_cor.iloc[:, a.values[0]]
        writeResult1(dat_cor, output_file, gold_standard_drug, tmp.median(axis=1)[0])

def f_rRank():
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    for GSE, cell_line, trTime in mylist:
        output_file  = 'Alzheimer/{}/{}/{}_rRank.tsv'.format(GSE, cell_line,  trTime)
        file_Xtr = 'TCGA/{}/GSE92742_{}/L4/Xtr.h5'.format(cell_line, trTime)
        file_Xte = 'Alzheimer/{}/Xte_ZScore.tsv'.format(GSE)
        Xtr = pd.read_hdf(file_Xtr)
        for i in Xtr.index:
            tmp = Xtr.loc[i,:]; tmp.sort_values(inplace=True)
            a = sum(tmp <0); b = sum(tmp >=0)
            tmp.iloc[:] = [i for i in range(-a, b)]; Xtr.loc[i, :] = tmp.loc[Xtr.columns]
        Xte = pd.read_csv(file_Xte, sep='\t', index_col=0, header=0)
        tmp = [i for i in Xtr.columns if i in Xte.columns]
        Xtr = Xtr.loc[:, tmp]; Xte = Xte.loc[:, tmp]
        dat_cor = rRank(Xtr, Xte)
        pValue = calPvalue1(ref = Xtr, query = Xte, experiment='negative', fun=rRank)
        a = pValue <= p_value; tmp = dat_cor.iloc[:, a.values[0]]
        writeResult1(dat_cor, output_file, gold_standard_drug, tmp.median(axis=1)[0])

def f_XSum():
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    for GSE, cell_line, trTime in mylist:
        output_file  = 'Alzheimer/{}/{}/{}_XSum.tsv'.format(GSE, cell_line,  trTime)
        file_Xtr = 'TCGA/{}/GSE92742_{}/L4/Xtr.h5'.format(cell_line, trTime)
        file_Xte = 'Alzheimer/{}/Xte_ZScore.tsv'.format(GSE)
        Xtr = pd.read_hdf(file_Xtr)
        for i in Xtr.index:
            tmp = Xtr.loc[i,:]; tmp.sort_values(inplace=True); tmp.iloc[500:-500] = 0
            Xtr.loc[i, :] = tmp.loc[Xtr.columns]
        Xte = pd.read_csv(file_Xte, sep='\t', index_col=0)
        tmp = [i for i in Xtr.columns if i in Xte.columns]
        Xtr = Xtr.loc[:, tmp]; Xte = Xte.loc[:, tmp]
        dat_cor = XSum(Xtr, Xte)
        pValue = calPvalue1(ref = Xtr, query = Xte, experiment='negative', fun=XSum)
        a = pValue <= p_value; tmp = dat_cor.iloc[:, a.values[0]]
        writeResult1(dat_cor, output_file, gold_standard_drug, tmp.median(axis=1)[0])

###########################################################
def KSAndGSEA():
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    methods=[runKS, runGSEA]
    for GSE, cell_line, trTime in mylist:
        for method in methods:
            method_name = method.__name__[3:].lower()
            output_file  = 'Alzheimer/{}/{}/{}_{}.tsv'.format(GSE, cell_line,  trTime, method_name)
            file_Xtr = 'TCGA/{}/GSE92742_{}/L4/Xtr.h5'.format(cell_line, trTime)
            file_Xte = 'Alzheimer/{}/Xte_ZScore.tsv'.format(GSE)
            Xtr = pd.read_hdf(file_Xtr)
            Xte = pd.read_csv(file_Xte, sep='\t', index_col=0, header=0)
            tmp = [i for i in Xtr.columns if i in Xte.columns]
            Xtr = Xtr.loc[:, tmp]; Xte = Xte.loc[:, tmp]
            dat_cor = method(Xtr = Xtr, Xte = Xte, num_genes=100)
            pValue = calPvalue(ref = Xtr, query = Xte, experiment='negative', fun= method)
            a = pValue <= p_value; tmp = dat_cor.iloc[:, a.values[0]]
            writeResult1(dat_cor, output_file, gold_standard_drug, tmp.median(axis=1)[0])

def mergeResults(GSE):
    os.chdir('/home//project/Metric_learning/Alzheimer/{}'.format(GSE))
    dat_all = []
    for cell_line in cell_lines:
        for trTime in ['6H', '24H']:
            filein = '{}/{}_LMNN.tsv'.format(cell_line,  trTime)
            dat = pd.read_csv(filein, header=None, sep='\t')
            dat_all.append(dat)
    dat_all = pd.concat(dat_all, axis=0)
    dat_all.sort_values(by=1,inplace=True)
    dat_all.to_csv('results.tsv', sep='\t', header=False, index=False, float_format='%.3f')


cell_lines = ['MCF7', 'A375', 'PC3',  'HT29', 'A549', 'BT20','VCAP', 'HCC515', 'HEPG2']
GSEs = ['GSE26972']
trTimes = ['6H', '24H']
gold_standard_drug = 'Alzheimer/goldStandardDrug.tsv'
mylist = list(product(GSEs, cell_lines, trTimes))
if __name__ == '__main__':
    #runlfda()
    #f_Pearson()
    KSAndGSEA()
    #f_rRank()
    #f_XSum()
    #f_XPearson()
    #mergeResults(GSE='GSE26972')