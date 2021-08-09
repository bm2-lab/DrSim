import re, os, glob, subprocess, pickle, time
import pandas as pd, numpy as np
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
doPCA = True; rePCA = False; savaPCA = False; reverseExp = True
singleLabel = RunMultiProcess().singleLabel
level = RunMultiProcess().level
sig_id2pert_iname = sigid2iname('')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib, os
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font',family='DejaVu Sans Mono')

sig_pValue = 0.01
def fun2():
    filein = '/home//database/CMap/CMap_FDADrugs.tsv'
    dat = pd.read_csv(filein, sep='\t', header=0)
    dat = dat[dat['Phase'] == 'Launched']
    return dat['Name'].apply(convertDrugName).values


def prePro(cell_line, cancerType):
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    filein = 'TCGA_Predicting_OutCome/nationwidechildrens.org_clinical_drug_{}.txt'.format(cancerType)
    fileout = 'TCGA_Predicting_OutCome/{}/clinical_drug.txt'.format(cell_line)
    dat = pd.read_csv(filein, sep='\t', header=0)
    dat = dat.iloc[2:, :]
    dat_subset = dat[dat['treatment_best_response'].isin(['Complete Response', 'Stable Disease', 'Clinical Progressive Disease', 'Partial Response'])]
    dat_subset['drug'] = dat_subset['pharmaceutical_therapy_drug_name'].apply(lambda x:convertDrugName(x))
    dat_subset['drug'] = dat_subset['drug'].apply(lambda x : x.lower())
    dat_subset.to_csv(fileout, sep='\t', header=True, index=False)

def f_prePro():
    for cell_line in mydict:
        prePro(cell_line, mydict[cell_line].lower())


def preData(cell_line = 'MCF7', trTime='24H'):
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    filein = 'TCGA_Predicting_OutCome/{}/clinical_drug.txt'.format(cell_line)
    drugs = pd.read_csv(filein, sep='\t')['drug'].tolist()
    allSize = 3;  filterDrugs = True
    FDA_Approved = fun2().tolist()
    input_file = 'ZScore/{}/GSE92742_{}/zscoreL4.h5'.format(cell_line, trTime)
    dat = pd.read_hdf(input_file, key='dat')
    a = dat.index
    a = [i for i in a if i in sig_id2pert_iname]
    b = list(map(lambda x: sig_id2pert_iname[x].split('_')[0], a))
    if filterDrugs:
        selected = [True if i in FDA_Approved + drugs and b.count(i) >= allSize else False for i in b]
    else:
        selected = [True if b.count(i) >= allSize else False for i in b]
    a_ = np.array(a)[selected]; b_ = np.array(b)[selected]
    tmp = 'TCGA_Predicting_OutCome/{}/{}/'.format(cell_line, trTime)
    if not os.path.isdir(tmp): os.makedirs(tmp)
    file_Xtr = 'TCGA_Predicting_OutCome/{}/{}/Xtr.h5'.format(cell_line, trTime)
    if os.path.isfile(file_Xtr): os.remove(file_Xtr)
    Xtr = dat.loc[a_, :]
    Xtr.to_hdf(file_Xtr, key = 'dat')

def f_preData():
    for cell_line in mydict:
        for trTime in ['6H', '24H']:
            preData(cell_line, trTime)

def ZScorequery(cell_line, cancerType):
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    treat, control = getExp(cancerType)
    treat_ = np.log2(treat + 1); control_ = np.log2(control + 1)
    a = np.median(control_, axis=1, keepdims=True)
    b = stats.median_absolute_deviation(control_, axis=1).reshape(-1, 1)
    if b.shape[1] == 1:
        b[b==0] = 1   ##防止为0
    else:
        b[b==0] = np.median(b)   ##防止为0
    result = (treat_ - a) / 1.4826 / b; result.index.name = ''; result = result.T
    result.columns = ['Entrez_' + str(i) for  i in result.columns]
    file_Xte = '/home//project/Metric_learning/TCGA_Predicting_OutCome/{}/Xte_ZScore.tsv'.format(cell_line)
    result.to_csv(file_Xte, sep='\t', header=True, index = True)

def f_ZScorequery():
    for cell_line in mydict:
        ZScorequery(cell_line, mydict[cell_line])


def preXtr_Xte(cell_line, trTime):
    dat_subset = pd.read_csv('TCGA_Predicting_OutCome/{}/clinical_drug.txt'.format(cell_line), sep='\t')
    sample_list = dat_subset['bcr_patient_barcode'].tolist()
    file_Xtr = 'TCGA_Predicting_OutCome/{}/{}/Xtr.h5'.format(cell_line, trTime)
    Xtr = pd.read_hdf(file_Xtr, key='dat')    
    file_Xte = 'TCGA_Predicting_OutCome/{}/Xte_ZScore.tsv'.format(cell_line, trTime)
    Xte = pd.read_csv(file_Xte, sep= '\t', header=0, index_col = 0)
    temp = Xte.index.isin(sample_list)
    Xte = Xte.loc[temp, :]
    tmp = [i for i in Xtr.columns if i in Xte.columns]
    Xtr = Xtr.loc[:, tmp]; Xte = Xte.loc[:, tmp]
    return Xtr, Xte


def printResult(cell_line, dat_cor, pValues):
    response = 0; nonresponse = 0; total_response = 0; total_nonresponse = 0
    dat_subset = pd.read_csv('TCGA_Predicting_OutCome/{}/clinical_drug.txt'.format(cell_line), sep='\t')
    for i, j, k in zip(dat_subset['bcr_patient_barcode'], dat_subset['drug'], dat_subset['treatment_best_response']):
        if i in dat_cor.index and j in dat_cor.columns:  ### 药物必须同时在ref和clinical里面
            value = pValues.loc[i, j]
            if value <= sig_pValue:
                if k in ['Complete Response', 'Partial Response']:
                    response += 1
                else:
                    nonresponse += 1
            if k in ['Complete Response', 'Partial Response']:
                total_response += 1
            else:
                total_nonresponse += 1
    print (response, nonresponse, total_response, total_nonresponse)
    return response, nonresponse, total_response, total_nonresponse

def printResult1(cell_line, dat_cor, sigValue):
    response = 0; nonresponse = 0; total_response = 0; total_nonresponse = 0
    dat_subset = pd.read_csv('TCGA_Predicting_OutCome/{}/clinical_drug.txt'.format(cell_line), sep='\t')
    for i, j, k in zip(dat_subset['bcr_patient_barcode'], dat_subset['drug'], dat_subset['treatment_best_response']):
        if i in dat_cor.index and j in dat_cor.columns:  ### 药物必须同时在ref和clinical里面
            value = dat_cor.loc[i, j]
            if value <= sigValue[i]:
                if k in ['Complete Response', 'Partial Response']:
                    response += 1
                else:
                    nonresponse += 1
            if k in ['Complete Response', 'Partial Response']:
                total_response += 1
            else:
                total_nonresponse += 1
    print (response, nonresponse, total_response, total_nonresponse)
    return response, nonresponse, total_response, total_nonresponse

def lfda(cell_line, trTime):
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    Xtr, Xte = preXtr_Xte(cell_line, trTime=trTime)
    output_file  = 'TCGA_Predicting_OutCome/{}/{}/LMNN.tsv'.format(cell_line, trTime)
    file_PCA = 'TCGA_Predicting_OutCome/{}/{}/PCA.pkl'.format(cell_line, trTime)
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

    a = pd.DataFrame(Xtr_pca_lmnn, index=pert_iname)
    ref = a.groupby(pert_iname).median()
    query = pd.DataFrame(data = Xte_pca_lmnn, index = Xte.index)
    dat_cor = calCosine(Xtr = ref, Xte = query)
    if reverseExp: dat_cor = -dat_cor
    pValue_list = []
    for i in range(query.shape[0]):
        tmp = query.iloc[i:i+1]
        pValue = calPvalue(ref, tmp, experiment= experiment, fun=calCosine)
        pValue_list.append(pValue)
    pValues = pd.concat(pValue_list, axis=0)
    print (cell_line, trTime)
    response, nonresponse, _, _ = printResult(cell_line, dat_cor, pValues)
    return response, nonresponse

def lfda1(cell_line, trTime):   ### 返回PCA的结果
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    Xtr, Xte = preXtr_Xte(cell_line, trTime=trTime)
    output_file  = 'TCGA_Predicting_OutCome/{}/{}/LMNN.tsv'.format(cell_line, trTime)
    file_PCA = 'TCGA_Predicting_OutCome/{}/{}/PCA.pkl'.format(cell_line, trTime)
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

    a = pd.DataFrame(Xtr_pca, index=pert_iname)
    ref = a.groupby(pert_iname).median()
    query = pd.DataFrame(data = Xte_pca, index = Xte.index)
    dat_cor = calCosine(Xtr = ref, Xte = query)
    if reverseExp: dat_cor = -dat_cor 
    pValue_list = []
    for i in range(query.shape[0]):
        tmp = query.iloc[i:i+1]
        pValue = calPvalue(ref, tmp, experiment= experiment, fun=calCosine)
        pValue_list.append(pValue)
    pValues = pd.concat(pValue_list, axis=0)
    response, nonresponse, _, _ = printResult(cell_line, dat_cor, pValues)
    return response, nonresponse



def f_lfda():
    for cell_line in mydict:
        for trTime in ['6H', '24H']:
            lfda(cell_line, trTime)
            #lfda1(cell_line, trTime)
######
def f_Pearson(cell_line, trTime):
    output_file  = 'TCGA_Predicting_OutCome/{}/{}/cosine.tsv'.format(cell_line, trTime)
    Xtr, Xte = preXtr_Xte(cell_line, trTime)
    pert_iname = [sig_id2pert_iname[i].split('_')[0] for i in Xtr.index]
    Xtr = Xtr.groupby(pert_iname, axis=0).median()
    dat_cor = Pearson(Xtr, Xte)
    sigValue = dat_cor.quantile(q = sig_pValue, axis=1).to_dict()
    dat_cor.to_csv(output_file, sep='\t', header=True, index= True)
    printResult1(cell_line, dat_cor, sigValue)

def ff_Pearson():
    for cell_line in mydict:
        for trTime in ['6H']: f_Pearson(cell_line, trTime)


### 把Xtr不显著的基因score设置为0
def f_XPearson(cell_line, trTime):
    output_file  = 'TCGA_Predicting_OutCome/{}/{}/Xcosine.tsv'.format(cell_line, trTime)
    Xtr, Xte = preXtr_Xte(cell_line, trTime)
    pert_iname = [sig_id2pert_iname[i].split('_')[0] for i in Xtr.index]
    Xtr = Xtr.groupby(pert_iname, axis=0).median()
    for i in Xtr.index:
        tmp = Xtr.loc[i,:]; tmp.sort_values(inplace=True); tmp.iloc[500:-500] = 0
        Xtr.loc[i, :] = tmp.loc[Xtr.columns]
    dat_cor = Pearson(Xtr, Xte)
    sigValue = dat_cor.quantile(q = sig_pValue, axis=1).to_dict()
    dat_cor.to_csv(output_file, sep='\t', header=True, index= True)
    printResult1(cell_line, dat_cor, sigValue)


def ff_XPearson():
    for cell_line in mydict:
        for trTime in ['6H', '24H']: f_XPearson(cell_line, trTime)

def f_rRank(cell_line, trTime):
    output_file  = 'TCGA_Predicting_OutCome/{}/{}/rRank.tsv'.format(cell_line, trTime)
    Xtr, Xte = preXtr_Xte(cell_line, trTime)
    pert_iname = [sig_id2pert_iname[i].split('_')[0] for i in Xtr.index]
    Xtr = Xtr.groupby(pert_iname, axis=0).median()
    for i in Xtr.index:
        tmp = Xtr.loc[i,:]; tmp.sort_values(inplace=True)
        a = sum(tmp <0); b = sum(tmp >=0)
        tmp.iloc[:] = [i for i in range(-a, b)]; Xtr.loc[i, :] = tmp.loc[Xtr.columns]
    Xte_list = [Xte.iloc[i:i+1,:] for i in range(Xte.shape[0])]
    results = [rRank(Xtr, i) for i in Xte_list]
    dat_cor = pd.concat(results, axis=0)
    sigValue = dat_cor.quantile(q = sig_pValue, axis=1).to_dict()
    dat_cor.to_csv(output_file, sep='\t', header=True, index= True)
    printResult1(cell_line, dat_cor, sigValue)

def ff_rRank():
    for cell_line in mydict:
        for trTime in ['6H', '24H']:
            f_rRank(cell_line, trTime)

def f_XSum(cell_line, trTime):
    output_file  = 'TCGA_Predicting_OutCome/{}/{}/XSum.tsv'.format(cell_line, trTime)
    Xtr, Xte = preXtr_Xte(cell_line, trTime)
    pert_iname = [sig_id2pert_iname[i].split('_')[0] for i in Xtr.index]
    Xtr = Xtr.groupby(pert_iname, axis=0).median()
    for i in Xtr.index:
        tmp = Xtr.loc[i,:]; tmp.sort_values(inplace=True); tmp.iloc[500:-500] = 0
        Xtr.loc[i, :] = tmp.loc[Xtr.columns]
    Xte_list = [Xte.iloc[i:i+1,:] for i in range(Xte.shape[0])]
    results  = [XSum(Xtr, i) for i in Xte_list]
    dat_cor = pd.concat(results, axis=0, sort=False)
    sigValue = dat_cor.quantile(q = sig_pValue, axis=1).to_dict()
    dat_cor.to_csv(output_file, sep='\t', header=True, index= True)
    printResult1(cell_line, dat_cor, sigValue)

def ff_XSum():
    for cell_line in mydict:
        for trTime in ['6H', '24H']: f_XSum(cell_line, trTime)


##############  KS and  GSEA
def KSAndGSEA(cell_line, trTime):
    Xtr, Xte = preXtr_Xte(cell_line, trTime)
    pert_iname = [sig_id2pert_iname[i].split('_')[0] for i in Xtr.index]
    Xtr = Xtr.groupby(pert_iname, axis=0).median()
    methods=[runGSEA]
    for method in methods:
        method_name = method.__name__[3:].lower()
        output_file  = 'TCGA_Predicting_OutCome/{}/{}/{}.tsv'.format(cell_line, trTime, method_name)
        dat_cor = method(Xtr = Xtr, Xte = Xte, num_genes=100)
        sigValue = dat_cor.quantile(q = sig_pValue, axis=1).to_dict()
        dat_cor.to_csv(output_file, sep='\t', header=True, index= True)
        printResult1(cell_line, dat_cor, sigValue)

def f_KSAndGSEA():
    for cell_line in mydict:
        for trTime in ['6H', '24H']: KSAndGSEA(cell_line, trTime)

methods = ['LMNN', 'pca', 'cosine', 'Xcosine', 'rRank', 'XSum', 'ks', 'gsea']
def checkResults(method, sig_pValue=.05, trTime = '6H', cell_line = 'MCF7'):
    input_file  = 'TCGA_Predicting_OutCome/{}/{}/{}.tsv'.format(cell_line, trTime, method)
    dat_cor = pd.read_csv(input_file, sep='\t', header=0, index_col=0)
    sigValue = dat_cor.quantile(q = sig_pValue, axis=1).to_dict()
    return printResult1(cell_line, dat_cor, sigValue)

def f_checkResults():
    os.chdir('/home//project/Metric_learning/')
    a, b, c, d = [], [], [], [] ### method, time, result  cell_line
    for cell_line in mydict:
        for method in methods:
            for trTime in ['6H', '24H']:
                if method == 'LMNN':
                    response, nonresponse = lfda(cell_line, trTime)
                elif method == 'pca':
                    response, nonresponse = lfda1(cell_line, trTime)
                else:
                    response, nonresponse, _, _ = checkResults(method, sig_pValue=sig_pValue, trTime=trTime, cell_line=cell_line)
                if response + nonresponse == 0:
                    acc = 0.5
                else:
                    acc = response / (response + nonresponse)
                a.append(method); b.append(trTime); c.append(acc); d.append(cell_line)

    dat = pd.DataFrame({'methods':a, 'trTime':b, 'value':c, 'cell': d})
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.stripplot(dat['methods'], dat['value'], linewidth=0, color= 'black', edgecolor=None, order=dat.groupby('methods')['value'].mean().sort_values(ascending=False).index)
    sns.boxplot(dat['methods'], dat['value'], fliersize=0, linewidth=1, order=dat.groupby('methods')['value'].mean().sort_values(ascending=False).index)
    fig.savefig("Plot/Drug_disease/invivo-with.pdf", transparent=True)    
    dat.to_csv('Plot/Drug_disease/invivo-with.tsv', sep='\t', index=False, float_format='%.4f')   


"""
MCF7, BT20       Breast_Cancer      BRCA   
A375             Melanoma           SKCM
PC3,  VCAP       Prostate           PRAD
HT29             Colon_Cancer       COAD
A549  HCC515     Lung_Cancer        LUAD   
HEPG2            Hepatocellular     LIHC
数据下载TCGA Exploration模块
"""
mydict = {'A549':'LUAD', 'HCC515': 'LUAD', 'MCF7':'BRCA'}


if __name__ == '__main__':
    print ('hello, world')
    #f_prePro()
    #f_preData()
    #f_ZScorequery()
    #f_lfda()
    ff_Pearson()
    #ff_XPearson()
    #ff_rRank()
    #ff_XSum()
    #f_KSAndGSEA()
    #f_checkResults()
    #f_checkResults1()

