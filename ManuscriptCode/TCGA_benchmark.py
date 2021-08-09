import re, os, glob, subprocess, pickle, time, warnings
import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder
from util import calCosine, calPearson, calSpearman, RunMultiProcess, sigid2iname
import metric_learn, dml
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from CMapKS import runKS
from CMapGSEA import runGSEA
from CMapPvalue import calPvalue, calPvalue1
from GDSC_benchmark import Pearson, rRank, XSum
warnings.filterwarnings('ignore')
doPCA = True; rePCA = False; savaPCA = True; reverseExp = True
singleLabel = RunMultiProcess().singleLabel
level = RunMultiProcess().level
sig_id2pert_iname = sigid2iname('')

### 求质心
cell_lines = ['MCF7', 'A549', 'HCC515', 'PC3', 'VCAP']

p_value = 0.01
def getMOA():
    label_file1 = '/home/wzt/project/Metric_learning/MOA1SigInfo.tsv'
    label_file2 = '/home/wzt/project/Metric_learning/ATCSigInfo.tsv'
    pert_iname2MOA = {}; pert_iname2ATC = {}; 
    with open(label_file1, 'r') as fin:
        fin.readline()
        for line in fin:
            lines = line.strip().split('\t')
            pert_iname2MOA[lines[2]] = lines[6]
    with open(label_file2, 'r') as fin:
        fin.readline()
        for line in fin:
            lines = line.strip().split('\t')
            pert_iname2ATC[lines[2]] = lines[6]
    return pert_iname2MOA, pert_iname2ATC

def writeResult(dat_cor, output_file, gold_standard_drug, FDA_drug):
    dat_cor = dat_cor.T
    dat_cor.columns = ['Score']
    dat_cor.sort_values(by='Score', ascending=True, inplace=True)
    mydict = {}
    with open(gold_standard_drug, 'r') as fin:
        for line in fin:
            lines = line.strip().split('\t')
            mydict[lines[0]] = lines[1]
    with open(FDA_drug, 'r') as fin:
        for line in fin:
            lines = line.strip().split('\t')
            mydict[lines[0]] = 'FDA'
    temp = []
    for i in dat_cor.index:
        if i in mydict:
            if mydict[i] == 'vivo': temp.append('YES')
            elif mydict[i] == 'FDA': temp.append('FDA')
            else: temp.append('NO')
        else: temp.append('NOT_Check')
    dat_cor['Validated'] = temp
    dat_cor['rank'] = np.arange(1, dat_cor.shape[0] + 1)
    pert_iname2MOA, pert_iname2ATC = getMOA()
    dat_cor['MOA'] =  dat_cor.index.map(lambda x : pert_iname2MOA.get(x, 'unKnown'))
    if dat_cor.shape[0] >=10: dat_cor = dat_cor.iloc[:10, :]
    dat_cor.to_csv(output_file, sep='\t', header=True, index=True, float_format= '%.4f')

def lfda(X):
    GSE, cell_line, trTime = X
    basepath = '/home/wzt/project/Metric_learning'; os.chdir(basepath)
    output_file1  = 'TCGA/{}/{}_{}/{}/LMNN.tsv'.format(cell_line, GSE, trTime, level)
    output_file2  = 'TCGA/{}/{}_{}/{}/pca.tsv'.format(cell_line, GSE, trTime, level)
    file_Xtr = 'TCGA/{}/{}_{}/{}/Xtr.h5'.format(cell_line, GSE, trTime, level)
    file_Xte = 'TCGA/{}/Xte_ZScore.tsv'.format(cell_line)
    file_PCA = 'TCGA/{}/{}_{}/{}/PCA.pkl'.format(cell_line, GSE, trTime, level)
    gold_standard_drug = 'TCGA/{}/goldStandardDrug.tsv'.format(cell_line)
    FDA_drug = 'TCGA/{}/FDAapproved.txt'.format(cell_line)
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
    a = pValue <= p_value; tmp = dat_cor.iloc[:, a.values[0]]
    writeResult(tmp, output_file2, gold_standard_drug, FDA_drug)
    
    a = pd.DataFrame(Xtr_pca_lmnn, index=pert_iname)
    ref = a.groupby(pert_iname).median()
    query = pd.DataFrame(data = Xte_pca_lmnn, index = Xte.index)
    dat_cor = calCosine(Xtr = ref, Xte = query)
    if reverseExp: dat_cor = -dat_cor
    pValue = calPvalue(ref, query, experiment=experiment, fun=calCosine)
    a = pValue <= p_value; tmp = dat_cor.iloc[:, a.values[0]]
    writeResult(tmp, output_file1, gold_standard_drug, FDA_drug)
    print ('{}\t{}\t{}\tcompleted\n'.format(GSE, cell_line, trTime))

def runlfda():
    for trTime in ['6H', '24H']:
        for cell_line in cell_lines:
            lfda(('GSE92742', cell_line, trTime))


methods = ['LMNN', 'pca', 'cosine', 'ks', 'gsea', 'rRank', 'XSum', 'Xcosine']
def mergeResults():
    for cell_line in ['MCF7', 'A549', 'HCC515', 'PC3', 'VCAP']:
        os.chdir('/home/wzt/project/Metric_learning/TCGA/{}'.format(cell_line))
        for method in methods:
            filein1 = 'GSE92742_6H/L4/{}.tsv'.format(method)
            filein2 = 'GSE92742_24H/L4/{}.tsv'.format(method)
            dat1 = pd.read_csv(filein1, sep='\t')
            dat2 = pd.read_csv(filein2, sep='\t')
            dat1.columns = ['drug','score', 'Validated', 'rank', 'MOA']
            dat2.columns = ['drug','score', 'Validated', 'rank', 'MOA']
            for i in range(dat2.shape[0]):
                tmp = dat2.iloc[i:i+1]
                drug = tmp.iloc[0, 0]
                if drug in dat1['drug'].tolist():
                    dat1.loc[dat1['drug'] == drug, 'score'] = (tmp.iloc[0, 1] + dat1.loc[dat1['drug'] == drug, 'score']) / 2
                else:
                    dat1 = pd.concat([dat1, tmp])
            dat1.sort_values('score', ascending=True, inplace=True)
            dat1['rank'] = [i for i in range(1, dat1.shape[0] + 1)]
            dat1.to_csv('GSE92742/{}.tsv'.format(method), sep='\t', index=False)

if __name__ == '__main__':
    print ('hello, world')
    #runlfda()
    mergeResults()