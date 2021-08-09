import os,sys,re,glob,subprocess,glob
from collections import defaultdict
import numpy as np,pandas as pd
from multiprocessing import Pool
import string, mygene
from util import RunMultiProcess, convertDrugName
from scipy.stats import stats

### 用TCGA的数据生成pancancer的query
def getExp(CC):
    tumor_list = []; normal_list = []
    tumor_names = []; normal_names = []
    path = '/home/wzt/project/GeneFusion/SNVIndel'
    filein = '{}/{}/{}.tsv'.format(path, CC, CC)
    dat = pd.read_csv(filein,sep='\t',header=0)
    temp1 = (dat['Data Category'] == 'Transcriptome Profiling') & (dat['Sample Type'] == 'Primary Tumor')
    temp2 = (dat['Data Category'] == 'Transcriptome Profiling') & (dat['Sample Type'] == 'Solid Tissue Normal')
    tumor = dat[temp1]; normal = dat[temp2];
    for i in range(tumor.shape[0]):
        lines = tumor.iloc[i, :]
        if lines[1].endswith('gz'): lines[1] = lines[1][:-3]
        file = '{}/{}/data/'.format(path, CC) + lines[0] + '/' + lines[1]
        if os.path.isfile(file): tumor_list.append(file); tumor_names.append(lines[5])
    for i in range(normal.shape[0]):
        lines = normal.iloc[i, :]
        if lines[1].endswith('gz'): lines[1] = lines[1][:-3]
        file = '{}/{}/data/'.format(path, CC) + lines[0] + '/' + lines[1]
        if os.path.isfile(file): normal_list.append(file); normal_names.append(lines[5])
    all_tumor = []; all_normal = []
    for file in tumor_list[:]:
        dat = pd.read_csv(file, index_col=0, sep= '\t', header=None)
        all_tumor.append(dat)
    tumor_dat = pd.concat(all_tumor,axis=1)
    tumor_dat.columns = tumor_names
    for file in normal_list[:]:
        dat = pd.read_csv(file, index_col=0, sep= '\t', header=None)
        all_normal.append(dat)
    normal_dat = pd.concat(all_normal,axis=1)
    normal_dat.columns = normal_names
    normal_dat.index =[i.strip().split('.')[0] for i in normal_dat.index]
    tumor_dat.index =[i.strip().split('.')[0] for i in tumor_dat.index]
    ID = pd.read_csv('/home/wzt/project/Metric_learning/TCGA/ensembl2entrez.tsv', sep='\t')
    ID.drop_duplicates(subset='ENSEMBL', inplace=True)
    ID.drop_duplicates(subset='ENTREZID', inplace=True)
    tumor_dat = pd.merge(tumor_dat, ID, left_index= True, right_on='ENSEMBL')
    normal_dat = pd.merge(normal_dat, ID, left_index= True, right_on='ENSEMBL')
    tumor_dat.index = tumor_dat['ENTREZID']
    tumor_dat.drop(labels=['ENSEMBL', 'ENTREZID'], axis=1, inplace=True)
    normal_dat.index = normal_dat['ENTREZID']
    normal_dat.drop(labels=['ENSEMBL', 'ENTREZID'], axis=1, inplace=True)
    return tumor_dat, normal_dat

def ZScorequery(cell_line, cancerType):
    basepath = '/home/wzt/project/Metric_learning'; os.chdir(basepath)
    path = '/home/wzt/project/GeneFusion/SNVIndel'
    treat, control = getExp(cancerType)
    print (cell_line, control.shape[1])
    if control.shape[1] <= 10: return
    treat_ = np.log2(treat + 1); control_ = np.log2(control + 1)
    treat_ = np.median(treat_, axis=1)
    treat_ = pd.DataFrame(treat_, index = treat.index, columns=['treat'])
    a = np.median(control_, axis=1, keepdims=True)
    b = stats.median_absolute_deviation(control_, axis=1).reshape(-1, 1)
    b[b==0] = np.median(b)   ##防止为0
    result = (treat_ - a) / 1.4826 / b; result = result.T
    result.columns = ['Entrez_' + str(i) for  i in result.columns]
    file_Xte = '/home/wzt/project/Metric_learning/TCGA/{}/Xte_ZScore.tsv'.format(cell_line)
    result.to_csv(file_Xte, sep='\t', header=True, index = True)


"""
MCF7   BT20  Breast_Cancer     A375  Melanoma  PC3  VCAP  Prostate
HT29   Colon_Cancer   A549  HCC515  Lung_Cancer    HEPG2  Hepatocellular_Carcinoma 
"""


mydict = {'MCF7':'BRCA', 'A375':'SKCM', 'PC3':'PRAD', 'HT29': 'COAD', 
        'A549':'LUAD', 'BT20':'BRCA','VCAP':'PRAD', 
        'HCC515':'LUAD', 'HEPG2':'LIHC'}



def f_ZScorequery():
    for cell_line in mydict:
        ZScorequery(cell_line, mydict[cell_line])

if __name__ == '__main__':
    f_ZScorequery()
