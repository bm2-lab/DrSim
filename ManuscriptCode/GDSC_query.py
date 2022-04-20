import os,sys,re,glob,subprocess,glob
from collections import defaultdict
import numpy as np,pandas as pd
from multiprocessing import Pool
import string, mygene
from util import RunMultiProcess, convertDrugName
from scipy.stats import stats
### 生成癌症细胞系signature

#### script to generate cell line query signature
#### cancer cell line downloaded from ccle, control downloaded from GTEx project 
doMultiProcess = RunMultiProcess()
def getGTEx(cell_line = 'MCF7'):
    mydict = {'MCF7':'Breast', 'A375':'Skin','PC3':'Prostate', 'HT29':'Colon','YAPC':'Pancreas',
              'HELA':'Cervix Uteri', 'A549':'Lung','BT20':'Breast','VCAP':'Prostate',
              'HCC515':'Lung','HEPG2':'Liver'}
    os.chdir('/home//database/GTEx')
    pheno = pd.read_table('GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt')
    pheno = pheno.iloc[np.array((pheno['SMTS']==mydict[cell_line])
            & (pheno['SMATSSCR'] <= 1) & (pheno['SMRIN'] >= 6)),]
    exp = pd.read_table('GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct',skiprows=2,header=0)
    pheno = ['Description'] +  [i for i in pheno.SAMPID if i in exp.columns]
    subset = exp.loc[:,pheno]
    subset.to_csv('raw_{}.exp'.format(cell_line),header=True,sep='\t',index=False)

def getCCLE(cell_line = 'MCF7'):
    os.chdir('/home//database/CCLE')
    dat = pd.read_table('CCLE_RNAseq_genes_counts_20180929.gct',skiprows=2,header=0)
    columns = ['Description'] + dat.columns[dat.columns.str.contains(cell_line)].tolist()
    dat = dat[columns]
    dat.to_csv('raw_{}.exp'.format(cell_line),header=True,index=False,sep='\t')

def IDconvert():
    for cell_line in cell_lines:
        cmd = 'Rscript   /home//project/Personal_Drug/IDconversion.r ' \
              '{} SYMBOL raw_{}.exp'.format(cell_line,cell_line)
        subprocess.call(cmd,shell=True)


### 不是CCLE的话改变os.chdir的地址
def generateExp(cell_line = 'MCF7', path = '/home//database/CCLE'):
    mydict1 = {'MCF7':'Breast', 'A375':'Skin','PC3':'Prostate', 'HT29':'Colon','YAPC':'Pancreas',
              'HELA':'Cervix Uteri', 'A549':'Lung','BT20':'Breast','VCAP':'Prostate',
              'HCC515':'Lung','HEPG2':'Liver'}
    mydict2 = {'MCF7':['LCSET-7849'], 'A375':['LCSET-6441'],'PC3':['LCSET-3205','LCSET-3098'],'HT29':['LCSET-6806'],
               'YAPC':['LCSET-4813'],'HELA':['LCSET-3098','LCSET-3205','LCSET-10341'], 'A549':['LCSET-4815'],
               'BT20':['LCSET-7849'],'VCAP':['LCSET-3205','LCSET-3098'],'HCC515':['LCSET-4815'],'HEPG2':['LCSET-4813']}

    keep_only_protein_coding = False
    mg = mygene.MyGeneInfo()
    def f(entrezs):
        results = []
        for i in mg.getgenes(entrezs, species='human', fields=['type_of_gene']):
            try:
                results.append(i['type_of_gene'])
            except:
                results.append('unknown')
        return results

    os.chdir('/home//project/Personal_Drug/CCLE/{}'.format(cell_line))
    cancer_file = '{}/{}.exp'.format(path,cell_line)
    GTEx = pd.read_csv('/home//database/GTEx/{}.exp'.format(cell_line), sep='\t')
    pheno = pd.read_csv('/home//database/GTEx/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt', sep='\t')
    pheno = pheno.iloc[np.array(pheno['SMTS'] == mydict1[cell_line]),]
    #ids = ['ENTREZID'] + pheno['SAMPID'][pheno['SMGEBTCH'].apply(lambda x: True if x in mydict2[cell_line] else False)].tolist()  ### 取一个batch的正常对照
    ids = ['ENTREZID'] + pheno['SAMPID'][pheno['SMGEBTCH'].apply(lambda x: True)].tolist()  ### 取全部
    ids = [i for i in ids if i in GTEx.columns]
    GTEx = GTEx.loc[:,ids]
    CCLE = pd.read_csv(cancer_file, sep='\t')
    n_control = GTEx.shape[1] - 1
    n_treat = CCLE.shape[1] - 1
    GTEx.drop_duplicates(subset='ENTREZID',keep=False,inplace=True)  ##去除重复
    CCLE.drop_duplicates(subset='ENTREZID',keep=False,inplace=True)  ##去除重复
    data = pd.merge(CCLE, GTEx, on='ENTREZID')                       ##会减少基因
    if keep_only_protein_coding:
        data['genetype'] = f(data['ENTREZID'].tolist())
    else:
        data['genetype'] = 'protein-coding'
    data = data.iloc[np.array(data['genetype'] == 'protein-coding'),:]  ## 不过滤不影响
    data.drop(labels = ['genetype'],axis=1,inplace=True)
    treat = data.iloc[:,:n_treat+1]
    control_index = [data.columns[0]] + data.columns[-n_control:].tolist()
    n = 100
    #if len(control_index) >= n : control_index = control_index[:n]
    control = data.loc[:,control_index]
    treat.to_csv('{}_treat.exp'.format(cell_line),header=True,index=False,sep='\t')
    control.to_csv('{}_control.exp'.format(cell_line), header=True, index=False, sep='\t')

def f_generateExp():
    for cell_line in doMultiProcess.cell_lines:
        generateExp(cell_line)

def Normalized(cell_line):
    os.chdir('/home//project/Personal_Drug/CCLE/{}'.format(cell_line))
    cmd = 'Rscript ../../normalize.r  {} {}_treat.exp  {}_control.exp'.format(cell_line,cell_line,cell_line)
    subprocess.call(cmd,shell=True)
    treat_file = '{}_log_normalized_treat.exp'.format(cell_line)
    control_file = '{}_log_normalized_control.exp'.format(cell_line)
    treat = pd.read_csv(treat_file, sep='\t', index_col=0)
    control = pd.read_csv(control_file, sep='\t', index_col=0)
    rs = np.random.RandomState(seed = 2020)
    control = control.applymap(lambda x : rs.random(1)[0] / 10 if x <=0 else x)
    treat = treat.applymap(lambda x : rs.random(1)[0] / 10 if x <=0 else x)
    treat.to_csv(treat_file, sep='\t', header=True, index=True)
    control.to_csv(control_file, sep='\t', header=True, index=True)
    

def f_Normalized():
    doMultiProcess.myPool(Normalized, doMultiProcess.cell_lines, 10)

## 把有小于0的行去除
def deleteZeros(X):
    temp = np.sum(X<=0, axis=1)
    index = temp[temp ==0].index
    return X.loc[index, :]

###############################################################
### 对ccle细胞系计算相对于正常的zscore
def CCLE_ZScorequery():
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    path = '/home//project/Personal_Drug/CCLE'
    for cell_line in doMultiProcess.cell_lines:
        treat_file = '{}/{}/{}_log_normalized_treat.exp'.format(path, cell_line, cell_line)
        control_file = '{}/{}/{}_log_normalized_control.exp'.format(path, cell_line, cell_line)
        treat = pd.read_csv(treat_file, sep='\t', index_col=0)
        control = pd.read_csv(control_file, sep='\t', index_col=0)
        tmp = [i for i in control.index if i in treat.index]
        control = control.loc[tmp, :]; treat = treat.loc[tmp, :]
        a = np.median(control, axis=1, keepdims=True)
        b = stats.median_absolute_deviation(control, axis=1).reshape(-1, 1)
        b[b==0] = np.median(b)   ##防止为0
        result = (treat - a) / 1.4826 / b; result = result.T
        result.columns = ['Entrez_' + str(i) for  i in result.columns]
        file_Xte = '/home//project/Personal_Drug/CCLE/{}/Xte_ZScore.tsv'.format(cell_line)
        result.to_csv(file_Xte, sep='\t', header=True, index = True)


if __name__ == '__main__':
    print ('hello, world')
    #f_generateExp()
    f_Normalized()
    CCLE_ZScorequery()
