#coding=utf-8
import string, os, sys, glob, subprocess
import pandas as pd, numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool
from itertools import product, chain
from scipy import stats
import torch; from tqdm import tqdm
from collections import defaultdict

def convertDrugName(name):
    temp = []
    for i in str(name):
        if i in string.ascii_letters or i in string.digits or i =='-':
            temp.append(i)
    return ''.join(temp)

def getDrugiDose(x):
    x = x.strip().split()
    if x[1] == 'µM' or x[1] == 'um' or x[1] == 'µL':
        return int('{:.0f}'.format(float(x[0])*1000))
    elif x[1] == 'nM':
        return int('{:.0f}'.format(float(x[0])))

# LINCS id to drug name or MOA class dict
def sigid2iname(MOA = ''):
    label_file = '/home//project/Metric_learning/{}SigInfo.tsv'.format(MOA)
    sig_id2pert_iname = {}
    with open(label_file, 'r') as fin:
        fin.readline()
        for line in fin:
            lines = line.strip().split('\t')
            distil_ids = lines[5].split('|')
            if MOA: label = lines[6]  + '_' + lines[4]     ### iname, idose, MOA或ATC
            else:   label = lines[2]  + '_' + lines[4]
            sig_id2pert_iname[lines[0]] = label
            for i in distil_ids:
                sig_id2pert_iname[i] = label
    return sig_id2pert_iname

## drug name to MOA class dict
def drug2MOA(MOA = 'MOA'):
    label_file = '/home//project/Metric_learning/{}SigInfo.tsv'.format(MOA)
    drugtoMOA ={}
    with open(label_file, 'r') as fin:
        fin.readline()
        for line in fin:
            lines = line.strip().split('\t')
            drugtoMOA[lines[2]] = lines[6]
    return drugtoMOA

def calCosine(Xtr, Xte):
    dat_cor = pd.DataFrame(cosine_similarity(Xte,Xtr))  ###行是Xte, 列是Xtr
    dat_cor.columns = Xtr.index
    dat_cor.index = Xte.index
    return dat_cor

# def calPearson(Xtr, Xte):
#     dat = pd.concat((Xtr, Xte), axis=0)
#     samples = Xtr.index.tolist() + Xte.index.tolist()
#     dat_cor = np.corrcoef(dat)
#     dat_cor = pd.DataFrame(dat_cor, index=samples, columns=samples)
#     dat_cor = dat_cor.loc[Xte.index, Xtr.index]   ####行是Xte, 列是Xtr
#     return dat_cor

def calPearson(Xtr, Xte):    ###   (x-xmean) *(y - ymean) / ((x-xmean)**2)**.5 * ((y-ymean)**2)**.5
    Xtr_index = Xtr.index; Xte_index = Xte.index
    Xtr_tensor = torch.from_numpy(Xtr.values); Xtr_tensor = Xtr_tensor.double()
    Xte_tensor = torch.from_numpy(Xte.values); Xte_tensor = Xte_tensor.double()
    Xtr_tensor_mean = Xtr_tensor.mean(dim = 1, keepdim= True)
    Xte_tensor_mean = Xte_tensor.mean(dim = 1, keepdim= True)
    Xtr_submean = Xtr_tensor - Xtr_tensor_mean
    Xte_submean = Xte_tensor - Xte_tensor_mean
    a = torch.mm(Xtr_submean, Xte_submean.T)
    b = torch.sqrt(torch.sum(Xtr_submean ** 2, axis=1, keepdim=True))
    c = torch.sqrt(torch.sum(Xte_submean ** 2, axis=1, keepdim=True))
    cor = a / (b @ c.T); cor = cor.T; cor = cor.numpy()
    dat_cor = pd.DataFrame(cor, index=Xte_index, columns=Xtr_index)
    return dat_cor

def calSpearman(Xtr, Xte):    ###   (x-xmean) *(y - ymean) / ((x-xmean)**2)**.5 * ((y-ymean)**2)**.5
    Xtr = Xtr.rank(axis=1); Xte = Xte.rank(axis=1)
    Xtr_index = Xtr.index; Xte_index = Xte.index
    Xtr_tensor = torch.from_numpy(Xtr.values)
    Xte_tensor = torch.from_numpy(Xte.values)
    Xtr_tensor_mean = Xtr_tensor.mean(dim = 1, keepdim= True)
    Xte_tensor_mean = Xte_tensor.mean(dim = 1, keepdim= True)
    Xtr_submean = Xtr_tensor - Xtr_tensor_mean
    Xte_submean = Xte_tensor - Xte_tensor_mean
    a = torch.mm(Xtr_submean, Xte_submean.T)
    b = torch.sqrt(torch.sum(Xtr_submean ** 2, axis=1, keepdim=True))
    c = torch.sqrt(torch.sum(Xte_submean ** 2, axis=1, keepdim=True))
    cor = a / (b @ c.T); cor = cor.T; cor = cor.numpy()
    dat_cor = pd.DataFrame(cor, index=Xte_index, columns=Xtr_index)
    return dat_cor

### calculation the precision score
def precision(GSE, cell_line, trTime, method):
    doMultiProcess = RunMultiProcess()
    MOA = doMultiProcess.MOA
    landmarker = doMultiProcess.landmarker
    Nums_right = 0; Nums_ypd = 1   ## Nums_ypd  取多少个结果
    sig_id2pert_iname = sigid2iname()
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    filein  = 'ZScore/{}/{}_{}/{}{}{}.tsv'.format(cell_line, GSE, trTime, method, MOA, landmarker)
    if not os.path.isfile(filein) or os.path.getsize(filein) ==0:
        return '', ''
    dat = pd.read_csv(filein, sep='\t', header=None)
    Nums_ypds = int((dat.shape[1] - 1) / 2)  ## 结果的个数
    if Nums_ypds < Nums_ypd:  Nums_ypd = Nums_ypds
    dat.columns = ['pert_iname'] + ['pert_iname_' + str(i) for i in range(Nums_ypds)] + ['cor_' + str(i) for i in range(Nums_ypds)]
    for i in dat.columns[:Nums_ypds+1]:
        dat[i] = dat[i].apply(lambda x: sig_id2pert_iname[x].split('_')[0])
    for i in range(dat.shape[0]):
        yte = dat.loc[i, 'pert_iname']
        ypd_list = dat.loc[i, ['pert_iname_' + str(i) for i in range(Nums_ypd)]].tolist()
        if yte in ypd_list:
            Nums_right += 1
    result = str(round(Nums_right / dat.shape[0], 3))
    return result, str(len(np.unique(dat['pert_iname'])))

class RunMultiProcess(object):
    def __init__(self, methods = ''):
        self.KNN_size = 3; self.singleLabel = True; self.MOA = ''  ###  MOA, ATC
        self.landmarker = ''  ### lm
        self.methods = methods; self.level = 'L4'; self.query_set = 'CCLE'
        self.ref_set = 'GDSC_ChEMBL_CTRP'     ####  GDSC  ChEMBL  CTRP
        self.GSEs = ['GSE92742']
        self.cell_lines = ['MCF7', 'A375', 'PC3',  'HT29', 'A549', 'BT20', 'VCAP', 'HCC515', 'HEPG2', 'HA1E', 'NPC']
        self.trTimes = ['24H', '6H']
        if self.methods:
            self.mylist = list(product(self.GSEs, self.cell_lines, self.trTimes, self.methods))
        else:
            self.mylist = list(product(self.GSEs, self.cell_lines, self.trTimes))
    
    def myPool(self, func, mylist, processes):
        with Pool(processes) as pool:
            results = list(tqdm(pool.imap(func, mylist), total=len(mylist)))
        return results

def getLmGenes():
    filein = '/home//database/CMap/gene_info.txt'
    lm = pd.read_csv(filein, sep='\t', header=0)
    lmgenes = lm[lm['pr_is_lm'] == 1]['pr_gene_id'].tolist()
    lmgenes = ['Entrez_' + str(i) for i in lmgenes]
    return lmgenes
