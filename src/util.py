# -*- coding: utf-8 -*-

import string, os, subprocess
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd, numpy as np
Datapath = os.path.dirname(os.path.abspath(__file__))

### only keep ascii_letters and '-' in drug name
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


def myPool(func, mylist, processes):
    with Pool(processes) as pool:
        results = list(tqdm(pool.imap(func, mylist), total=len(mylist)))
    return results

### convert sigid in LINCS to MOA class
def sigidTo(MOA):
    label_filegz = '{}/../data/{}SigInfo.tsv.gz'.format(Datapath, MOA)
    if os.path.isfile(label_filegz):
        cmd = 'gunzip   {}'.format(label_filegz)
        subprocess.call(cmd, shell=True)
    label_file   = '{}/../data/{}SigInfo.tsv'.format(Datapath, MOA)
    sigid2MOA = {}; sigid2drug = {}
    with open(label_file, 'r') as fin:
        fin.readline()
        for line in fin:
            lines = line.strip().split('\t')
            distil_ids = lines[5].split('|')
            if MOA:
                for i in distil_ids:
                    sigid2MOA[i] = lines[6]
                    sigid2drug[i] = lines[2]              
            else:
                for i in distil_ids:
                    sigid2drug[i] = lines[2]
    return sigid2MOA, sigid2drug


### get drug MOA class
def drugTOMOA():
    label_file = '{}/../data/MOASigInfo.tsv'.format(Datapath)
    drug2MOA = {}
    with open(label_file, 'r') as fin:
        fin.readline()
        for line in fin:
            lines = line.strip().split('\t')
            drug2MOA[lines[2]] = lines[-1]
    return drug2MOA

###  cosine similarity between signatures
def calCosine(Xtr, Xte):
    dat_cor = pd.DataFrame(cosine_similarity(Xte,Xtr))
    dat_cor.columns = Xtr.index
    dat_cor.index = Xte.index
    return dat_cor

#### using randomly generated signatures to calculate the pvalue of a compound,
#### for more details, please refer to our manuscript
def calPvalue(ref, query, experiment, fun):
    nperm = 1000
    rs = np.random.RandomState(seed=2020)
    perm = np.repeat(query.values, nperm + 1, axis=0).reshape(nperm+1, -1)
    np.apply_along_axis(rs.shuffle, 1, perm[1:, ])
    query_perm_index = query.index.tolist() + [query.index[0] + '_' + str(i) for i in range(nperm)]
    query_perm = pd.DataFrame(perm, index= query_perm_index, columns=query.columns)
    dat_cor = fun(ref, query_perm)
    if experiment == 'positive':
        result = np.sum(dat_cor.iloc[0, :].values <= dat_cor.iloc[1:,:].values, axis=0) / nperm
    else:
        result = np.sum(dat_cor.iloc[0, :].values >= dat_cor.iloc[1:,:].values, axis=0) / nperm
    result = pd.DataFrame(result.reshape(1, -1), index = query.index, columns=ref.index)
    return result
