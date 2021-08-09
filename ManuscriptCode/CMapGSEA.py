import numpy as np,pandas as pd
import sys,re
from multiprocessing import Pool
from scipy import stats
from util import RunMultiProcess
import warnings
warnings.filterwarnings('ignore')

def isin2D(full_array, sub_arrays):
    out = np.zeros((sub_arrays.shape[0],len(full_array)),dtype=bool)
    sidx = full_array.argsort()
    idx = np.searchsorted(full_array, sub_arrays, sorter=sidx)
    idx[idx==len(full_array)] = 0
    idx0 = sidx[idx]
    np.put_along_axis(out, idx0, full_array[idx0] == sub_arrays, axis=1)
    out = out.astype(int)
    return out.T


def calculateScore(all_Genelist,corMat_subset):
    axis=0
    corMat_subset = corMat_subset.abs()
    refGene = np.array(corMat_subset.index)
    perm_tag_tensor = isin2D(refGene, all_Genelist)
    no_tag_tensor = 1 - perm_tag_tensor
    rank_alpha = perm_tag_tensor * corMat_subset[:, np.newaxis]
    P_GW_denominator = np.sum(rank_alpha,axis=axis,keepdims=True)
    P_NG_denominator = np.sum(no_tag_tensor,axis=axis,keepdims=True)
    REStensor = np.cumsum(rank_alpha / P_GW_denominator - no_tag_tensor / P_NG_denominator,axis=axis)
    esmax, esmin = REStensor.max(axis=axis), REStensor.min(axis=axis)
    esmatrix = np.where(np.abs(esmax) > np.abs(esmin), esmax, esmin)
    return esmatrix

def GSEA(corMat_subset):
    up_es = calculateScore(all_upGenelist, corMat_subset)
    dn_es = calculateScore(all_dnGenelist, corMat_subset)
    es = np.where(np.sign(up_es) == np.sign(dn_es), 0, up_es - dn_es)
    es = pd.DataFrame(es.reshape(-1, 1), index = Xte_index, columns=[corMat_subset.name])
    return es


def runGSEA(Xtr, Xte, processes = 32, num_genes=100):
    global all_upGenelist; global all_dnGenelist; global Xte_index
    all_upGenelist = []; all_dnGenelist = []; Xte_index = Xte.index
    for i in range(Xte.shape[0]):
        tmp = Xte.iloc[i, :]
        tmp.sort_values(ascending=True, inplace=True)
        upGenelist = tmp.index[-num_genes:]; dnGenelist = tmp.index[:num_genes]
        all_upGenelist.append(upGenelist)
        all_dnGenelist.append(dnGenelist)
    all_upGenelist = np.array(all_upGenelist); all_dnGenelist = np.array(all_dnGenelist)
    mylist = [Xtr.iloc[i,:].sort_values(ascending=False) for i in range(Xtr.shape[0])]
    doMultiProcess = RunMultiProcess()
    result = doMultiProcess.myPool(GSEA, mylist, processes)
    result = pd.concat(result, axis = 1)
    return result