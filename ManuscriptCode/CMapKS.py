import numpy as np, pandas as pd
import sys, re
from multiprocessing import Pool
from scipy import stats
from util import RunMultiProcess

def get_ScoreA(ranka,rankb,Genelist,refGene):
    score = ranka / len(Genelist) - rankb / len(refGene)
    return score.round(4)

def get_ScoreB(ranka,rankb,Genelist,refGene):
    score = rankb / len(refGene) - (ranka - 1) / len(Genelist)
    return score.round(4)


def calculateScore(Genelist):
    ranka = np.arange(len(Genelist)).reshape((len(Genelist), 1)) + 1
    rankb = Xtr_rank[Genelist].T.values
    score_a = get_ScoreA(ranka=ranka, rankb=rankb, Genelist=Genelist, refGene=Xtr_rank.columns)
    score_b = get_ScoreB(ranka=ranka, rankb=rankb, Genelist=Genelist, refGene=Xtr_rank.columns)
    score_a = score_a.max(axis=0)
    score_b = -score_b.min(axis=0) + 1 / len(Genelist)
    score = np.where(score_a >= score_b, score_a, -score_b)
    return score

def KS(X):
    upGenelist, dnGenelist, index = X
    up_es = calculateScore(upGenelist)
    dn_es = calculateScore(dnGenelist)
    es = np.where(np.sign(up_es) == np.sign(dn_es), 0, up_es - dn_es)
    es = pd.DataFrame(es.reshape(1, -1), index = [index], columns= Xtr_rank.index)
    return es

def runKS(Xtr, Xte, processes=16, num_genes=100):
    global Xtr_rank
    mylist = []; Xtr_rank = Xtr.rank(axis=1, ascending=False, method='first')
    for i in range(Xte.shape[0]):
        tmp = Xte.iloc[i, :]
        tmp.sort_values(ascending=True, inplace=True)
        upGenelist = tmp.index[-num_genes:]
        dnGenelist = tmp.index[:num_genes]
        mylist.append([upGenelist, dnGenelist, tmp.name])
    doMultiProcess = RunMultiProcess()
    result = doMultiProcess.myPool(KS, mylist, processes)
    result = pd.concat(result, axis = 0, sort=True)
    return result

