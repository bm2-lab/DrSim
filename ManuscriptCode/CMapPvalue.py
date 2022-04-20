import numpy as np,pandas as pd
import sys,re
from multiprocessing import Pool
from scipy import stats
from statsmodels.stats.multitest import multipletests
from util import calCosine, RunMultiProcess

### calculation the p-value of a compound using random generated query signatures
pValue = True
def calPvalue(ref, query, experiment, fun):
    nperm = 1000 if pValue else 1
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

def calPvalue1(ref, query, experiment, fun):
    nperm = 1000 if pValue else 1
    rs = np.random.RandomState(seed=2020)
    perm = np.repeat(query.values, nperm + 1, axis=0).reshape(nperm+1, -1)
    np.apply_along_axis(rs.shuffle, 1, perm[1:, ])
    query_perm_index = query.index.tolist() + [query.index[0] + '_' + str(i) for i in range(nperm)]
    query_perm = pd.DataFrame(perm, index= query_perm_index, columns=query.columns)
    query_list = [query_perm.iloc[i:i+1, :] for i in range(query_perm.shape[0])]
    dat_cor = [fun(ref, i) for i in query_list]
    dat_cor = pd.concat(dat_cor, axis=0)
    if experiment == 'positive':
        result = np.sum(dat_cor.iloc[0, :].values <= dat_cor.iloc[1:,:].values, axis=0) / nperm
    else:
        result = np.sum(dat_cor.iloc[0, :].values >= dat_cor.iloc[1:,:].values, axis=0) / nperm
    result = pd.DataFrame(result.reshape(1, -1), index = query.index, columns=ref.index)
    return result
