import re, os, glob, subprocess, pickle, time
import pandas as pd, numpy as np
from util import sigid2iname
from sklearn.preprocessing import LabelEncoder
import torch, warnings
from util import calCosine, RunMultiProcess
import metric_learn
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
warnings.filterwarnings('ignore')
doPCA = True; rePCA = True; savePCA = True
singleLabel = RunMultiProcess().singleLabel


### 对药物求质心在求相关性
MOA = ''   ### 训练的时候用的是什么标签

def outer(func):
    def inner(x):
        start = time.time(); func(x)
        spendTime = round((time.time() - start) / 60, 2)
        print ('cost {} min\t'.format(spendTime))
    return inner

@outer
def main(X):
    try:
        GSE, cell_line, trTime = X
        basepath = '/home/wzt/project/Metric_learning'; os.chdir(basepath)
        output_file1  = 'ZScore/{}/{}_{}/LMNN{}.tsv'.format(cell_line, GSE, trTime, MOA)
        output_file2  = 'ZScore/{}/{}_{}/pca{}.tsv'.format(cell_line, GSE, trTime, MOA)
        file_PCA = 'ZScore/{}/{}_{}/PCA{}.pkl'.format(cell_line, GSE, trTime, MOA)
        file_Xtr = 'ZScore/{}/{}_{}/Xtr.h5'.format(cell_line, GSE, trTime) ### reference signature file
        file_Xte = 'ZScore/{}/{}_{}/Xte.h5'.format(cell_line, GSE, trTime) ### query signature file
        if not os.path.isfile(file_Xtr) or not os.path.isfile(file_Xte):
            if os.path.isfile(output_file1): os.remove(output_file1)
            if os.path.isfile(output_file2): os.remove(output_file2)
            print ('{}\t{}\t{}\tfile not exist\t'.format(GSE, cell_line, trTime))
            return ''
        sig_id2pert_iname = sigid2iname(MOA)     ### 单个标签  iname_idose, iname
        Xtr = pd.read_hdf(file_Xtr); Xte = pd.read_hdf(file_Xte)
        tmp = [i for i in Xtr.columns if i in Xte.columns] ### intersection gene name
        Xtr = Xtr.loc[:, tmp]; Xte = Xte.loc[:, tmp]
        if singleLabel:
            pert_iname = [sig_id2pert_iname[i].split('_')[0] for i in Xtr.index]
        else:
            pert_iname = [sig_id2pert_iname[i] for i in Xtr.index]
        if len(np.unique(pert_iname)) == 1: return ''
        if doPCA and rePCA: ### dimension reduction
            n_components = .98
            pca = PCA(random_state=0, n_components = n_components) ## 改变维度
            Xtr_pca = pca.fit_transform(Xtr); Xte_pca = pca.transform(Xte)
            if savePCA:
                with open(file_PCA, 'wb') as fout:
                    pickle.dump(pca, fout)
        elif doPCA and os.path.isfile(file_PCA):
            with open(file_PCA, 'rb') as fin:
                pca = pickle.load(fin)
                Xtr_pca = pca.transform(Xtr); Xte_pca = pca.transform(Xte)
        else:
            Xtr_pca = Xtr.values; Xte_pca = Xte.values
        labelencoder = LabelEncoder()
        ytr = labelencoder.fit_transform(pert_iname) ### use drug name as the training label
        ml = LinearDiscriminantAnalysis(solver='svd', n_components=50, )
        #ml = metric_learn.LFDA(n_components=100, k=5)
        Xtr_pca_lmnn = ml.fit_transform(Xtr_pca, ytr); Xte_pca_lmnn = ml.transform(Xte_pca)
        Xtr_pca_lmnn = Xtr_pca_lmnn[:, ~np.isnan(Xtr_pca_lmnn)[0]]   ### 取出不是Nan的列
        Xte_pca_lmnn = Xte_pca_lmnn[:, ~np.isnan(Xte_pca_lmnn)[0]]   ### 取出不是Nan的列

        a = pd.DataFrame(Xtr_pca_lmnn, index=pert_iname)
        ref = a.groupby(pert_iname).median()
        query = pd.DataFrame(data = Xte_pca_lmnn, index = Xte.index)
        dat_cor = calCosine(Xtr = ref, Xte = query)
        with open(output_file1, 'w') as fout:
            n = min(dat_cor.shape[1], 10)
            for i in dat_cor.index:
                tmp = dat_cor.loc[i,:]
                positive = tmp.sort_values(ascending=False)[:n].index.tolist()
                values = tmp.sort_values(ascending=False)[:n].values.tolist()
                values = [str(round(i,4)) for i in values]
                fout.write('{}\t{}\t{}\n'.format(i, '\t'.join(positive), '\t'.join(values)))
        
        a = pd.DataFrame(data=Xtr_pca, index = pert_iname)
        ref = a.groupby(pert_iname).median()
        query = pd.DataFrame(data = Xte_pca, index = Xte.index)
        dat_cor = calCosine(Xtr = ref, Xte = query)
        with open(output_file2, 'w') as fout:
            for i in dat_cor.index:
                tmp = dat_cor.loc[i,:]
                positive = tmp.sort_values(ascending=False)[:10].index.tolist()
                values = tmp.sort_values(ascending=False)[:10].values.tolist()
                values = [str(round(i,4)) for i in values]
                fout.write('{}\t{}\t{}\n'.format(i, '\t'.join(positive), '\t'.join(values)))
        print ('{}\t{}\t{}\tcompleted\t'.format(GSE, cell_line, trTime))
    except Exception as e:
        print (e)
        print ('{}\t{}\t{}\tfailed\t'.format(GSE, cell_line, trTime))

def runMain():
    doMultiProcess = RunMultiProcess()
    for GSE, cell_line, trTime in doMultiProcess.mylist:
        main((GSE, cell_line, trTime))

if __name__ == '__main__':
    print ('hello, world')
    runMain()
