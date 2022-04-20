import os, pickle, time
import pandas as pd, numpy as np
from util import sigid2iname
from sklearn.preprocessing import LabelEncoder
import warnings
from util import calCosine, RunMultiProcess, drug2MOA
import metric_learn
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
warnings.filterwarnings('ignore')
doPCA = True; rePCA = False; savePCA = False
singleLabel = RunMultiProcess().singleLabel
sig_id2pert_iname = sigid2iname('')
sig_id2MOA = sigid2iname('MOA')



### 和其它的算法LFDA, LMNN, NCA, MLKR进行比较
### 对药物求质心在求相关性
MOA = ''   ### 训练的时候用的是什么标签, using drug name as the training label

def outer(func):
    def inner(x):
        start = time.time(); func(x)
        spendTime = round((time.time() - start) / 60, 2)
        print ('cost {} min\t'.format(spendTime))
    return inner

#@outer
def main(X):
    with open('/home//project/Metric_learning/ZScore1/spendTime.tsv', 'a') as spTime:
        GSE, cell_line, trTime = X
        basepath = '/home//project/Metric_learning'; os.chdir(basepath)
        file_PCA = 'ZScore/{}/{}_{}/PCA{}.pkl'.format(cell_line, GSE, trTime, MOA)
        file_Xtr = 'ZScore/{}/{}_{}/Xtr.h5'.format(cell_line, GSE, trTime)
        file_Xte = 'ZScore/{}/{}_{}/Xte.h5'.format(cell_line, GSE, trTime)
        if not os.path.isfile(file_Xtr) or not os.path.isfile(file_Xte):
            print ('{}\t{}\t{}\tfile not exist\t'.format(GSE, cell_line, trTime))
            return ''
        sig_id2pert_iname = sigid2iname(MOA)     ### 单个标签  iname_idose, iname
        Xtr = pd.read_hdf(file_Xtr); Xte = pd.read_hdf(file_Xte)
        tmp = [i for i in Xtr.columns if i in Xte.columns]
        Xtr = Xtr.loc[:, tmp]; Xte = Xte.loc[:, tmp]
        if singleLabel:
            pert_iname = [sig_id2pert_iname[i].split('_')[0] for i in Xtr.index]
        else:
            pert_iname = [sig_id2pert_iname[i] for i in Xtr.index]
        if len(np.unique(pert_iname)) == 1: return ''
        if doPCA and rePCA:
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
        ytr = labelencoder.fit_transform(pert_iname)
        methods = [
        metric_learn.LFDA(n_components=50), 
        metric_learn.NCA(n_components=50), 
        metric_learn.MLKR(n_components=50), 
        LinearDiscriminantAnalysis(solver='svd', n_components=50),
        ]
        for ml in methods:
            start = time.time()
            methodName = ml.__class__.__name__
            if methodName == 'LinearDiscriminantAnalysis':  methodName = 'LDA'
            Xtr_pca_lmnn = ml.fit_transform(Xtr_pca, ytr); Xte_pca_lmnn = ml.transform(Xte_pca)
            Xtr_pca_lmnn = Xtr_pca_lmnn[:, ~np.isnan(Xtr_pca_lmnn)[0]]   ### 取出不是Nan的列
            Xte_pca_lmnn = Xte_pca_lmnn[:, ~np.isnan(Xte_pca_lmnn)[0]]   ### 取出不是Nan的列

            a = pd.DataFrame(Xtr_pca_lmnn, index=pert_iname)
            ref = a.groupby(pert_iname).median()
            query = pd.DataFrame(data = Xte_pca_lmnn, index = Xte.index)
            dat_cor = calCosine(Xtr = ref, Xte = query)
            output_file1  = 'ZScore1/{}/{}_{}/{}{}.tsv'.format(cell_line, GSE, trTime, methodName, MOA)
            with open(output_file1, 'w') as fout:
                n = min(dat_cor.shape[1], 10)
                for i in dat_cor.index:
                    tmp = dat_cor.loc[i,:]
                    positive = tmp.sort_values(ascending=False)[:n].index.tolist()
                    values = tmp.sort_values(ascending=False)[:n].values.tolist()
                    values = [str(round(i,4)) for i in values]
                    fout.write('{}\t{}\t{}\n'.format(i, '\t'.join(positive), '\t'.join(values)))
            spendTime = round((time.time() - start) / 1, 2)
            spTime.write('{}\t{}\t{}\t{}\n'.format(cell_line, trTime, methodName, spendTime))
            print ('{}\t{}\t{}\t{}\n'.format(cell_line, trTime, methodName, spendTime))


def runMain():
    filein = '/home//project/Metric_learning/ZScore1/spendTime.tsv'
    with open(filein, 'w') as fout:
        fout.write('cellLine\ttrTime\tmethodName\tspTime\n')
    doMultiProcess = RunMultiProcess()
    for GSE, cell_line, trTime in doMultiProcess.mylist:
        main((GSE, cell_line, trTime))


#### 计算各个算法的准确性
def runPrecision(X):
    GSE, cell_line, trTime, methods = X
    results1 = []; results2 = []
    for method in methods:
        p, n = precision(GSE, cell_line, trTime, method)
        results1.append(p); results2.append(n)
    return [results1, results2, GSE, cell_line, trTime]


def f_runPrecision(threads):
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    methods = ['LFDA', 'NCA', 'MLKR', 'LDA']
    fileout  = 'ZScore1/MOA_benchmarkDrug.tsv'
    with open(fileout, 'w') as fout:
        fout.write('GSE\tcell_line\ttrTime\t{}\tNumsSamples\n'.format('\t'.join(methods)))
        doMultiProcess = RunMultiProcess(methods= [methods])
        results = doMultiProcess.myPool(runPrecision, doMultiProcess.mylist, threads)
        for results1, results2, GSE, cell_line, trTime in results:
            if any(results1):
                fout.write('{}\t{}\t{}\t{}\t{}\n'.format(GSE,cell_line,trTime, '\t'.join(results1), results2[0]))
    tmp = pd.read_csv(fileout, header=0, sep='\t')  ## 处理输出的结果
    tmp.sort_values(by='NumsSamples', ascending=False, inplace=True)
    tmp = tmp[tmp['NumsSamples'] >= 10]
    tmp['rs'] =  round(1 / tmp['NumsSamples'], 6)
    tmp['Rank_LDA']  = list(map(int, tmp[methods].rank(axis=1, ascending=False)['LDA'].values))
    tmp.to_csv(fileout, sep='\t', header=True, index=False, float_format='%.3f')



def precision_2(GSE, cell_line, trTime, method):
    Nums_right = 0; Nums_sample = 0
    drugtoMOA = drug2MOA('MOA')
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    filein  = 'ZScore1/{}/{}_{}/{}.tsv'.format(cell_line, GSE, trTime, method)
    if not os.path.isfile(filein): return '', ''
    dat = pd.read_csv(filein, sep='\t', header=None)
    Nums_ypds = int((dat.shape[1] - 1) / 2)  ## 结果的个数
    dat.columns = ['pert_iname'] + ['pert_iname_' + str(i) for i in range(Nums_ypds)] + ['cor_' + str(i) for i in range(Nums_ypds)]
    for i in dat.columns[:1]:
        dat[i] = dat[i].apply(lambda x: sig_id2MOA.get(x, 'None_None').split('_')[0])
    for i in dat.columns[1:Nums_ypds+1]:
        dat[i] = dat[i].apply(lambda x: drugtoMOA.get(x, 'None_None').split('_')[0])
    for i in dat.index:
        yte = dat.loc[i, 'pert_iname']
        ypd_list = dat.loc[i, ['pert_iname_' + str(k) for k in range(Nums_ypds)]].tolist()
        ypd_list = [k for k in ypd_list if k != 'None'][:2]
        if yte != 'None':
            Nums_sample += 1
            if yte in ypd_list:  Nums_right += 1
    result = round(Nums_right / Nums_sample, 3)
    return str(result), str(len(np.unique(dat['pert_iname'])))

precision = precision_2



if __name__ == '__main__':
    print ('hello, world')
    #runMain()
    f_runPrecision(60)
