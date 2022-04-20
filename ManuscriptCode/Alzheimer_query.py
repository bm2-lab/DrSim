import re, os, glob, subprocess, pickle, time
import pandas as pd, numpy as np

### 预处理基因注释信息
### preprocess downloaded metadata
def GSE26972():
    os.chdir('/home//project/Metric_learning/Alzheimer/GSE26972')
    filein1 = '/home//database/CMap/gene_info.txt'
    filein2 = 'annoRaw.tsv'
    dat = pd.read_csv(filein1, sep='\t')
    with open(filein2, 'r') as fin, open('anno.tsv', 'w') as fout:
        fout.write('ProbeID\tSymbol\tEntrez\n')
        fin.readline()
        for line in fin:
            lines = line.strip().split('\t', 1)
            if len(lines) !=1 and lines[1] != '---':
                temp  = lines[1].split('/')
                temp = [i.strip() for i in temp if i]
                for i in temp:
                    if i in dat.pr_gene_symbol.values:
                        fout.write('{}\t{}\t{}\n'.format(lines[0], i, dat.loc[dat['pr_gene_symbol'] == i, 'pr_gene_id'].values[0]))
                        break

#### using Z-score to generate AD disease signature
def ZScorequery(GSE):
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    filein = 'Alzheimer/{}/exp.tsv'.format(GSE)
    dat = pd.read_csv(filein, sep='\t', index_col=0)
    if GSE == 'GSE26972':
        control_ = dat.iloc[:, :3]; treat = dat.iloc[:, 3:]
    else:
        pass
    treat_ = np.median(treat, axis=1)
    treat_ = pd.DataFrame(treat_, index = treat.index, columns=['treat'])
    a = np.median(control_, axis=1, keepdims=True)
    b = stats.median_absolute_deviation(control_, axis=1).reshape(-1, 1)
    if b.shape[1] == 1:
        b[b==0] = 1   ##防止为0
    else:
        b[b==0] = np.median(b)   ##防止为0
    result = (treat_ - a) / 1.4826 / b; result.index.name = ''; result = result.T
    result.columns = ['Entrez_' + str(i) for  i in result.columns]
    file_Xte = 'Alzheimer/{}/Xte_ZScore.tsv'.format(GSE)
    result.to_csv(file_Xte, sep='\t', header=True, index = True)


if __name__ == '__main__':
    #GSE26972()
    ZScorequery('GSE26972')
