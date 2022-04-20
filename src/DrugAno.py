import argparse
parser = argparse.ArgumentParser(description='It is used for drug annotation',formatter_class=argparse.RawDescriptionHelpFormatter,add_help=True)

Req = parser.add_argument_group('Required')
Req.add_argument('-ref', metavar='\b', action='store', help='reference used for training', required=True, )
Req.add_argument('-query', metavar='\b', action='store', help='query used for assignment', required=True, )

Opt = parser.add_argument_group('Optional')
Opt.add_argument('-pvalue', metavar='\b', action='store', default=0.01, type = float, help='pvalue, default: 0.01')
Opt.add_argument('-variance', metavar='\b', action='store', type = float, default=0.98, help='variance to keep, default: 0.98')
Opt.add_argument('-dimension', metavar='\b', action='store', type = int, default=50, help='dimension of LDA, default: 50')
Opt.add_argument('-output', metavar='\b', action='store', type = str, default='DrSim.tsv', help='outfile prefix, default: DrSim.tsv')
Opt.add_argument('-num', metavar='\b', action='store', type = int, default=10, help='number of results to keep, default: 10')


args = parser.parse_args()
import os, subprocess
import pandas as pd, numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from util import calCosine, drugTOMOA, sigidTo

Datapath = os.path.dirname(os.path.abspath(__file__))

### output ranked drug annotation result
def writeResult(dat_cor, output_file):
    drug2MOA = drugTOMOA()
    with open(output_file, 'w') as fout:
        a = ['Query'] + ['MOA_' + str(i) for i in range(1, args.num +1 )] + ['MOAScore_' + str(i) for i in range(1, args.num+1)]
        fout.write('{}\n'.format('\t'.join(a)))
        for i in dat_cor.index:
            tmp = dat_cor.loc[i, :]
            positive = tmp.sort_values(ascending=False)[:args.num].index.tolist()
            positive =  [drug2MOA.get(i, 'unKnown') for i in positive]
            values = tmp.sort_values(ascending=False)[:args.num].values.tolist()
            values = [str(round(i,4)) for i in values]
            fout.write('{}\t{}\t{}\n'.format(i, '\t'.join(positive), '\t'.join(values)))

### LDA main function
def runLDA():
    Xtr = pd.read_hdf(args.ref)
    if os.path.isfile(args.query):
        Xte = pd.read_csv(args.query, sep='\t', header=0, index_col=0)
    else:
        print ('query file not exist!')
    tmp = [i for i in Xtr.columns if i in Xte.columns]
    Xtr = Xtr.loc[:, tmp]; Xte = Xte.loc[:, tmp]
    sigid2MOA, sigid2drug = sigidTo('')
    pert_iname = [sigid2drug[i] for i in Xtr.index]  ### use drug name as the training label
    pca = PCA(random_state=2020, n_components=args.variance) ### dimension reduction with PCA
    Xtr_pca = pca.fit_transform(Xtr)
    Xte_pca = pca.transform(Xte)
    labelencoder = LabelEncoder()
    ytr = labelencoder.fit_transform(pert_iname)
    ml = LinearDiscriminantAnalysis(solver='svd', n_components=args.dimension)
    Xtr_pca_lda = ml.fit_transform(Xtr_pca, ytr)
    Xte_pca_lda = ml.transform(Xte_pca)
    Xtr_pca_lda = Xtr_pca_lda[:, ~np.isnan(Xtr_pca_lda)[0]] ## filter NA column
    Xte_pca_lda = Xte_pca_lda[:, ~np.isnan(Xte_pca_lda)[0]] ## filter NA column
    a = pd.DataFrame(Xtr_pca_lda, index = pert_iname)
    ref = a.groupby(pert_iname).median()
    query = pd.DataFrame(data = Xte_pca_lda, index = Xte.index)
    dat_cor = calCosine(Xtr = ref, Xte = query) ### calculation of similarity between query and reference signature
    writeResult(dat_cor, args.output)


if __name__ == '__main__':
    runLDA()
