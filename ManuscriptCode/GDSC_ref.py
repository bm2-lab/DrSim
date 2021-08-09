import os,sys,re,glob,subprocess
from collections import defaultdict
import numpy as np,pandas as pd
from multiprocessing import Pool
import string
from shutil import copyfile
from util import convertDrugName, sigid2iname
from util import calCosine, calPearson, calSpearman, RunMultiProcess
from scipy.stats import stats

## depmap 找相关的细胞系信息
### HCC515和HEPG2在GDSC中没有,找最相似的代替
#              乳腺癌   黑色素瘤  前列腺癌 结肠癌  胰腺癌   宫颈癌   肺腺癌   乳腺癌  前列腺癌  肺腺癌    肝癌
#cell_lines = ['MCF7', 'A375', 'PC3', 'HT29', 'YAPC', 'HELA', 'A549', 'BT20','VCAP', 'HCC515', 'HEPG2']
cell_lines =  ['MCF7', 'A375', 'PC-3', 'HT-29', 'A549', 'BT-20', 'VCaP']
sig_id2pert_iname = sigid2iname('')
singleLabel = RunMultiProcess().singleLabel
level = RunMultiProcess().level
ref_set = RunMultiProcess().ref_set

def fun1():
    os.chdir('/home//database/GDSC')
    dat = pd.read_excel('GDSC1_fitted_dose_response_15Oct19.xlsx')
    dat = dat[dat.CELL_LINE_NAME.isin(cell_lines)]
    dat.to_csv('GDSC1_cell_lines_IC50.tsv', sep='\t', header=True, index=False)
    dat = pd.read_excel('GDSC2_fitted_dose_response_15Oct19.xlsx')
    dat = dat[dat.CELL_LINE_NAME.isin(cell_lines)]
    dat.to_csv('GDSC2_cell_lines_IC50.tsv', sep='\t', header=True, index=False)

def subset():
    gdsc1 = pd.read_csv('/home//database/GDSC/GDSC1_cell_lines_IC50.tsv', sep='\t')
    gdsc2 = pd.read_csv('/home//database/GDSC/GDSC2_cell_lines_IC50.tsv', sep='\t')
    dat = pd.concat([gdsc2, gdsc1], axis=0)
    dat['DRUG_ID'] = dat['DRUG_ID'].astype(np.str)
    dat['CELL_LINE_NAME_AND_DRUG_ID'] = dat['CELL_LINE_NAME'] + '_' + dat['DRUG_ID']
    dat.drop_duplicates(subset='CELL_LINE_NAME_AND_DRUG_ID', keep='first', inplace=True)
    dat.drop(columns=['CELL_LINE_NAME_AND_DRUG_ID'], inplace=True)
    dat.to_csv('/home//database/GDSC/GDSC_cell_lines_IC50.tsv',sep='\t',header=True,index=False)

##############################################################
def getIC50(cell_line = 'MCF7',metric='IC50'):
    os.chdir('/home//database/GDSC/{}'.format(metric))
    drugIC50 = {}; pubchemIC50 = {}; perttypeIC50 = defaultdict(list)
    with open('../GDSC_cell_lines_IC50.tsv','r') as fin:
        fin.readline()
        for line in fin:
            lines = line.strip().split('\t')
            if lines[4] == cell_line:
                if metric == 'IC50':
                    drugIC50[lines[7]] = np.e**(float(lines[15])) * 1000     ## 一一对应,转换成nM
                else:
                    drugIC50[lines[7]] = float(lines[16])
    with open('../GDSC_drug_info.txt','r') as fin:
        fin.readline()
        for line in fin:
            lines = line.strip().split('\t')
            if lines[5] != '-' and lines[0] in drugIC50:
                pubchems = lines[5].strip().split(',')
                for i in pubchems:
                    i = i.strip()
                    pubchemIC50[i] = drugIC50[lines[0]]
    with open('/home//database/CMap/CMap_drug_info.txt','r') as fin:
        fin.readline()
        for line in fin:
            lines = line.strip().split('\t')
            lines[1] = convertDrugName(lines[1])
            if lines[7] in pubchemIC50:
                perttypeIC50[lines[1]].append(pubchemIC50[lines[7]]) ## 一个药物有几个pubchem ID
    mydict = {}
    for i in perttypeIC50:
        mydict[i] = np.median(perttypeIC50[i])     ## median or mean

    with open('{}.txt'.format(cell_line), 'w') as fout:
        for i in mydict:
            fout.write('{}\t{}\n'.format(i,mydict[i]))

def f_getIC50(metric):
    for cell_line in cell_lines:
        getIC50(cell_line = cell_line, metric= metric)
    copyfile('PC-3.txt', 'PC3.txt')
    copyfile('HT-29.txt', 'HT29.txt')
    copyfile('BT-20.txt', 'BT20.txt')
    copyfile('VCaP.txt', 'VCAP.txt')


#################################################################
### 用原始的zscore signature
def CMapSignature(X):
    allSize = 3
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    GSE, cell_line, trTime = X
    if trTime == '6H': return
    input_file = 'ZScore/{}/{}_{}/zscore{}.h5'.format(cell_line, GSE, trTime, level)
    if not os.path.isfile(input_file): return ''
    dat = pd.read_hdf(input_file, key='dat')
    sig_id2pert_iname = sigid2iname('')
    a = [i for i in dat.index if i in sig_id2pert_iname]
    b = list(map(lambda x: sig_id2pert_iname[x].split('_')[0], a))
    filein = '/home//database/{}/IC50/{}.txt'.format(ref_set, cell_line)
    if not os.path.isfile(filein): return ''
    druglist = []
    with open(filein, 'r') as fin:
        for line in fin:
            lines = line.strip().split('\t')
            druglist.append(convertDrugName(lines[0]))
    selected = [True if i in druglist and b.count(i) >= allSize else False for i in b]
    #selected = [True if b.count(i) >= 3 else False for i in b]
    Xtr = dat.iloc[selected, :]
    if Xtr.shape[0] <= 10: return
    mydir = 'msViper/{}/{}_{}/{}'.format(cell_line, GSE, trTime, level)
    if not os.path.isdir(mydir): os.makedirs(mydir)
    file_Xtr = 'msViper/{}/{}_{}/{}/{}_Xtr_ZScore.h5'.format(cell_line, GSE, trTime, level, ref_set)
    if os.path.isfile(file_Xtr): os.remove(file_Xtr)
    Xtr.to_hdf(file_Xtr, key = 'dat')


def f_CMapSignature():
    doMultiProcess = RunMultiProcess()
    doMultiProcess.myPool(CMapSignature, doMultiProcess.mylist, 6)

def mergeSignature(GSE, cell_line, trTime, metric):
    basepath = '/home//project/Metric_learning'; os.chdir(basepath)
    path = 'msViper/{}/{}_{}/{}'.format(cell_line, GSE, trTime, level)
    all_dat = []; druglist = [];  signatures = []
    filein = '/home//database/{}/IC50/{}.txt'.format(ref_set, cell_line)
    with open(filein, 'r') as fin:
        for line in fin:
            lines = line.strip().split('\t')
            druglist.append(convertDrugName(lines[0]))
    signatures = glob.glob('{path}/{metric}/*_{cell_line}_{trTime}*.signature'
        .format(path = path, cell_line = cell_line, trTime = trTime, metric = metric))  ##
    signatures = [i for i in signatures if sig_id2pert_iname[os.path.basename(i)[:-10]].split('_')[0] in druglist]
    if len(signatures) == 0:
        return ''
    for signature in signatures:
        name = os.path.basename(signature)[:-10]
        dat = pd.read_csv(signature, index_col=0, sep= '\t')
        dat.columns = [name]
        all_dat.append(dat)
    dat = pd.concat(all_dat,axis=1)
    file_Xtr = '{}/Xtr_{}.h5'.format(path, metric)
    dat = dat.T; dat.columns = ['Entrez_' + str(i) for i in dat.columns]
    if os.path.isfile(file_Xtr): os.remove(file_Xtr)
    #dat.to_csv(file_Xtr, sep= '\t', header=True, index=True)
    dat.to_hdf(file_Xtr, key = 'dat')

def f_mergeSignature(metric):
    doMultiProcess = RunMultiProcess(methods=[metric])
    for GSE, cell_line, trTime, metric in doMultiProcess.mylist:
        mergeSignature(GSE, cell_line, trTime, metric)
        print (GSE, cell_line, trTime)

def mergeIC50():
    cell_lines =  ['MCF7', 'A375', 'PC3',  'HT29', 'A549', 'BT20','VCAP', 'HCC515', 'HEPG2']
    for cell_line in cell_lines:
        file1 = '/home//database/GDSC/IC50/{}.txt'.format(cell_line)
        file2 = '/home//database/ChEMBL/IC50/{}.txt'.format(cell_line)
        file3 = '/home//database/CTRP/IC50/{}.txt'.format(cell_line)
        fileout = '/home//database/GDSC_ChEMBL_CTRP/IC50/{}.txt'.format(cell_line)
        dat_list = []
        for i in ['GDSC', 'ChEMBL', 'CTRP']:
            filein = '/home//database/{}/IC50/{}.txt'.format(i, cell_line)
            if os.path.isfile(filein):
                dat = pd.read_csv(filein, sep='\t', header=None)
                dat.columns = ['drug', 'IC50']
                if i == 'CTRP':  cutoff = np.median(dat['IC50'])
                else: cutoff = 10000
                dat['label'] = np.where(dat.IC50 >= cutoff, 'ineffective', 'effective')
                dat_list.append(dat)
        if dat_list:
            dat = pd.concat(dat_list, axis=0)
            dat.to_csv(fileout, sep='\t', index=False, header=True)
                
if __name__ == '__main__':
    print ('hello, world')
    #subset(); f_getIC50(metric = 'IC50')
    #mergeIC50()
    f_CMapSignature()    ### 原始ZScore signature
