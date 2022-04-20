#coding=utf-8
import os, sys, re, glob, subprocess, glob, string
from collections import defaultdict
import requests, json, time
import numpy as np,pandas as pd
from multiprocessing import Pool, Process
try:
    from cmapPy.pandasGEXpress.parse import parse
    import cmapPy.pandasGEXpress.subset_gctoo as sg
except:
    pass
from itertools import product
try:
    from util import convertDrugName, getDrugiDose, RunMultiProcess
except:
    pass

### generate LINCS reference signature from downloaded GCTX file and metadata

### only retain cell lines with enough signatures
cell_lines = ['MCF7', 'A375', 'PC3', 'HT29', 'YAPC', 'HELA', 'A549', 'BT20','VCAP', 'HCC515', 'HEPG2']
cell_lines = ['HA1E', 'NPC']

### preprocess metadata file
def processGSE92742():
    keep_idose = [1, 10, 100, 500, 1000, 3000, 5000, 10000]
    os.chdir('/home//project/Metric_learning')
    sig_info = '/home//database/CMap/GSE92742/GSE92742_Broad_LINCS_sig_info.txt'
    #with open(sig_info, 'r') as fin, open('GSE92742SigInfo.tsv', 'w') as fout:
    with open(sig_info, 'r') as fin, open('GSE92742OtherCelllineSigInfo.tsv', 'w') as fout:
        fout.write('sig_id\tpert_id\tpert_iname\tpert_dose\tpert_idose\tdistil_id\n')
        fin.readline()
        for line in fin:
            try:
                lines = line.strip().split('\t')
                cell_line = lines[4]; pert_type = lines[3]
                treat_time = lines[0].strip().split(':')[0].split('_')[-1]
                sig_id = lines[0]; pert_id = lines[1]; distil_id = lines[-1]
                if cell_line not in cell_lines or pert_type != 'trt_cp' or treat_time not in ['24H', '6H']:
                    continue
                pert_idose = getDrugiDose(lines[7])
                pert_iname = convertDrugName(lines[2])
                pert_dose = '{:.0f}'.format(float(lines[5].split()[0])*1000)
                if pert_idose not in keep_idose:
                    continue
                fout.write('{}\t{}\t{}\t{}nM\t{}nM\t{}\n'.format(sig_id, pert_id, pert_iname, pert_dose, pert_idose, distil_id))
            except:
                print (line)
                break

def processGSE92742_sh():
    os.chdir('/home//project/Metric_learning')
    sig_info = '/home//database/CMap/GSE92742/GSE92742_Broad_LINCS_sig_info.txt'
    with open(sig_info, 'r') as fin, open('SigInfo_sh.tsv', 'w') as fout:
        fin.readline()
        for line in fin:
            try:
                lines = line.strip().split('\t')
                cell_line = lines[4]; pert_type = lines[3]
                treat_time = lines[0].strip().split(':')[0].split('_')[-1]
                sig_id = lines[0]; pert_id = lines[1]; distil_id = lines[-1]
                if cell_line not in cell_lines or pert_type != 'trt_sh' or lines[7] =='-666' or treat_time not in ['96H']:
                    continue
                pert_idose = getDrugiDose(lines[7])
                pert_iname = lines[2]
                pert_dose = '{:.0f}'.format(float(lines[5].split()[0])*1000)
                fout.write('{}\t{}\t{}\t{}nM\t{}nM\t{}\n'.format(sig_id, pert_id, pert_iname, pert_dose, pert_idose, distil_id))
            except Exception as e:
                print (e)
                break

## subset signature using cell line and trTime factor
def exPress(X):
    GSE, cell_line, time = X
    #fileout = 'ZScore/{}/{}_{}/zscore.tsv'.format(cell_line, GSE, time)
    #fileout_h5 = 'ZScore/{}/{}_{}/zscore.h5'.format(cell_line, GSE, time)
    mydir = 'ZScore/{}/{}_{}'.format(cell_line, GSE, time)
    if not os.path.isdir(mydir): os.makedirs(mydir)
    fileout = 'ZScore/{}/{}_{}/zscoreL4.tsv'.format(cell_line, GSE, time)
    fileout_h5 = 'ZScore/{}/{}_{}/zscoreL4.h5'.format(cell_line, GSE, time)
    cid = []
    #filein = 'SigInfo.tsv'
    filein = 'GSE92742OtherCelllineSigInfo.tsv'
    with open(filein, 'r') as fin:
        fin.readline()
        for line in fin:
            lines = line.strip().split('\t')
            information = lines[0].split(':')[0].split('_')
            distil_ids = lines[5].split('|')
            if information[1] == cell_line and information[2] == time:
                cid.extend(distil_ids)
    if len(cid) == 0: return
    dat = sg.subset_gctoo(level5, cid = cid)
    dat = dat.data_df.T; dat.sort_index(axis=0, inplace= True)
    dat.columns = 'Entrez_' + dat.columns
    dat.columns.name = ''; dat = dat.round(4)
    dat.to_csv(fileout, sep = '\t', header=True, index=True, float_format='%.4f')
    dat.to_hdf(fileout_h5, key='dat')

def f_exPress():
    os.chdir('/home//project/Metric_learning')
    for GSE in ['GSE92742']:
        global level5
        if GSE == 'GSE70138':
            #gctx = '/home//database/CMap/GSE70138/GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx'
            gctx = '/home//database/CMap/GSE70138/GSE70138_Broad_LINCS_Level4_ZSPCINF_mlr12k_n345976x12328_2017-03-06.gctx'
        elif GSE == 'GSE92742':
            #gctx = '/home//database/CMap/GSE92742/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx'
            gctx = '/home//database/CMap/GSE92742/GSE92742_Broad_LINCS_Level4_ZSPCINF_mlr12k_n1319138x12328.gctx'
        else:
            gctx = ''
        level5 = parse(gctx)
        mylist =  list(product([GSE], cell_lines, ['6H', '24H']))    
        pool = Pool(processes=2); pool.map(exPress, mylist); pool.close(); pool.join()

# def mergeExpress(cell_line):
#     os.chdir('/home//project/Metric_learning/ZScore/{}'.format(cell_line))
#     for time in ['24H', '6H']:
#         files = glob.glob('*{time}_zscore.tsv'.format(time = time))
#         fileout = 'merge_{time}_zscore.tsv'.format(time = time)
#         if len(files) == 0:
#             continue
#         elif len(files) == 1:
#             cmd = 'cp {} {}'.format(files[0], fileout)
#         elif len(files) == 2:
#             dat1 = pd.read_csv(files[0], sep='\t', index_col=0)
#             dat2 = pd.read_csv(files[1], sep='\t', index_col=0)
#             dat = pd.concat([dat1, dat2], axis=0, sort=False)
#             dat.sort_index(axis=0, inplace= True)
#             dat.to_csv(fileout, sep = '\t', header=True, index=True, float_format='%.4f')

def mergeExpressL4(cell_line, trTime):
    os.chdir('/home//project/Metric_learning/')
    fileout_h5 = 'ZScore/{}/_{}/zscoreL4.h5'.format(cell_line, trTime)
    filein1 = 'ZScore/{}/GSE92742_{}/zscoreL4.h5'.format(cell_line, trTime)
    filein2 = 'ZScore/{}/GSE70138_{}/zscoreL4.h5'.format(cell_line, trTime)
    if os.path.isfile(filein1) and os.path.isfile(filein2):
        dat1 = pd.read_hdf(filein1)
        dat2 = pd.read_hdf(filein2)
        dat  = pd.concat([dat1, dat2], axis=0, sort=True)
        dat.to_hdf(fileout_h5, key='dat')
    elif os.path.isfile(filein1) and not os.path.isfile(filein2):
        cmd = 'cp {} {}'.format(filein1, fileout_h5)
        subprocess.call(cmd, shell=True)
    elif not os.path.isfile(filein1) and os.path.isfile(filein2):
        cmd = 'cp {} {}'.format(filein2, fileout_h5)
        subprocess.call(cmd, shell=True)
    else:
        pass    

def f_mergeExpressL4():
    doMultiProcess = RunMultiProcess()
    for cell_line in doMultiProcess.cell_lines:
        for trTime in doMultiProcess.trTimes:
            mergeExpressL4(cell_line, trTime)

if __name__ == '__main__':
    print ('hello, world')
    #processGSE92742()
    #processGSE70138()
    #processGSE92742_sh()
    f_exPress()
    #f_mergeExpressL4()
    #f_getDrugTarget()
    
