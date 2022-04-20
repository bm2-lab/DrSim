import os,sys,re,glob,subprocess,glob
import numpy as np,pandas as pd
from util import getDrugiDose, convertDrugName, sigidTo
try:
    from cmapPy.pandasGEXpress.parse import parse
    import cmapPy.pandasGEXpress.subset_gctoo as sg
    from multiprocessing import Pool
except Exception as e:
    print (e)

    
### only cell line with enough signature is kept
cell_lines = ['MCF7', 'A375', 'PC3', 'HT29', 'A549', 'BT20','VCAP', 'HCC515', 'HEPG2']

#### preprocess metadata downloaded from LINCS and only retain data in the nine core cell lines.
def processInfo():
    keep_idose = [1, 10, 100, 500, 1000, 3000, 5000, 10000]
    os.chdir('/home/wzt')
    sig_info = 'data/GSE92742_Broad_LINCS_sig_info.txt'
    with open(sig_info, 'r') as fin, open('data/SigInfo.tsv', 'w') as fout:
        fout.write('sig_id\tpert_id\tpert_iname\tpert_dose\tpert_idose\tdistil_id\n')
        fin.readline()
        for line in fin:
                lines = line.strip().split('\t')
                cell_line = lines[4]; pert_type = lines[3]
                trTime = lines[0].strip().split(':')[0].split('_')[-1]
                sig_id = lines[0]; pert_id = lines[1]; distil_id = lines[-1]
                if cell_line not in cell_lines or pert_type != 'trt_cp' or trTime not in ['24H', '6H']: continue
                pert_idose = getDrugiDose(lines[7])
                pert_iname = convertDrugName(lines[2])
                pert_dose = '{:.0f}'.format(float(lines[5].split()[0])*1000)
                if pert_idose not in keep_idose: continue
                fout.write('{}\t{}\t{}\t{}nM\t{}nM\t{}\n'.format(sig_id, pert_id, pert_iname, pert_dose, pert_idose, distil_id))

    MOA = pd.read_csv("data/CMapMOA.tsv", sep='\t', header=0)
    GSE92742 = pd.read_csv('data/SigInfo.tsv', sep='\t', header=0)
    MOA = MOA.loc[MOA['Primary_MOA'] != 'others', ['pert_id', 'Primary_MOA', 'MOA']]
    dat = pd.merge(left=GSE92742, right=MOA, left_on='pert_id', right_on='pert_id')
    dat.to_csv('data/MOASigInfo.tsv', sep='\t', header=True, index=False)

### subset signature from LINCS raw gctx file using cell line and trTime factor
def exPress(X):
    cell_line, trTime = X
    try:        
        mydir = 'data/{}/{}'.format(cell_line, trTime)
        if not os.path.isdir(mydir): os.makedirs(mydir)
        fileout_h5 = '{}/exp.h5'.format(mydir)
        cid = []
        filein = 'data/SigInfo.tsv'
        with open(filein, 'r') as fin:
            fin.readline()
            for line in fin:
                lines = line.strip().split('\t')
                information = lines[0].split(':')[0].split('_')
                distil_ids = lines[5].split('|')
                if information[1] == cell_line and information[2] == trTime: cid.extend(distil_ids)
        if len(cid) == 0: return
        dat = sg.subset_gctoo(level4, cid = cid)
        dat = dat.data_df.T; dat.sort_index(axis=0, inplace= True)
        dat.columns.name = ''; dat = dat.round(4)
        dat.to_hdf(fileout_h5, key='dat')
    except Exception as e:
        print (e); print (X)


def f_exPress():
    os.chdir('/home/wzt')
    global level4
    gctx = 'data/GSE92742_Broad_LINCS_Level4_ZSPCINF_mlr12k_n1319138x12328.gctx'
    level4 = parse(gctx)
    mylist = [[i, j] for i in cell_lines for j in ['6H', '24H']]
    pool = Pool(4); pool.map(exPress, mylist);  pool.close(); pool.join()
    
#### prepare signature file for drug annotation scenario
def drugAnnotation(cell_line, trTime):
    minSize = 5
    filein = 'data/{}/{}/exp.h5'.format(cell_line, trTime)
    fileout = 'data/{}/{}/DrugAnoRef.h5'.format(cell_line, trTime)
    if not os.path.isfile(filein):
        print ('filein not exist'); return
    dat = pd.read_hdf(filein, key='dat')
    sigid2MOA, sigid2drug = sigidTo('MOA')
    a = [i for i in dat.index if i in sigid2MOA]
    b = [sigid2drug[i] for i in a]
    selected = [True if b.count(i) >= minSize else False for i in b]
    a_ = np.array(a)[selected]
    ref = dat.loc[a_, :]
    if os.path.isfile(fileout): os.remove(fileout)
    ref.to_hdf(fileout, key='dat')

def f_drugAnnotation():
    for cell_line in cell_lines:
        for trTime in ['6H', '24H']:
            drugAnnotation(cell_line, trTime)

def getFDA():
    filein = 'data/CMap_FDADrugs.tsv'
    dat = pd.read_csv(filein, sep='\t', header=0)
    dat = dat[dat['Phase'] == 'Launched']
    return dat['Name'].apply(convertDrugName).values

### prepare signature file for drug repositioning scenario
def drugReposition(cell_line, trTime):
    minSize = 3
    sigid2MOA, sigid2drug = sigidTo('')
    FDA_Approved = getFDA()
    filein = 'data/{}/{}/exp.h5'.format(cell_line, trTime)
    fileout = 'data/{}/{}/DrugRepRef.h5'.format(cell_line, trTime)
    dat = pd.read_hdf(filein, key='dat')
    a = [i for i in dat.index if i in sigid2drug]
    b = [sigid2drug[i] for i in a]
    selected = [True if i in FDA_Approved and b.count(i) >= minSize else False for i in b]
    a_ = np.array(a)[selected]; b_ = np.array(b)[selected]
    if os.path.isfile(fileout): os.remove(fileout)
    ref = dat.loc[a_, :]
    ref.to_hdf(fileout, key = 'dat')


def f_drugReposition():
    for cell_line in cell_lines:
        for trTime in ['6H', '24H']:
            drugReposition(cell_line, trTime)


if __name__ == '__main__':
    #processInfo()
    #f_exPress()
    #f_drugAnnotation()
    f_drugReposition()
