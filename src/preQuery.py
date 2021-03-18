import argparse
parser = argparse.ArgumentParser(description='It is used for drug repositioning',formatter_class=argparse.RawDescriptionHelpFormatter,add_help=True)

Req = parser.add_argument_group('Required')
Req.add_argument('-tumor', action='store', help='tumor expression profile', required=True, metavar='\b')
Req.add_argument('-normal', action='store', help='normal expression profile', required=True, metavar='\b')

Opt = parser.add_argument_group('Optional')
Opt.add_argument('-normalize', action='store_true', help= 'normalize the expression file, default: False', default=False)
Opt.add_argument('-log2', action='store_true', help= 'log2 transform the expression file, default: False', default=False)
Opt.add_argument('-output', metavar='\b', help='outfile prefix, default: Query',action='store', default='Query.tsv')

args = parser.parse_args()
import os,subprocess
import numpy as np,pandas as pd
from util import sigidTo
from scipy.stats import stats
import warnings
warnings.filterwarnings('ignore')
Datapath = os.path.dirname(os.path.abspath(__file__))


##RNA-seq
def Normalized():
    try:
        cmd = 'Rscript  {}/normalize.r  {}  {}'.format(Datapath, args.tumor, args.normal)
        subprocess.call(cmd, shell=True)
        treat = pd.read_csv('tmp_treat.tsv', sep='\t', index_col=0)
        control = pd.read_csv('tmp_control.tsv', sep='\t', index_col=0)
        rs = np.random.RandomState(seed = 2020)
        control = control.applymap(lambda x : rs.random(1)[0] / 10 if x <=0 else x)
        treat = treat.applymap(lambda x : rs.random(1)[0] / 10 if x <=0 else x)
        os.remove('tmp_treat.tsv'); os.remove('tmp_control.tsv')
        return treat, control
    except Exception as e:
        print (e); print ('{} failed!'.format(cmd))


def calZScore():
    treat = pd.read_csv(args.tumor, sep='\t', index_col=0)
    control = pd.read_csv(args.normal, sep='\t', index_col=0)
    if args.normalize:
        treat_, control_ = Normalized()
    else:
        if args.log2:
            treat_ = np.log2(treat + 1); control_ = np.log2(control + 1)
        else:
            treat_ = treat.copy(); control_ = control.copy()
    index = treat_.index
    treat_ = np.median(treat_, axis=1)
    treat_ = pd.DataFrame(treat_, index = index, columns=['treat'])
    a = np.median(control_, axis=1, keepdims=True)
    b = stats.median_absolute_deviation(control_, axis=1).reshape(-1, 1)
    b[b==0] = np.median(b)
    result = (treat_ - a) / 1.4826 / b; result = result.T
    result.to_csv(args.output, sep='\t', header=True, index = True)

if __name__ == '__main__':
    calZScore()


