# **Dr.Sim: Similarity learning for transcriptional phenotypic drug discovery**  
## Introduction
Dr.Sim is a general learning-based framework that automatically infers similarity measurement that can be used to characterize the transcriptional profile for drug discovery
with a generalized well performance. Traditionally, such similarity measurement has been defined empirically in an unsupervised way, while due to the high-dimensionality and 
the existence of high noise in these high-throughput data, it lacks robustness with limited performance. We evaluated Dr.Sim on the comprehensively publicly available in-vitro 
and in-vivo datasets in drug annotation and repositioning using the high-throughput transcriptional perturbation data, and indicated that Dr.Sim outperforms the existing methods by two folds, proven to be a conceptual improvement by learning the transcriptional similarity to facilitate the broad utility of the high-throughput transcriptional perturbation data for phenotypic drug discovery.

## Dependencies
#### Required Software:
* [sklearn](https://scikit-learn.org/stable/index.html/)
* [numpy](https://numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [scipy](https://www.scipy.org/)   

## Installation
#### Install via docker, highly recommended
Docker image of Dr.Sim is available at https://hub.docker.com/r/bm2lab/dr.sim/.
if you have docker installed, you call pull the image:  

    docker pull bm2lab/dr.sim

#### Install from github   

    git clone https://github.com/bm2-lab/DrSim.git  
    
## Usage
Dr.Sim can be applied for:  
**Drug annotation:**    

    python  DrugAno.py --help
    python  DrugAno.py  -ref  DrugAnoRef.h5   -query  query.tsv
    
**Drug repositioning**    

    python  DrugRep.py  --help
    python  DrugRep.py   -ref  DrugRepRef.h5  -query  query.tsv
    
## User Manual
For detailed information about usage, data preparation, input and output files and examples, please refer to the [Dr.Sim User Manual](/doc/Dr.Sim_User_Manual.md).
 
## Dr.Sim flowchart
![](workflow.png)<!-- -->
### **Dr.Sim** comprises three steps: data preprocessing, model training and similarity calculation
* **(i)** In the first step, only signatures treated by compounds for 6H or 24H in the nine human cancer cell lines are retained and retained signatures are split into subsets according to cell type and time-point attributes. 
*  **(ii)** In the second step, Dr. Sim automatically infers a similarity measurement used for query assignment based on the training reference signatures. First, PCA was applied to reference signatures to denoise and reduce dimensionality. A transformation matrix P is learned. Second, through applying LDA to the dimensionality reduced signatures, a transformation matrix L is learned based on the signature labels indicating similarities and dissimilarities between signatures. The label of a signature is the compound class that it is induced by. Finally, the transformed references denoted as TR belonging to the identical class are median centered to derive the transformed median centered references (denoted as TMR). The transformed references TR is calculated using Eq. 1. The C denotes the classes of signatures. 
*  **(iii)** In the third step, given a query signature, after transformed by P and L, its similarities to the TMR were calculated by cosine similarity (Eq. 3).
 

## Citation:
Similarity learning for transcriptional phenotypic drug discovery, submitted, 2021.

## Contact
Zhiting Wei 1632738@tongji.edu.cn  
Qi Liu qiliu@tongji.edu.cn  
Tongji University, Shanghai, China
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


