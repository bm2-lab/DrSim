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
if you have docker installed, you call pull the images:  

    docker pull bm2lab/dr.sim

#### Install from github   

    git clone https://github.com/bm2-lab/Dr.Sim.git  
    
## Usage
Dr.Sim can be applied for drug annotation:  

    python  DrugAno.py --help
    
Drug repositioning  

    python  DrugRep.py  --help
    
## User Manual
For detailed information about usage, input and output files, test examples and data preparation, please refer to the [Dr.Sim User Manual](/doc/Dr.Sim_User_Manual.md).
 
## Dr.Sim flowchart
![](workflow.png)
 

## Citation:
Similarity learning for transcriptional phenotypic drug discovery, submitted, 2021.

## Contact
Zhiting Wei 1632738@tongji.edu.cn  
Qi Liu qiliu@tongji.edu.cn  
Tongji University, Shanghai, China
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


