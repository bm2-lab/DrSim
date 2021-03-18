# Dr.Sim User Manual
## General framework of Dr.Sim
* **Dr.Sim** is a learning-based framework designed to assign a label to a query transcriptional signature by measuring the similarity between the query transcriptional 
signature and each reference transcriptional signature cluster centroid using a measurement learned from reference transcriptional signatures, rather than empirically
designing. The basic idea of Dr. Sim is a similarity learning schema which aims at making query signatures and reference signatures belonging to identical class become
more similar, while query signatures and reference signatures belonging to different classes become more dissimilar. For illustration purpose, LINCS is used as the reference transcriptional signatures data resource since it holds the largest-scale signatures that produced by treating human cancer cell lines with different compounds under different conditions. Nevertheless, the application of Dr.Sim is not restricted to LINCS and it can be applied directly to other transcriptional perturbation-based data resources for phenotypic drug discovery. Basically, Dr.Sim comprises three main steps: data preprocessing, model training, and similarity calculation.
  * **Data preprocessing**:  
    * **Preparation of reference signature**: (1) Transcriptional signatures treated by compounds for 6H or 24H in the nine human cancer cell lines in LINCS are retained; 
  (2) signatures in LINCS are split into 18 subsets according to cell type and time-point.
