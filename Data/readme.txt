The pipeline for the preparation of the datasets
1. You can choose the omic types according to your own projects. In our demo experiments, we provide omics data, clinical information (including the survival status) of brain lower grade glioma (LGG) of 475 patients downloaded from LinkedOmics database and functional gene modules of 860 pathways in the LGG folder.
Note: The different types of omics data and clinical information in one expriment have to be from the same cohorts of patients. and the omics data have been normalzied at the gene level. 

2. Then, you can use the TransferData.py to process the input data into 'train.csv','test.csv','val.csv'and 'pathway_module_input.csv' files in the output folders. 
The users could choose at least one type of omics data. We provide example of pre-process omics data of one single type of omics data of RNA gene expression data and two types of omics data of methylation and RNA expression. We create folder of Single and Multiple and process the original omics data to get the format DeepMusics needs.

3. You can change the input data file name and path to run the demo. We provide the MultiRun.py and SingleRun.py. 

In addition, some other clinical information including age,sex,weight or subtype,stage, etc. could also be the predicted labels.
If your data are not in the gene level, the pathway layer have to be redefined accordingly.

