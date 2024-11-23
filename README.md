# SWEEM: Multi-Omics Transformer for Cancer Survival Analysis

## Introduction of SWEEM
Since biological processes associated with cancer are complex and multifaceted, accurately diagnosing cancer progression status is a critical step toward developing strategies for treatment and prevention. To achieve this goal, a multi-omics approach shows great promise for more accurate and biologically-conscious in-silico cancer discovery.  Here we trained a multimodal transformer encoder model called SWEEM on an integrated dataset that combines data across the transcriptome, proteome, and epigenome. We observe that the transformer model, compared with machine learning and deep learning baselines, outperforms previous methods by a comfortable margin. Moreover, we applied various interpretative methods to understand the model’s decision-making and detected biological significance. Overall, SWEEM successfully identifies relevant biological features associated with cancer and can be used to accurately assess the cancer survival of patients from multi-omics datasets.

## Install
To use SWEEM, run ```python sweem.py```. You can run the model with pre-trained model weights by only running line 78, or if you want to load the weights from inference, than only run line 85.
Once running and saving the model weights, running through the visualizations.ipynb notebook will be sufficient.

