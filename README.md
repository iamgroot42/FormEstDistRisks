# Formalizing and Estimating Distribution Inference Risks

This is the code for reproducing the experiments reported in this paper:
[_Formalizing and Estimating Distribution Inference Risks_](https://arxiv.org/abs/2109.06024).

Exploratory experiments and meta-classifier property-inference attacks on various datasets across several domains.
Folder structure:

- `boneage` : Experiments on the [RSNA Bone-Age](https://www.kaggle.com/kmader/rsna-bone-age) task, reduced to binary prediction
- `arxiv` : Experiments on the [ogbn-arxiv](https://ogb.stanford.edu/docs/nodeprop/) node-classfication task
- `census` : Experiments on the [Adult Income](https://www.kaggle.com/uciml/adult-census-income) dataset
- `celeba` : Experiments on the [Celeb-A](https://www.kaggle.com/jessicali9530/celeba-dataset) dataset
- `botnets`: Experiments on [HarvardNLP's Botnet Detection](https://github.com/harvardnlp/botnet-detection) datasets

Each folder has its own README.md file that explains how to reproduce those experiments.
