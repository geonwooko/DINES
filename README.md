# DINES
This is the official implementation of **DINES** (Disentangled Neural Networks for Signed Digraph). 
The paper is submitted to the journal track of ECML PKDD 2023, and under review:

* Learning Disentangled Representations in Signed Digraphs for Accurate Sign Prediction <br/>
  Geonwoo Ko and Jinhong Jung<br/>
  Journal track of ECML PKDD 2023 (submitted)

## Overview
Node representation learning is essential for mining signed graphs with diverse applications. However, most existing methods have limitations in relying on domain knowledge of signed social graphs while producing entangled embeddings. These hinder the models from capturing various latent factors inherent in the formation of signed edges, limiting their performance. 

In this paper, we propose DINES (Disentangled Neural Networks for Signed Digraphs), a new GNN-based model for accurate sign prediction in signed digraphs. Our main contributions are summarized as follows:
*  **Disentangled signed digraph encoder**: We design a new encoder that models the disentangled representation of a node so that it captures latent factors without using the social theories. Our encoder separates an embedding into multiple factors, and only considers the direction and sign of edges when aggregating disentangled factors of neighbors.
* **Pairwise correlation decoder**: We propose a novel decoder for link sign prediction, which predicts the sign of an edge whose feature is built with pairwise correlations between disentangled factors of two nodes. We empirically show that our strategy produces better edge features than simply concatenating the node embeddings.
* **Enhanced disentanglement**: We adopt a graph self-supervised approach classifying each factor to enhance the disentanglement of representations. We jointly train the self-supervised classifier on top of our encoder and decoder, and show its effect on representation learning in signed graphs.
* **Scalable algorithm**: We theoretically analyze that our algorithms for DINES have linear scalability with respect to the number of edges.

## Prerequisites
The packages used in this repository are as follows:
```
python>=3.9
torch==1.13.1
numpy==1.23.1
scipy==1.9.3
scikit-learn==1.2.0
torchmetrics==0.8.1
loguru==0.6.0
fire==0.5.0
tqdm==4.64.1
```

## Datasets 
We provide datasets used in the paper for reproducibility. 
You can find raw datasets at `./data/${DATASET}` folder where the file's name is `edges.csv`. 
The `${DATASET}` is one of `BC_ALPHA`, `BC_OTC`, `WIKI_RFA`, `SLASHDOT`, and `EPINIONS`.
This file contains the list of signed edges where each line consists of a tuple of `(src, dst, sign)`.
The details of datasets are provided in the following table:
|**Dataset**|**$\|\mathcal{V}\|$**|**$\|\mathcal{E}\|$**|**$\|\mathcal{E}^{+}\|$**|**$\|\mathcal{E}^{-}\|$**|**$p$(+)**|
|:-:|-:|-:|-:|-:|:-:|
|[BitcoinAlpha](https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html)|3,783|24,186|22,650|1,536|93.6|
|[BitcoinOTC](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html)|5,881|35,592|32,029|3,563|90.0|
|[Wiki-RFA](https://snap.stanford.edu/data/wiki-RfA.html)|11,258|178,096|138,473|38,623|78.3|
|[Slashdot](http://konect.uni-koblenz.de/networks/slashdot-zoo)|79,120|515,581|392,326|123,255|76.1|
|[Epinions](http://www.trustlet.org/wiki/Extended_Epinions_dataset)|131,828|841,372|717,667|123,705|85.3|
* $\|\mathcal{V}\|$: the number of nodes
* $\|\mathcal{E}\|$: the number of edges
* $\|\mathcal{E}^{+}\|$ and $\|\mathcal{E}^{-}\|$: the numbers of positive and negative edges, respectively
* $p$(+): the ratio of positive edges

## Demo
You can run the simple demo by typing the following command in your terminal:
```bash
bash demo.sh
```

This trains DINES on the `BC_ALPHA` dataset with the hyperparameters stored at `./pretrained/BC_ALPHA/config.json`. 
After the training phase completes, the trained model is saved as `encoder.pt` and `decoder.pt` at the folder `./output/BC_ALPHA`. 
Then, it evaluates the trained model on the link sign prediction task in terms of AUC and Macro-F1.

## Pre-trained DINES
We provide pre-trained models of DINES for each data stored at `./pretrained/${DATASET}` folder where the file names are `encoder.pt` and `decoder.pt`.
The hyperparameters used for training them are reported in the Appendix section of the paper, and they are saved in `./pretrained/${DATASET}/config.json`.

## Results of Pre-trained DINES
The results of the pre-trained models are as follows:
|**Dataset**|**AUC**|**Macro-F1**|
|:-:|:-:|:-:|
|**BC_ALPHA**|0.924|0.801|
|**BC_OTC**|0.946|0.858|
|**WIKI_RFA**|0.914|0.802|
|**SLASHDOT**|0.927|0.821|
|**EPINIONS**|0.967|0.892|

All experiments are conducted on RTX 3090 (24GB) with cuda version 12.0, and the above results were produced with the random seed `seed=1`.

## How to Reproduce the Above Results with the Pre-traied Models
You can reproduce the results the following command which evaluates a test dataset using a pre-trained model.

```bash
python ./src/run_evaluate.py --input-dir ./pretrained --dataset ${DATASET} --gpu-id ${GPU_ID}
```

The pre-trained models were generated by the following command:

```bash
python ./src/run_train.py --load-config --output_dir ./pretrained --dataset ${DATASET} --gpu-id ${GPU_ID}
```

## Detailed Usage and Options
You can train and evaluate with your own datasets or custom hyperparmeters using `run_train.py` and `run_evaluate.py`.

### Training
You can perform the training process of DINES with the following command:
```bash
python src/run_train.py [--<argument name> <argument value>] [...]
```
We describe the detailed options of `src/run_train.py` in the following table:

|**Option**|**Description**|**Default**|
|:-:|:-:|:-:|
|`load-config`|whether to load the configuration used in a pre-trained model|False|
|`dataset`|dataset name|BC_ALPHA|
|`data-dir`|data directory path|./data|
|`output-dir`|output directory path|./output|
|`test-ratio`|ratio of test edges|0.2|
|`gpu-id`|GPU id; If None, a CPU is used|None|
|`seed`|random seed; If None, the seed is not fixed|None|
|`in-dim`|input feature dimension|64|
|`out-dim`|output embedding dimension|64|
|`num-epochs`|number of epochs|100|
|`lr`|learning rate $\eta$ of an optimizer|0.005|
|`weight-decay`|strength $\lambda_{\texttt{reg}}$ of L2 regularization|0.005|
|`num-factors`|number $K$ of factors |8|
|`num-layers`|number $L$ of layers |2|
|`lambda-disc`|strength $\lambda_{\texttt{disc}}$ of the discriminative loss|0.1|

* Note that several PyTorch APIs such as `torch.index_add_` run non-deterministically on a GPU [[link]](https://pytorch.org/docs/stable/notes/randomness.html); thus, the results on the GPU could be slightly different every run although we fix the random seed (but, the difference is not statistically significant). 
* For a strict reproducibility, we provide an additional option using a CPU, i.e., `--device=None` forces the code to run on the CPU, and makes the procedure deterministic by setting `torch.use_deterministic_algorithms(True)`. If you want PyTorch to use its non-deterministic algorithms on the CPU, please remove the function call from the code.


### Evaluation
We provide a script that evaluates the trained model of DINES, and reports AUC and Macro-F1 scores on a test dataset.
This uses `encoder.pt`, `decoder.pt`, and `config.json`; thus, you first need to check tif they are appropriately generated by `./src/run_train.py`. Note that it uses the same random seed used by `./src/run_train.py` where the seed is saved at `config.json` so that the test dataset is valid for the evaluation.

```
python src/run_evaluate.py [--<argument name> <argument value>] [...]
```

We describe the detailed options of `src/run_evaluate.py` in the following table:

|**Option**|**Description**|**Default**|
|:-:|:-:|:-:|
|`dataset`|dataset name|BC_ALPHA|
|`input-dir`|directory path where a pre-trained DINES is stored|./output|
|`gpu-id`|GPU id; If None, a CPU is used|None|
