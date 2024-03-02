# DINES
This is the official implementation of **DINES** (Disentangled Neural Networks for Signed Digraph). 

* Learning Disentangled Representations in Signed Directed Graphs without Social Assumptions <br/>
  Geonwoo Ko and Jinhong Jung<br/>
  Information Sciences (accepted)
* The pre-print version is available at [here](https://arxiv.org/abs/2307.03077).

## Overview
Signed graphs can represent complex systems of positive and negative relationships such as trust or preference in various domains.
Learning node representations is indispensable because they serve as pivotal features for downstream tasks on signed graphs.
However, most existing methods often oversimplify the modeling of signed relationships by relying on social theories, while real-world relationships can be influenced by multiple latent factors.
This hinders those methods from effectively capturing the diverse factors, thereby limiting the expressiveness of node representations.  

In this paper, we propose DINES, a novel method for learning disentangled node representations in signed directed graphs without social assumptions. 
We adopt a disentangled framework that separates each embedding into distinct factors, allowing for capturing multiple latent factors. 
We also explore lightweight graph convolutions that focus solely on sign and direction, without depending on social theories. Additionally, we propose a decoder that effectively classifies an edge's sign by considering correlations between the factors.
To further enhance disentanglement, we jointly train a self-supervised factor discriminator with our encoder and decoder. 

## Prerequisites
The packages used in this repository are as follows:
```
python==3.9.16
numpy==1.24.3
pytorch==2.0.1
pytorch-cuda==11.7
pytorch-scatter==2.1.1
scikit-learn==1.2.2
scipy==1.10.1
fire==0.5.0
loguru==0.7.0
torchmetrics==0.8.1
tqdm==4.65.0
```
You can create a conda environment with these packages by typing the following command in your terminal:
```bash
conda env create --file environment.yml
conda activate DINES
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
|[Slashdot](http://konect.cc/networks/slashdot-zoo)|79,120|515,397|392,326|123,255|76.1|
|[Epinions](https://snap.stanford.edu/data/soc-sign-epinions.html)|131,828|841,372|717,667|123,705|85.3|
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
|**BC_ALPHA**|0.937|0.789|
|**BC_OTC**|0.950|0.860|
|**WIKI_RFA**|0.914|0.786|
|**SLASHDOT**|0.927|0.831|
|**EPINIONS**|0.967|0.895|

All experiments are conducted on RTX 3090 (24GB) with cuda version 12.0, and the above results were produced with the random seed `seed=1`.

## How to Reproduce the Above Results with the Pre-traied Models
You can reproduce the results the following command which evaluates a test dataset using a pre-trained model.

```bash
python ./src/run_evaluate.py --input-dir ./pretrained --dataset ${DATASET} --gpu-id ${GPU_ID}
```

The pre-trained models were generated by the following command:

```bash
python ./src/run_train.py --load-config --output_dir ./pretrained --dataset ${DATASET} --seed 1 
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
|`aggr-type`|aggregator type (sum, max, mean, attn) |sum|

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
