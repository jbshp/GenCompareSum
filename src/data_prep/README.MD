#  Step 1. Download Stanford CoreNLP
We will need Stanford CoreNLP to tokenize the data. Download it and follow the instructions [here](https://stanfordnlp.github.io/CoreNLP/) and unzip it.

# Step 2. Download and prep datasets 

## PubMed and ArxiV datasets

### Download
Associated with the paper: [A discourse-aware attention model for abstractive summarization of long documents (Cohan et al., 2018)](https://aclanthology.org/N18-2097.pdf). Download the data sets from [here](https://github.com/armancohan/long-summarization). Put all raw files in a directory.

### Preprocess

To preprocess PubMed and Arxiv and data for GenCompareSum and comparative methods implemented in ./comparative_methods, run:

```
python ./src/data_prep/preprocess.py -mode preprocess_pubmed_for_GenCompareSum -raw_path </path/to/raw_data> -save_path </path/to/preprocessed/data>  -log_file ./preprocess.log
```
To preprocess PubMed and Arxiv and data for use with [BERTExtSum](https://github.com/nlpyang/PreSumm)

```
python ./src/data_prep/preprocess.py -mode preprocess_pubmed_for_GenCompareSum -raw_path </path/to/raw_data> -save_path </path/to/preprocessed/data>  -log_file ./preprocess_bertsum.log
```
For example usage, run:
```
python ./src/data_prep/preprocess.py -mode preprocess_pubmed_for_GenCompareSum -raw_path ./data/sample_raw/pubmed/ -save_path ./data/output/  -log_file ./preprocess.log
python ./src/data_prep/preprocess.py -mode preprocess_pubmed_for_BERTSum -raw_path ./data/sample_raw/pubmed/ -save_path ./data/output/  -log_file ./preprocess.log
```


## allenai (CORD-19 & S2ORC) datasets

### CORD-19 download
Download and unzip the `CORD-19` directories from [here](https://allenai.org/data/cord-19). More info about working with the data set can be found [here](https://github.com/allenai/cord19). Put all files in the directory `./data/raw/`

### S2ORC download
You must contact owners to get access to this dataset - details can be found [here](https://github.com/allenai/s2orc). It is in the same format as CORD-19.

### Pre-process
To preprocess CORD-19 and S2ORC and data for GenCompareSum and comparative methods implemented in ./comparative_methods, run the command below. The data sets are not split into train/val/test splits by default. We have added this into the preprocessing to produce a randomly sampled 75/15/10 split. 

```
python ./src/data_prep/preprocess.py -mode preprocess_allenai_datasets_for_GenCompareSum -raw_path </path/to/raw_data> -save_path </path/to/preprocessed/data>  -log_file ./preprocess.log
```
To preprocess CORD-19 and S2ORC and data for use with [BERTExtSum](https://github.com/nlpyang/PreSumm)

```
python ./src/data_prep/preprocess.py -mode preprocess_allenai_datasets_for_GenCompareSum -raw_path </path/to/raw_data> -save_path </path/to/preprocessed/data> -log_file ./preprocess_bertsum.log
```

For example usage, run:
```
python ./src/data_prep/preprocess.py -mode preprocess_allenai_datasets_for_GenCompareSum -raw_path ./data/sample_raw/S2ORC/ -save_path ./data/output/  -log_file ./preprocess.log
python ./src/data_prep/preprocess.py -mode preprocess_allenai_datasets_for_BERTSum -raw_path ./data/sample_raw/S2ORC/ -save_path ./data/output/  -log_file ./preprocess.log
```




