###  Step 1. Download Stanford CoreNLP
We will need Stanford CoreNLP to tokenize the data. Download it and follow the instructions [here](https://stanfordnlp.github.io/CoreNLP/) and unzip it.

### Step 2. Download and prep datasets 

#### CORD-19 dataset
Download and unzip the `CORD-19` directories from [here](https://allenai.org/data/cord-19). Put all files in the directory `./raw_data`

#### PubMed and ArxiV dataset
Associated with the paper: [A discourse-aware attention model for abstractive summarization of long documents (Cohan et al., 2018)](https://aclanthology.org/N18-2097.pdf). Download the data sets from [here] (https://github.com/armancohan/long-summarization). You can also use the command below to download the files via the cli using linux. Put all raw files in directory and run the following command:

```
python ./src/data_prep/preprocess.py -mode preprocess_pubmed_for_GenCompareSum -raw_path </path/to/raw_data> -save_path ./data/preprocesed_data/  -log_file ./to_lines.log
```

#### CORD-19 dataset
Download and unzip the `CORD-19` directories from [here](https://allenai.org/data/cord-19). Put all files in the directory `./data/raw/`

####  S2ORC dataset
You must contact owners to get access to this dataset - details can be found [here] (https://github.com/allenai/s2orc). 

