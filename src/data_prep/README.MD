### Step 1. Download datasets 

#### CORD-19 dataset
Download and unzip the `CORD-19` directories from [here](https://allenai.org/data/cord-19). Put all files in the directory `./raw_data`

#### PubMed and ArxiV dataset
Associated with the paper: [A discourse-aware attention model for abstractive summarization of long documents (Cohan et al., 2018)](https://aclanthology.org/N18-2097.pdf). Download the data sets from [here] (https://github.com/armancohan/long-summarization). You can also use the command below to download the files via the cli using linux. Put all files in directory `./data/raw/`.

#### CORD-19 dataset
Download and unzip the `CORD-19` directories from [here](https://allenai.org/data/cord-19). Put all files in the directory `./data/raw/`

####  S2ORC dataset
You must contact owners to get access to this dataset - details can be found [here] (https://github.com/allenai/s2orc). 

###  Step 2. Download Stanford CoreNLP
We will need Stanford CoreNLP to tokenize the data. Download it [here](https://stanfordnlp.github.io/CoreNLP/) and unzip it. Then add the following command to your bash_profile (`/.bashrc` file):
```
 for file in `find /path/to/stanford-corenlp-4.2.1  -name "*.jar"`; do export CLASSPATH="$CLASSPATH:`realpath $file`"; done
```
replacing `/path/to/` with the path to where you saved the `stanford-corenlp-4.2.0` directory. 

###  Step 3. Cleaning data and and Tokenization

For CORD19 or S2ORC data (both from allenai), use the following command to preprocess the data

```
python src/data_prep/preprocess.py -mode tokenize_allenai_datasets -raw_path ./data/raw/ -save_path ./data/token_data/ -log ./tokenize_allenai.log
```

For the PubMed and Arxiv datasets: 
```
python ./src/data_prep/preprocess.py -mode tokenize_pubmed_dataset -raw_path ./data/sample_raw/pubmed/ -save_path ./data/sample_raw/pubmed/  -log ./tokenize_pubmed.log
```


###  Step 4. Format to JSON Files
 
```
python src/data_prep/preprocess.py -mode format_to_lines -raw_path ./data/token_data/ -save_path ./data/json_data -log ./tokenize.log
```


###  Step 5. (GenCompareSum) - Create csv files for GenCompareSum 
 
```
python src/data_prep/prepare_json_to_csv.py -input_path ./data/json_data/ -save_path ./data/preprocessed/ -log ./tokenize.log
```


###  Step 5. (BERTExtSum comparative method) - Format to PyTorch Files
```
python src/preprocess.py -mode format_to_bert -raw_path ./json_data/ -save_path ./bert_data/  -lower -n_cpus 1 -log_file ./logs/preprocess.log 
```

* `JSON_PATH` is the directory containing json files, `BERT_DATA_PATH` is the target directory to save the generated binary files
* Note depending on model type you want to use, you can change `format_to_bert` to `format_to_pubmed_bert` or `format_to_robert`

