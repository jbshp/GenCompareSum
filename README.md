# GenCompareSum

This is the repository asssociated with the paper 'GenCompareSum: a hybrid unsupervised summarization method using salience'

# Data
An example of how data should be formatted in included in the ./data/pubmed_sample.csv. `GenCompareSum.py` expects to be pointed to a csv file which contains:
* A column containing the article you want to summarize, split into sentences and saved in an array. 
* A column called 'summary_text_combined' which contains your target summary, which is used for the ROUGE calculated and compared with the generated summary. It is **important** to note that the summary should be split with `\n` characters between sentences. 

In our example data, we have columns `article_text`, which contains the full articles which have been split into sentences using the [StanfordNLP](https://stanfordnlp.github.io/CoreNLP/ssplit.html) software package, and a columns `short_article` which contains the same data, but truncated at the end of sentence which contains the 512th token. 

# Set Up

This repo requires python>=3.7.
To install the dependencies for this repo:
```
pip install -r ./requirements.txt
git clone https://github.com/andersjo/pyrouge.git ./src/rouge
pyrouge_set_rouge_path <PATH_TO_THIS_REPO>/src/rouge/tools/ROUGE-1.5.5
```

# Run GenCompareSum 

This code can be run on cpu or gpu, however, runs significantly slower on cpu. 
## To evaluate method with ROUGE metrics

```
python GenCompareSum.py --num_generated_texts 10 --block_n_gram_generated_texts 4 --col_name short_article --summary_len_metric sentences --num_sentences 6 --block_n_gram_sum 4   --visible_device 0 --texts_per_section 3 --temperature 0.5 --stride 4 --gen_text_weights 1 --data_path ./data/pubmed_sample.csv --generative_model_path doc2query/S2ORC-t5-base-v1  --similarity_model_name bert_score --similarity_model_path bert-base-uncased 
```

The config params are as follows:
* `--num_generated_texts` - Number of generated salient texts to select to carry forward to comparative step
* `--block_n_gram_generated_text` - If set, use an `Int` value to give the number of words which need to be the same consecutively between two texts for one of them to be removed. If unset, no n-gram blocking is applied to generated texts
* `--col_name` - Name of the column which contains the text you want to summarize 
* `--summary_len_metric` - Metric by which to get the target length of the summary, either `sentences` or `tokens`
* `--num_sentences` - Number of sentences to select for predicted exractive summary
* `--target_tokens` - Target number of tokens to aim for in predicted summary. 
* `--block_n_gram_sum` - If set, use an `Int` value to give the number of words which need to be the same consecutively between two texts for one of them to be removed. If unset, no n-gram blocking is applied to generated texts
* `--visible_device` - gpu device to use, if using a gpu
* `--gen_text_weights` -  Set to `1` if you would like to use the scores associated with generated texts to weight the similarity calculation
* `--temperature`- Temperature parameter of generative model
* `--texts_per_section` - Number of salient texts to generate per section
* `--stride` - Number of sentences to combine to make a section to feed into the generative model
* `--data_path` - Path to csv containing data to summarize
* `--similarity_model_path` - Either name of similarity model to take from [Hugging Face](https://huggingface.co), or local path to model
* `--generative_model_path` - Either name of T5-based generative model to take from [Hugging Face](https://huggingface.co), or local path to model
* `--similarity_model_name` - Type of similarity comparison model, either `bert_score`, `sentence_transformers` or `simcse`.


