# GenCompareSum

This is the repository asssociated with the paper 'GenCompareSum: a hybrid unsupervised summarization method using salience'


# Set Up

This repo requiress python>=3.7.
To install the dependencies for this repo:
```
pip install -r requirements.txt
```

Run GenCompareSum on your Data
```
python GenCompareSum.py --num_questions 10 --block_n_gram_questions 4 --col_name article_text --batch_size 64 --summary_len_metric sentences --bert_score_model_type bert-base-uncased --num_sentences 6 --block_n_gram_sum 4   --visible_device 0 --q_per_section 3 --temperature 0.5 --stride 4 --data_path ./data/test_data.csv --query_generation_model_path doc2query/S2ORC-t5-base-v1  --similarity_model_name bert_score 
```

Prepare Data Sets


Train models

Comparitive Methods

