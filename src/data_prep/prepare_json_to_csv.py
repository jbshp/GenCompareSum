import os
from pydoc import describe
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import json
from copy import deepcopy
import argparse

input_path = 'path/to/json/data'
save_path = '/mnt/disk/jenny/qg-qfs/data/s2orc/csv_prepro_data'
parser = argparse.ArgumentParser()
parser.add_argument("--input_path",required=True,help='path to json files containing articles to summarize and their target summaries')
parser.add_argument("--save_path",required=True,help='path to location to store csv files')
args = parser.parse_args()
save_path = args.save_path
input_path = args.input_path


files = os.listdir(input_path)
files = [file for file in files if '.json' in file]

new_articles = []
for file in files:
    data_split = file.split('.')[0]
    print(f'starting file {file}')
    with open(f"{input_path}/{file}",'r') as f:
        articles = json.load(f)
    
    for article in tqdm(articles):
        # process article text       
        new_article = []
        short_article = []
        token_count = 0
        short_article_token_count = 0
        text = article['src']
        for sentence in text:
            new_sentence = ''
            for token in sentence:
                new_sentence+=f" {token}"
            new_article.append(new_sentence)
            if token_count<512:
                short_article.append(new_sentence)
                short_article_token_count += len(sentence)
            token_count+=len(sentence)
        
        # process summary text      
        abs_text = article['tgt']
        new_summary = ''
        summary_count = 0
        for sentence in abs_text:
            new_sentence = ''
            summary_count+=len(sentence)
            for token in sentence:
                new_sentence+=f" {token}"
            new_summary+=f"\n{new_sentence} "
            
        new_articles.append({
            'article_text':new_article,
            'summary_text_combined':new_summary,
            'article_sentences':len(text),
            'article_tokens':token_count,
            'summary_sentences':len(abs_text),
            'summary_tokens':summary_count,
            'data_split':data_split,
            'short_article': short_article,
            'short_article_tokens': short_article_token_count,
            'short_article_sentences': len(short_article)
        })

df = pd.DataFrame(new_articles)
df['article_tokens'].mean()
df['article_sentences'].mean()
df['summary_sentences'].mean()
df['summary_tokens'].mean()
df['short_article_tokens'].mean()
df['short_article_sentences'].mean()

df = df[['article_text','short_article','summary_text_combined']]

for d_split in ['test','train','valid']:
    df_split = df[df['data_split']==d_split]
    print(f'Number of articles in {d_split} set: {len(d_split)}')
    df_split.to_csv(os.path.join(save_path,f'{d_split}.csv'),index=False)

