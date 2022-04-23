from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
import pandas as pd
import ast

from pyrouge import Rouge155
from tqdm import tqdm
import time
import os
import shutil
import argparse
import numpy as np


def test_rouge(predicted_summaries, gold_summaries):
    """Calculate ROUGE scores of sequences passed as an iterator
       e.g. a list of str, an open file, StringIO or even sys.stdin
    """
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = ".rouge-tmp-{}".format(current_time)
    try:
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
            os.mkdir(tmp_dir + "/candidate")
            os.mkdir(tmp_dir + "/reference")
        print('preparing predicted summaries')
        candidates = [line.strip() for line in tqdm(predicted_summaries,total=len(predicted_summaries))]
        print('preparing gold summaries')
        gold = [line.strip() for line in tqdm(gold_summaries,total=len(gold_summaries))]
        assert len(candidates) == len(gold)
        cnt = len(candidates)
        print('Writing temp files')
        for i in tqdm(range(cnt)):
            if len(gold[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(gold[i])
        print("Doing ROUGE calculation")
        r = Rouge155()
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        results_dict = r.output_to_dict(rouge_results)
        return results_dict
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir) 
    
def format_rouge_results(results):
    return f"ROUGE-F(1/2/l)/ROUGE-R(1/2/l): {results['rouge_1_f_score']}/{results['rouge_2_f_score']}/{results['rouge_l_f_score']} /{results['rouge_1_recall']}/{results['rouge_2_recall']}/{results['rouge_l_recall']}"

def combine_array_sentences(sentence_array):
    combined = ''
    for sentence in sentence_array:
        combined+=('\n '+sentence)
    return combined

def sumy_summarizers(df,col,num_sentences,method):

    if method == 'TextRank':
        summarizer = TextRankSummarizer()
    elif method == 'LexRank':
        summarizer = LexRankSummarizer()
    else:
        summarizer = SumBasicSummarizer()

    gold_summaries = []
    our_predictions = []
    for idx, row in tqdm(df.iterrows(),total=len(df)):
        doc_text = ast.literal_eval(df.loc[idx,col])
        text = combine_array_sentences(doc_text)
        my_parser = PlaintextParser.from_string(text,Tokenizer('english'))
        lexrank_summary = summarizer(my_parser.document,sentences_count=num_sentences)
        summary = []
        for sentence in lexrank_summary:
            sentence = str(sentence).replace('<Sentence: ','').replace('.>','.')
            summary.append(sentence)
        pred = combine_array_sentences(summary)
        our_predictions.append(pred)
        
        gold_sum = df.loc[idx,'summary_text_combined']
        gold_summaries.append(gold_sum)

    return our_predictions, gold_summaries


def random_summarizer(df,col,num_sentences):
    gold_summaries = []
    our_predictions = []
    for idx, row in tqdm(df.iterrows(),total=len(df)):    
        doc_text = np.array(ast.literal_eval(df.loc[idx,col]))
        if len(doc_text)>num_sentences:
            doc_text =doc_text[np.random.randint(0,len(doc_text),num_sentences)]
        pred = combine_array_sentences(doc_text)
        our_predictions.append(pred)
        gold_sum = df.loc[idx,'summary_text_combined']
        gold_summaries.append(gold_sum)
    return our_predictions, gold_summaries

def lead_summarizer(df,col,num_sentences):
    gold_summaries = []
    our_predictions = []
    for idx, row in tqdm(df.iterrows(),total=len(df)):    
        doc_text = np.array(ast.literal_eval(df.loc[idx,col]))
        if len(doc_text)>num_sentences:
            doc_text = doc_text[0:num_sentences]
        pred = combine_array_sentences(doc_text)
        our_predictions.append(pred)
        
        gold_sum = df.loc[idx,'summary_text_combined']
        gold_summaries.append(gold_sum)
    return our_predictions, gold_summaries

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv",required=True)
    parser.add_argument("--num_sentences",required=True)
    parser.add_argument("--col_name",required=True)
    parser.add_argument("--method",choices=['LexRank','TextRank','SumBasic','RANDOM','LEAD'])
    args = parser.parse_args()
    num_sentences = int(args.num_sentences)
    col_name = args.col_name
    method = args.method

    df = pd.read_csv(args.data_csv)

    if ((method == 'LexRank') or (method == 'TextRank') or (method == 'SumBasic')):
        pred,gold = sumy_summarizers(df,col_name,num_sentences,method)
    elif method =='RANDOM':
        pred,gold = random_summarizer(df,col_name,num_sentences)
    elif method =='LEAD':
        pred,gold = lead_summarizer(df,col_name,num_sentences)

    print(format_rouge_results(test_rouge(pred,gold)))


