import json
import os
from typing import List, Iterable
import ast
import nltk
import torch
import pandas as pd
from tqdm import tqdm
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
import bert_score
import random
import math
import functools

from pyrouge import Rouge155
import time
import shutil
import numpy as np
import re
from datetime import datetime
from nltk import word_tokenize
from simcse import SimCSE
import argparse
from scipy import spatial
from sentence_transformers import SentenceTransformer

from IPython.utils import io


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return value
    return wrapper_timer


def preprocess(document: str, stride=5, list=False) -> List[str]:
    """
    This function takes a corpus document and outputs a list of generation
    spans where the 'stride' is the number of sentences in each section. 
    """
    if list==False:
        sentences = nltk.tokenize.sent_tokenize(document)
    else:
        sentences = document
    chunks = [" ".join(sentences[i:i+stride]) for i in range(0, len(sentences), stride)]

    return chunks
    


def generate_questions(
        text,
        model,
        tokenizer,
        device,
        num_q_per_section,
        temperature,
        max_len
    ):
    """
    This function takes a text passage and generate a list of questions
    """

    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    
    questions = []
    try:
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_len,
            do_sample=True,
            top_k=10,
            num_return_sequences=num_q_per_section,
            temperature=temperature
        )
        questions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs] 
    except RuntimeError:
        print(len(input_ids))

    return questions


def get_questions_across_corpus(
        model,
        tokenizer,
        doc_text,
        device,
        stride,
        num_q_per_section,
        temperature,
        top_k_questions,
        block_n_gram
    ):
    """
    This function takes a document, which is pre-split into an array, with one sentence per element.  
    The function then combines several sentences into paragraphs (num sentences in section is 'stride').
    Several questions are then geenrated per section. 
    The questions are combined and the most frequent k questions generated from across the whole corpus are taken
    """
    text_split_into_sections = preprocess(doc_text,stride=stride,list=True)
    questions = [
        generate_questions(
            span,
            model,
            tokenizer,
            device,
            num_q_per_section,
            temperature=temperature,
            max_len=64
        ) for span in text_split_into_sections
    ]
    question_df = pd.DataFrame([
        dict(
            document_id=doc_idx,
            span_id=f"{doc_idx}:{span_idx}",
            gen_id=f"{doc_idx}:{span_idx}",
            question=question,
        )
        for doc_idx, document_gen in enumerate(questions)
        for span_idx, question in enumerate(document_gen)
    ])

    questions_grouped_tbl = question_df \
        .groupby("question") \
        .nunique() \
        .sort_values("gen_id", ascending=False)
    
    top_questions = list(questions_grouped_tbl.index[0:top_k_questions])
    top_weight = np.array(questions_grouped_tbl.gen_id[0:top_k_questions])
    
    
    # experiment with trigram blocking version   
    if (block_n_gram):
        _pred = []
        _weight = []
        for candidate, weight in zip(questions_grouped_tbl.index,questions_grouped_tbl.gen_id):
            idx_ngram_blocker = _block_n_gram(block_n_gram,candidate, _pred,True)
            if (idx_ngram_blocker != False):
                _weight[idx_ngram_blocker] =  _weight[idx_ngram_blocker] + weight
            else:
                _pred.append(candidate)
                _weight.append(weight)
        _pred = np.array(_pred)
        _weight = np.array(_weight)
        _pred = _pred[np.argsort(-_weight)]
        _weight = _weight[np.argsort(-_weight)]
        top_questions_trigram_block = _pred[0:top_k_questions]
        weight_trigram_block = _weight[0:top_k_questions]      

        return top_questions, top_weight, top_questions_trigram_block, weight_trigram_block
    
    return top_questions, top_weight


    
def flatten(t):
    return [item for sublist in t for item in sublist]  


def dedupe_doc_text(seq):
    seen = set()
    seen_add = seen.add
    return [x.replace('\n','') for x in seq if not (x in seen or seen_add(x))]

 
def calculate_similarity_bert_score(
        model_bertscore,
        tokenizer_bertscore,
        questions,
        doc_text,
        model_type,
        all_layers,
        num_layers,
        device,
        batch_size
    ):
    # create an array of questions of length no_questions*num_sentences
    questions_compare_array = flatten(
        np.array([[str(question)]*len(doc_text) for question in questions])
        .astype('str')
    )
    # create an array of sentences of length no_questions*num_sentences
    sentences_compare_array = flatten(np.array([doc_text for question in questions]).astype('str'))
    P_sci, R_sci, F1_sci = bert_score.score(
        questions_compare_array, 
        sentences_compare_array,
        model_bertscore,
        tokenizer_bertscore,
        model_type,
        num_layers=num_layers,
        device=device, 
        verbose=False,
        all_layers=all_layers,
        batch_size=batch_size
    )
    return F1_sci

@timer  
def calculate_similarity_simsce(questions,doc_text,device):
    questions = list(questions)
    return model_simcse.similarity(questions,doc_text)

def calculate_similarity_sentence_transformers(model,questions,doc_text):
    vectors_bert_sent = model.encode(doc_text)
    vectors_bert_question = model.encode(questions)
    scores_marix = np.zeros((len(questions),len(doc_text)))
    for idx_1,ii in enumerate(vectors_bert_sent):
        for idx_2, jj in enumerate(vectors_bert_question):
            scores_marix[idx_2,idx_1] =  spatial.distance.cosine(ii, jj)
    return scores_marix
    

def rank_answers_based_on_similarity_scores(doc_text,questions,scores,similarity_model,question_weights=np.array([])):
    """
    Params:
        doc_text: Array[<string>] : array of sentences in article
        questions: Array[<string>] : array of questions summarising the article
        scores: Array[<number>] : np.array of simialirty scores between each sentence and each question
        question_weights Array[<number>] : np.array of weights associated with importance of each question 
        
    Returns:
        sorted_idxs: Array[<int>] : np.array of indicies of sentencs in doc_text, 
                     sorted in order of importance for summary 
    """
    # reshape so that rows represent different questions and columns represent different sentences in source text 
    scores_reshaped = np.array(scores.reshape(len(questions),len(doc_text)))
    # optionally multiply by weights associated with questions
    if len(question_weights>0):
        scores_reshaped = scores_reshaped*question_weights.reshape(question_weights.size,1)
    # Take mean bert-score for sentences across all questions
    scores_average = np.mean(scores_reshaped,axis=0)
    # sort indicies of scores in descendng order
    if similarity_model =='sentence_transformers':
        # distance based - minimize scores      
        sorted_idxs = np.argsort(scores_average)
    else:
        # siimilarity based - maximize scores
        sorted_idxs = np.argsort(-scores_average)
    return sorted_idxs


def select_sentences(
    doc_text,
    sorted_idxs,
    metric,
    target_tokens,
    top_k_sentences,
    block_n_gram
    ):
    """
    Params:
        doc_text: Array[<string>] : array of sentences in article
        sorted_idxs: Array[<int>] : np.array of indicies of sentencs in doc_text, 
                     sorted in order of importance for summary 
        metric: 'tokens' or 'sentences' : how to calculate how long summary should be
        target_tokens: int : number of tokens to aim for in target summary
        target_tokens: int : number of sentences to aim for in target summary
        question_weights: Array[<number>] : np.array of weights associated with importance of each question 
        block_n_gram: int or None: if int, number of consecutive words 
                      required to match for a sentence to be blocked
        
    Returns:
        sorted_idxs: Array[<int>] : np.array of indicies of sentencs in doc_text, 
                     sorted in order of importance for summary 
    """
    if metric == 'tokens':
        len_sentences = np.array([len(word_tokenize(s)) for s in doc_text])
        article_len = np.sum(len_sentences)
        max_len = target_tokens
    elif metric == 'sentences':
        article_len = len(doc_text)
        max_len = top_k_sentences
        
    #  if article is too short, whole things is summary  
    if article_len <= max_len:
        pred = doc_text
    else :
        _count = 0
        _pred = []
        _pred_idxs = []
        for sentence_idx in sorted_idxs:
            candidate = doc_text[sentence_idx]
            if metric == 'tokens':
                candidate_len = len_sentences[sentence_idx]
            else:
                candidate_len = 1
            if (_count < max_len):
                if (block_n_gram):
                    idx_ngram_blocker = _block_n_gram(block_n_gram,candidate, _pred,False)
                    if (idx_ngram_blocker ==True):
                        continue
                _count += candidate_len
                _pred.append(candidate)
                _pred_idxs.append(sentence_idx)
        sorted_pred_idxs = np.sort(_pred_idxs)
        doc_text = np.array(doc_text)
        pred  = doc_text[sorted_pred_idxs]
    return pred
  
def pick_top_sentences_and_join_into_one_str(
    model,
    tokenizer,
    doc_text,
    questions,
    device,
    num_layers, 
    all_layers,
    metric,
    top_k_sentences,
    weights,
    similarity_model_name,
    block_n_gram,
    target_tokens,
    batch_size,
    bert_model_type
    ):
    """
    Passed an array of strings representing an article  (doc_text)
    and an array of strings representing the questions (questions) which summarise it,
    the function returns one string containing the top k scoring sentences to be included in a summary, 
    in the order that they appear in the text.
    """
            
    # calculate similarity
    with io.capture_output() as captured:
        if similarity_model_name == 'bert_score':
            scores = calculate_similarity_bert_score(
                model,
                tokenizer,
                questions,
                doc_text,
                bert_model_type,
                all_layers,
                num_layers,
                device,
                batch_size
            )
        elif similarity_model_name == 'simcse':
            scores = calculate_similarity_simsce(
                model,
                questions,
                doc_text,
                device
            )
        elif similarity_model_name =='sentence_transformers':
            scores = calculate_similarity_sentence_transformers(model,questions,doc_text)
        else:
            raise ValueError('similarity_model name not recognised.')    
    
    #  sort indexes of doc_text sentences in order of importance   
    sorted_idxs = rank_answers_based_on_similarity_scores(
        doc_text,
        questions,
        scores,
        similarity_model,
        question_weights=weights
    )
    
    # select sentences based on requires number of sentences or
    # tokens and integrate optional trigram blocking
    pred = select_sentences(
        doc_text,
        sorted_idxs,
        metric=metric,
        target_tokens=target_tokens,
        top_k_sentences=top_k_sentences,
        block_n_gram=block_n_gram
        )
    pred_ext_summary = combine_array_sentences(pred)
    
    return pred_ext_summary, sorted_idxs


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
        with io.capture_output() as captured:
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
        combined+=(' \n'+sentence)
    return combined

def _get_ngrams(n, text):
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set

def _block_n_gram(n, c, p,questions=False):
    """
    Params:
        n: int : number of consecutive words required to match for a sentence to be blocked
        c: string : candidate string to compare to array of strings
        p: Array[<string>]: array of strings to compare to
        q: Bool: indicates whether the canddiate and prediction strings are questions 
    """
    tri_c = _get_ngrams(n, c.split())
    for idx, s in enumerate(p):
        if (questions):
            s = s.replace('what is',' ')
            s = s.replace('why is',' ')
            s = s.replace('what is the',' ')
            s = s.replace('how long does', ' ')
        tri_s = _get_ngrams(n, s.split())
        if len(tri_c.intersection(tri_s)) > 0:
            return idx
    return False


def get_question_model(model,device):
    
    tokenizer_doc2query = T5Tokenizer.from_pretrained(model)
    model_doc2query = T5ForConditionalGeneration.from_pretrained(model)
    output = model_doc2query.to(device)
    return model_doc2query, tokenizer_doc2query
    

@timer
def main(df,
        model_doc2query,
        tokenizer_doc2query,
        stride,
        num_q_per_section,
        temperature,
        num_questions,
        block_n_gram_questions,
        similarity_model,
        similarity_tokenizer,
        num_layers,
        all_layers,
        summary_len_metric,
        num_sentences,
        target_tokens,
        block_n_gram_sum,
        batch_size,
        bert_score_model_type,
        device,
        col_name,
        question_weights
        ):

    gold_summaries = []
    our_predictions = []
    our_summary_lens = []
    
    for idx, row in tqdm(df.iterrows(),total=len(df)):
        #  read in article to summarise       
        doc_text = np.array(ast.literal_eval(row[col_name]))
        
        print(combine_array_sentences(doc_text))

        weights = None
        #  generate questions  
        if (block_n_gram_questions):
            questions, freq, q_tg, f_tg = get_questions_across_corpus(
                model_doc2query,
                tokenizer_doc2query,
                doc_text,
                device,
                stride=stride,
                num_q_per_section=num_q_per_section,
                temperature=temperature,
                top_k_questions=num_questions,
                block_n_gram=block_n_gram_questions,
            )
            questions = q_tg
            weights = f_tg if (question_weights) else np.array([])
        else:
            questions, freq= get_questions_across_corpus(
                model_doc2query,
                tokenizer_doc2query,
                doc_text,
                device,
                stride=stride,
                num_q_per_section=num_q_per_section,
                temperature=temperature,
                top_k_questions=num_questions,
                block_n_gram=block_n_gram_questions,
            )
            weights = freq if (question_weights) else np.array([])
        
        #print(questions)

        #  generate summary       
        pred_sum, idxs = pick_top_sentences_and_join_into_one_str(
            similarity_model,
            similarity_tokenizer,
            doc_text,
            questions,
            device,
            num_layers=num_layers, 
            all_layers=all_layers,
            metric=summary_len_metric,
            top_k_sentences=num_sentences,
            weights=weights,
            similarity_model_name=similarity_model_name,
            block_n_gram=block_n_gram_sum,
            target_tokens=target_tokens,
            batch_size=batch_size,
            bert_model_type=bert_score_model_type
        )

        #  append out predicted summary and gold summary to arrays for evaluation       
        our_summary_lens.append(len(word_tokenize(pred_sum)))
        our_predictions.append(pred_sum)       
        gold_sum = df.loc[idx,'summary_text_combined']
        gold_summaries.append(gold_sum)
        
        #print(pred_sum)
        #print(gold_sum)

    #  calculate ROUGE scores
    model_type = bert_score_model_type.split('/')[-1]
    our_pred = test_rouge(our_predictions,gold_summaries)
#     with open(f'our_pred_{num_questions}_{num_sentences}_{model_type}_{col_name}_{batch_size}.json','w') as f:
#         json.dump(our_pred,f)
    
    
    
    our_summary_lens = np.array(our_summary_lens)
    
    # print summaries
    print(f'\n\nData col: {col_name}.\n'+
          f'Num questions: {num_questions}.\n'+
          f'block_n_gram_questions: {block_n_gram_questions}.\n'+
          f'Similarity model type: {model_type}.\n'+
          f'block_n_gram_sum: {block_n_gram_sum}.\n'+
          f'summary_len_metric: {summary_len_metric}.\n'+
          f'Num sentences: {num_sentences}.\n'+
          f'target_tokens: {target_tokens}.\n'+
          f'Batch size: {batch_size}.\n\n'+
          f'{format_rouge_results(our_pred)}\n'+
          f'Average length of summary: {np.mean(our_summary_lens)}'
         )
    print(f'Question weights: {question_weights}')
    print()





if __name__=='__main__':
    
    # -------- DEFINE EXPERIMENT PARAMS --------------- 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_questions",default=8)
    parser.add_argument("--block_n_gram_questions",default=None)
    parser.add_argument("--col_name",default='article_text')
    parser.add_argument("--batch_size",default=64)
    parser.add_argument("--num_sentences",default=9)
    parser.add_argument("--summary_len_metric",default='sentences')
    parser.add_argument("--bert_score_model_type",default='bert-base-uncased')
    parser.add_argument("--target_tokens",default=250)
    parser.add_argument("--block_n_gram_sum",default=None)
    parser.add_argument("--visible_device",default='0')
    parser.add_argument("--question_weights",default=None)
    parser.add_argument("--temperature",default=0.5)
    parser.add_argument("--q_per_section",default=3)
    parser.add_argument("--stride",default=4)
    parser.add_argument("--data_path")
    parser.add_argument("--query_generation_model_path")
    parser.add_argument("--similarity_model_name")
    args = parser.parse_args()
    
    # data params  
    path = args.data_path
    col_name = args.col_name
    batch_size = int(args.batch_size)
    
    # question generation params     
    num_questions = int(args.num_questions)
    block_n_gram_questions = int(args.block_n_gram_questions) if (args.block_n_gram_questions != None) else None
    temperature = float(args.temperature)
    num_q_per_section = int(args.q_per_section)
    stride = int(args.stride)
    question_model_base = args.query_generation_model_path
    
    # extractive summarisation (similarity and ranking) params
    num_sentences = int(args.num_sentences)
    summary_len_metric = args.summary_len_metric
    similarity_model_name = args.similarity_model_name
    bert_score_model_type = args.bert_score_model_type
    simcse_model_type = "princeton-nlp/sup-simcse-roberta-large"
    all_layers=False                                                
    block_n_gram_sum = int(args.block_n_gram_sum) if (args.block_n_gram_sum != None) else None
    target_tokens = int(args.target_tokens)
    batch_size = int(args.batch_size)
    question_weights = bool(args.question_weights)
    

    
    # -------- LOAD MODELS ---------------  
    
    # Define the target device. Use GPU if available.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    # define model for question generation
    model_doc2query, tokenizer_doc2query = get_question_model(question_model_base,device)

    # define similarity model
    if (similarity_model_name == 'bert_score'):
        num_layers, similarity_model, similarity_tokenizer =  bert_score.get_model_and_tokenizer(
            bert_score_model_type,
            device,
            all_layers=all_layers
        )
    
    if (similarity_model_name =='simcse'):
        similarity_model = SimCSE(simcse_model_type)
        similarity_tokenizer, num_layers = None, None 
    if (similarity_model_name=='sentence_transformers'):
        similarity_model = SentenceTransformer('allenai/scibert_scivocab_cased')
        similarity_model.to(device)
        similarity_tokenizer, num_layers = None, None 
        
    print(similarity_model_name)
        
    
    # -------- LOAD DATA ---------------  
      
    df = pd.read_csv(path)
    

    # -------- RUN EXPERIMENT --------------- 

    main(df,
        model_doc2query,
        tokenizer_doc2query,
        stride,
        num_q_per_section,
        temperature,
        num_questions,
        block_n_gram_questions,
        similarity_model,
        similarity_tokenizer,
        num_layers,
        all_layers,
        summary_len_metric,
        num_sentences,
        target_tokens,
        block_n_gram_sum,
        batch_size,
        bert_score_model_type,
        device,
        col_name,
        question_weights
        )

