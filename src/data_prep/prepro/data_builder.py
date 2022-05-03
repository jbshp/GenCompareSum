import gc
import glob
import hashlib
import json
import os
import random
import pickle
import re
import subprocess
import csv
import shutil
from os.path import join as pjoin

import torch
from multiprocess import Pool

from others.logging import logger
from others.tokenization import BertTokenizer

from others.utils import clean
from prepro.utils import _get_word_ngrams

import pandas as pd
import time
from tqdm import tqdm
import numpy as np


def recover_from_corenlp(s):
    s = re.sub(r' \'{\w}', '\'\g<1>', s)
    s = re.sub(r'\'\' {\w}', '\'\'\g<1>', s)


def clean_json(json_dict):
    #
    # how about bib? they also indicate what the paper is about in general
    #
    try:
        title = json_dict['metadata']['title']
    except KeyError:
        title = 'NA'
    text = ''
    for p in json_dict['body_text']:
        p_text = p['text']
        if p['section'] == 'Pre-publication history':
            continue
        # remove references and citations from text
        citations = [*p['cite_spans'],*p['ref_spans']]
        if len(citations):
            for citation in citations:
                cite_span_len = citation['end']-citation['start']
                cite_span_replace = ' '*cite_span_len
                p_text  = p_text[:citation['start']] + cite_span_replace  + p_text[citation['end']:]
        # do other cleaning of text 
        p_text = p_text.strip()
        p_text = re.sub('\[[\d\s,]+?\]', '', p_text) # matches references e.g. [12]
        p_text = re.sub('\(Table \d+?\)', '', p_text) # matches table references e.g. (Table 1)
        p_text = re.sub('\(Fig. \d+?\)', '', p_text) # matches fig references e.g. (Fig. 1)
        p_text = re.sub('(?<=[0-9]),(?=[0-9])', '', p_text) # matches numbers seperated by commas
        p_text = re.sub('[^\x00-\x7f]+',r'', p_text) # strips non ascii
        p_text = re.sub('[\<\>]',r' ', p_text) # strips  <> tokens which are not compatable StanfordNLPtokenizer
        p_text = re.sub('(\([0-9]+\))(?= [0-9]+)',' ',p_text) # removes numbers in brackets followed by another number are not compatable StanfordNLPtokenizer
        p_text = re.sub('\n',' ',p_text) # replaces line break with full stop
        p_text = re.sub('\r',' ',p_text) # replaces line break with full stop
        p_text = re.sub(' +',' ',p_text) # removes multipe blank spaces. 
        p_text = re.sub('(?<=[0-9])( +)(?=[0-9])', '', p_text) # matches numbers seperated by space and combines
        p_text = re.sub('(?<=\.)( +)(?=\.)', '', p_text) # matches several full stops with one or more spaces in between and removes spaces
        text += '{:s}\n'.format(p_text)

    return {'title': title, 'text': text}

def load_json(f_src,f_tgt, lower):
    source = []
    tgt = []
    for sent in json.load(open(f_src))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        if (lower):
            tokens = [t.lower() for t in tokens]
        source.append(tokens)
    for sent in json.load(open(f_tgt))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        if (lower):
            tokens = [t.lower() for t in tokens]
        tgt.append(tokens)

    source = [clean(' '.join(sent)).split() for sent in source]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]
    return source, tgt

def tokenize_allenai_datasets(args):
    root_data_dir = os.path.abspath(args.raw_path)
    tokenized_data_dir = os.path.abspath(args.save_path)
    meta_path = os.path.join(root_data_dir, 'metadata','metadata.jsonl')
    pdf_dir = os.path.join(root_data_dir, 'pdf_parses')
    txt_dir = os.path.join(root_data_dir, 'pdf_parses', 'txt_json')
    
    files_count_real = 0
    no_path_counter = 0

    # make directories for saving data if they don't already exist
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    if not os.path.exists(tokenized_data_dir):
        os.makedirs(tokenized_data_dir)

    # load in pdf text data
    txt_files = [file for file in os.listdir(pdf_dir) if ('.jsonl' in file)]
    pdf_content = []
    for pdf_file in txt_files:
        with open(os.path.join(pdf_dir,pdf_file),'r') as f:
            pdf_content.append(f.readlines())
    pdf_content = [json.loads(item) for sublist in pdf_content for item in sublist]

    # read in csv containing metadata about files
    df = pd.read_json(meta_path, lines=True, dtype='unicode')
    df = df.replace(to_replace='None', value=np.nan)
    logger.info('Number of files before removing papers without abstract: {}'.format(len(df)))
    
    # skip papers without abstract
    df = df[~pd.isnull(df.abstract)]
    logger.info('Number of files after removing papers without abstract: {}'.format(len(df)))
    
    # drop duplicates
    df['title_lower'] = df.title.str.lower()
    df= df.drop_duplicates(subset='title_lower').drop(columns='title_lower')
    len_before = len(df)
    logger.info('Number of files once articles deduplicated: \t{}'.format(len_before)) # 56341 
    
    start = time.time()
    logger.info('... Processing files into readable .txt format for tokenizer into path: {}...'.format(txt_dir))

    for i,row in tqdm(df.iterrows(),total=len(df)):
            
        # read in file if available
        pid = row['paper_id']
        try:
            content = [paper for paper in pdf_content if paper['paper_id']==row['paper_id']]
            content = content[0]
        except IndexError: 
            no_path_counter +=1
            continue

        if len(content['body_text']) > 0:
            # preprocess / clean file
            cleaned_dict = clean_json(content)
            tpath = os.path.join(txt_dir, '{}.txt'.format(pid))
            tpath_abs = os.path.join(txt_dir, '{}.abs.txt'.format(pid))
            
            # write out main text and abstract 
            with open(tpath, 'w') as fil:
                fil.write(cleaned_dict['text'])
            with open(tpath_abs, 'w') as fil:
                fil.write(row['abstract'])
            files_count_real += 1
        else:         
            no_path_counter +=1

    end = time.time()
    logger.info('Real count for files with abstract: {} ({}%)'.format(files_count_real,files_count_real / len_before * 100))
    logger.info('... Ending (1), time elapsed {}'.format(end - start))

    logger.info("Preparing to tokenize %s to %s..." % (root_data_dir, tokenized_data_dir))
    num_files_to_tokenize = 0
    # make IO list file
    logger.info("Making list of files to tokenize...")
    with open('mapping_for_corenlp.txt', 'w') as fi:
        for fname in os.listdir(txt_dir):
            fpath = os.path.join(txt_dir, fname)
            fi.write('{}\n'.format(fpath))
            num_files_to_tokenize+=1

    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
               '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
               'json', '-outputDirectory', tokenized_data_dir]

    logger.info("Tokenizing %i files in %s and saving in %s..." % (num_files_to_tokenize, txt_dir, tokenized_data_dir))
    subprocess.call(command)
    logger.info("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")
    

    # Check that the tokenized data directory contains the same number of files as the original directory
    num_orig = len(os.listdir(txt_dir))
    num_tokenized = len(os.listdir(tokenized_data_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized data directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                tokenized_data_dir, num_tokenized, root_data_dir, num_orig))
    logger.info("Successfully finished tokenizing %s to %s.\n" % (root_data_dir, tokenized_data_dir))
    shutil.rmtree(txt_dir)

def clean_abstract(text_array):
    abstract = ''
    for sentence in text_array:
        sentence = sentence.replace("<S>","")
        sentence = sentence.replace("</S>","")
        abstract += sentence
    return abstract


def clean_text(doc):
    text = ''
    for paragraph in doc:
        for sentence in paragraph:
            # do other cleaning of text 
            sentence = sentence.strip()
            sentence = re.sub('\[[\d\s\,]+?\]', '', sentence) # matches references e.g. [12]
            sentence = re.sub('\(table \d+?\)', '', sentence) # matches table references e.g. (Table 1)
            sentence = re.sub('\(fig. \d+?\)', '', sentence) # matches fig references e.g. (Fig. 1)
            sentence = re.sub('[^\x00-\x7f]+',r'', sentence) # strips non ascii
            sentence = re.sub('[\<\>]',r' ', sentence) # strips  <> tokens which are not compatable StanfordNLPtokenizer
            sentence = re.sub('(\([0-9]+\))(?= [0-9]+)',' ',sentence) # removes numbers in brackets followed by another number are not compatable StanfordNLPtokenizer
            sentence = re.sub('\n',' ',sentence) # replaces line break with full stop
            sentence = re.sub('\r',' ',sentence) # replaces line break with full stop
            sentence = re.sub(' +',' ',sentence) # removes multipe blank spaces. 
            sentence = re.sub('(?<=[0-9])( +)(?=[0-9])', '', sentence) # matches numbers seperated by space and combines
            sentence = re.sub('(?<=\.)( +)(?=\.)', '', sentence) # matches several full stops with one or more spaces in between and removes spaces
            text += '{:s}\n'.format(sentence)

    return text



def tokenize_pubmed_dataset(args):
    
    root_data_dir = os.path.abspath(args.raw_path)
    
    dirs = ['train','val','test']

    for idx, dir in enumerate(dirs):
        files_count_real = 0
        tokenized_data_dir = os.path.join(os.path.abspath(args.save_path),dir)
        source_txt_file = os.path.join(root_data_dir, '{}.txt'.format(dir))
        txt_dir = os.path.join(root_data_dir, 'txt_json', dir)

        # make directories for saving data if they don't already exist
        if not os.path.exists(txt_dir):
            os.makedirs(txt_dir)
        if not os.path.exists(tokenized_data_dir):
            os.makedirs(tokenized_data_dir)
        
        logger.info('... Loading PMC data from {}'.format(root_data_dir))

        # read in txt file with raw data
        try:
            with open(source_txt_file,'r') as f:
                docs = f.readlines()
            docs = [json.loads(doc) for doc in docs]
            df = pd.DataFrame(docs)
            len_before = len(df)
        except FileNotFoundError:
            logger.error(f"File not found: {source_txt_file}")
            continue
        
        start = time.time()
        logger.info('...Processing pubmed files into readable .txt format for tokenizer into path: {}...'.format(txt_dir))
        
        # write out new csv containing files we use in our dataset
        pid = 0
        labels = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
                
            # read in pubmed file if available
        
            # preprocess / clean file
            try:
               label =  row['label']
            except KeyError:
                label = []
            cleaned_text = clean_abstract(row['article_text'])
            tpath = os.path.join(txt_dir, '{}.txt'.format(pid))
            tpath_abs = os.path.join(txt_dir, '{}.abs.txt'.format(pid))
            # preprocess/ clean abstract
            abstract = clean_abstract(row['abstract_text'])
            labels.append(label)
        
            # write out main text and abstract 
            with open(tpath, 'w') as fil:
                fil.write(cleaned_text)
            with open(tpath_abs, 'w') as fil:
                fil.write(abstract)
            files_count_real += 1
            pid += 1

        pic_path = os.path.join(args.save_path, '{}.pkl'.format(dirs[idx]))
        with open(pic_path, 'wb') as f:
            pickle.dump(labels, f)

        end = time.time()
    
        logger.info('Real count for files with abstract: {} ({}%)'.format(files_count_real,files_count_real / len_before * 100))
        logger.info('... Ending (1), time elapsed {}'.format(end - start))

        logger.info("Preparing to tokenize %s to %s..." % (root_data_dir, tokenized_data_dir))
        num_files_to_tokenize = 0
        # make IO list file
        logger.info("Making list of files to tokenize...")
        with open('mapping_for_corenlp.txt', 'w') as fi:
            for fname in os.listdir(txt_dir):
                fpath = os.path.join(txt_dir, fname)
                fi.write('{}\n'.format(fpath))
                num_files_to_tokenize+=1

        command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
                '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
                'json', '-outputDirectory', tokenized_data_dir]

        logger.info("Tokenizing %i files in %s and saving in %s..." % (num_files_to_tokenize, txt_dir, tokenized_data_dir))
        subprocess.call(command)
        logger.info("Stanford CoreNLP Tokenizer has finished.")
        os.remove("mapping_for_corenlp.txt")
        
    shutil.rmtree(os.path.join(root_data_dir, 'txt_json'))

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}

def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self, src, tgt, sent_labels, use_bert_basic_tokenizer=False, is_test=False):

        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            _sent_labels[l] = 1

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        sent_labels = [_sent_labels[i] for i in idxs]
        src = src[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]

        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        #segments_ids = [0]*len(src_subtoken_idxs)
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]

        tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt
             in tgt]) + ' [unused1]'
        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt


def format_to_bert(args):
    logger.info('...Converting data to BERT data... this will take a while')

    a_lst = []
    for json_f in glob.glob(pjoin(args.raw_path, '*.json')):

        if 'test' in json_f:
            corpus_type = 'test'
        else:
            corpus_type = None
        real_name = json_f.split('/')[-1]

        a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))

    pool = Pool(args.n_cpus)
    for d in pool.imap(_format_to_bert, a_lst):
        pass
    pool.close()
    pool.join()

    for json_f in glob.glob(pjoin(args.raw_path, '*.json')):
        os.remove(json_f)

def _format_to_bert(params):
    corpus_type, json_file, args, save_file= params
    is_test = corpus_type == 'test'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    for d in jobs:
        source, tgt = d['src'], d['tgt']
        sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 3)
        if (args.lower):
            source = [' '.join(s).lower().split() for s in source]
            tgt = [' '.join(s).lower().split() for s in tgt]
        b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                 is_test=is_test)
        # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)

        if (b_data is None):
            continue
        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                       "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt}
        datasets.append(b_data_dict)
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    gc.collect()



def format_to_lines(args):
    root_data_dir = os.path.abspath(args.raw_path)

    train_files, valid_files, test_files = [], [], []
    corpera = ['train', 'valid', 'test']
    for d_split in corpera:
        try:
            files = []
            txt_path = os.path.join(root_data_dir, d_split)
            corpora = sorted([os.path.join(txt_path, f) for f in os.listdir(txt_path)
                                if not f.startswith('.') and not f.endswith('.abs.txt.json') ])
            with open(os.path.join(root_data_dir, f'{d_split}.pkl'), 'rb') as f:
                label = pickle.load(f)
                logger.info(label)
            for f_main in corpora:
                f_abs_name = '{}.abs.txt.json'.format(os.path.basename(f_main).split('.')[0])
                f_abs = os.path.join(txt_path, f_abs_name)
                files.append((f_main, f_abs, args))
            if d_split == 'train':
                train_files = files
            elif d_split == 'test':
                test_files = files
            elif d_split =='val':
                valid_files = files
        except FileNotFoundError as e:
            logger.error(e)
            continue
   
    start = time.time()
    logger.info('... Packing tokenized data into shards...')

    # imap executes in sync multiprocess manner
    # use array and shard_size to save the flow of ordered data
    corporas = {'train': train_files, 'valid': valid_files, 'test': test_files}
    for corpus_type in corpera:
        a_lst = corporas[corpus_type]
        pool = Pool(args.n_cpus)
        dataset = []
        shard_count = 0
        with tqdm(total=len(a_lst)) as pbar:
            with tqdm(total=args.shard_size) as spbar:
                for i, data in enumerate(pool.imap(_format_to_lines, a_lst)):
                    if data:
                        dataset.append(data)
                    spbar.update()
                    if (len(dataset) > args.shard_size):
                        fpath = "{:s}/{:s}.{:d}.json".format(args.save_path, corpus_type, shard_count)
                        with open(fpath, 'w') as f:
                            f.write(json.dumps(dataset))
                        dataset = []
                        shard_count += 1
                        pbar.update()
                        spbar.reset()
                        # gc.collect()
                spbar.close()
            pbar.close()
        pool.close()
        pool.join()
        if len(dataset) > 0:
            fpath = "{:s}/{:s}.{:d}.json".format(args.save_path, corpus_type, shard_count)
            logger.info('last shard {} saved'.format(shard_count))
            with open(fpath, 'w') as f:
                f.write(json.dumps(dataset))
            dataset = []
            shard_count += 1
    end = time.time()
    logger.info('... Ending (4), time elapsed {}'.format(end - start))

def format_to_lines_no_split(args):
    root_data_dir = os.path.abspath(args.raw_path)
    files = []
    corpora = sorted([os.path.join(root_data_dir, f) for f in os.listdir(root_data_dir)
                        if not f.startswith('.') and not f.endswith('.abs.txt.json') ])
    for f_main in corpora:
        f_abs_name = '{}.abs.txt.json'.format(os.path.basename(f_main).split('.')[0])
        f_abs = os.path.join(root_data_dir, f_abs_name)
        files.append((f_main, f_abs, args))
   
    start = time.time()
    logger.info('... Packing tokenized data into shards...')

    # imap executes in sync multiprocess manner
    # use array and shard_size to save the flow of ordered data
    pool = Pool(args.n_cpus)
    dataset = []
    shard_count = 0
    with tqdm(total=len(files)) as pbar:
        with tqdm(total=args.shard_size) as spbar:
            for i, data in enumerate(pool.imap(_format_to_lines, files)):
                if data:
                    dataset.append(data)
                spbar.update()
                if (len(dataset) > args.shard_size):
                    fpath = os.path.join(args.save_path,"{:d}.json".format(shard_count))
                    with open(fpath, 'w') as f:
                        f.write(json.dumps(dataset))
                    dataset = []
                    shard_count += 1
                    pbar.update()
                    spbar.reset()
                    # gc.collect()
            spbar.close()
        pbar.close()
    pool.close()
    pool.join()
    if len(dataset) > 0:
        fpath = os.path.join(args.save_path,"{:d}.json".format(shard_count))
        logger.info('last shard {} saved'.format(shard_count))
        with open(fpath, 'w') as f:
            f.write(json.dumps(dataset))
        dataset = []
        shard_count += 1
    end = time.time()
    logger.info('... Ending (4), time elapsed {}'.format(end - start))

def _format_to_lines(params):
    f_src, f_tgt, args = params
    source, tgt = load_json(f_src,f_tgt, args.lower)
    return {'src': source, 'tgt': tgt}


def json_to_csv(args):

    corpera = ['train','val','test']

    try:
        input_path = args.raw_path
        files = os.listdir(input_path)
        files = [file for file in files if ('.json' in file)]

        new_articles = []
        for file in files:
            file_start = file.split('.')[0]
            logger.info(f'starting file {file}')
            json_file = f"{input_path}/{file}"
            with open(json_file,'r') as f:
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

                # split dataframe into train/test/val splits
                if file_start in corpera:
                    data_split = file_start
                else:
                    data_split = random.choices(corpera,weights=[7.5,1.5,1],k=1)[0]

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
            os.remove(json_file)
    except FileNotFoundError as e:
        logger.error(e)

    df = pd.DataFrame(new_articles)
    logger.info(f"Mean article tokens: {df['article_tokens'].mean()}")
    logger.info(f"Mean summary tokens: {df['summary_tokens'].mean()}")
    logger.info(f"Mean article sentences: {df['article_sentences'].mean()}")
    logger.info(f"Mean summary sentences: {df['summary_sentences'].mean()}")
    logger.info(f"Mean short article tokens: {df['short_article_tokens'].mean()}")
    logger.info(f"Mean short article sentences: {df['short_article_sentences'].mean()}")

    df = df[['article_text','short_article','summary_text_combined','data_split']]

    for d_split in corpera:
        df_split = df[df['data_split']==d_split].drop(columns=['data_split']).reset_index(drop=True)
        logger.info(f'Number of articles in {d_split} set: {len(df_split)}')
        df_split.to_csv(os.path.join(args.save_path,f'{d_split}.csv'),index=False)

def preprocess_pubmed_for_GenCompareSum(args):
    tmp_dir = './tmp/'
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    save_path = args.save_path
    args.save_path = tmp_dir
    tokenize_pubmed_dataset(args)
    args.raw_path = args.save_path
    args.save_path = save_path
    format_to_lines(args)
    shutil.rmtree(tmp_dir)
    args.raw_path = save_path
    json_to_csv(args)

def preprocess_pubmed_for_BERTSum(args):
    tmp_dir = './tmp/'
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    save_path = args.save_path
    args.save_path = tmp_dir
    tokenize_pubmed_dataset(args)
    args.raw_path = args.save_path
    args.save_path = save_path
    format_to_lines(args)
    shutil.rmtree(tmp_dir)
    args.raw_path = save_path
    format_to_bert(args)

def preprocess_allenai_datasets_for_GenCompareSum(args):
    tmp_dir = './tmp/'
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    save_path = args.save_path
    args.save_path = tmp_dir
    tokenize_allenai_datasets(args)
    args.raw_path = args.save_path
    args.save_path = save_path
    format_to_lines_no_split(args)
    shutil.rmtree(tmp_dir)
    args.raw_path = save_path
    json_to_csv(args)

def preprocess_allenai_datasets_for_BERTSum(args):
    tmp_dir = './tmp/'
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    save_path = args.save_path
    args.save_path = tmp_dir
    tokenize_allenai_datasets(args)
    args.raw_path = args.save_path
    args.save_path = save_path
    format_to_lines_no_split(args)
    shutil.rmtree(tmp_dir)
    args.raw_path = save_path
    format_to_bert(args)

