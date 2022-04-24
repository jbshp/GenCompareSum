import gc
import glob
import hashlib
import itertools
import string
# import zhon
import json
import os
import random
import pickle
import re
import math
import subprocess
import csv
import shutil
from collections import Counter
from os.path import join as pjoin
# from zhon.hanzi import punctuation
import torch
from multiprocess import Pool

from others.logging import logger
from others.tokenization import BertTokenizer
from transformers import RobertaTokenizer, AutoTokenizer
from pytorch_transformers import XLNetTokenizer

from others.utils import clean
from prepro.utils import _get_word_ngrams

import xml.etree.ElementTree as ET
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm

nyt_remove_words = ["photo", "graph", "chart", "map", "table", "drawing"]


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

def load_json(f_main, f_abs, f_tag):
    with open(f_main, 'r') as f:
        json_main = json.load(f)
    with open(f_abs, 'r') as f:
        json_abs = json.load(f)

    src_sent_tokens = [
        list(t['word'].lower() for t in sent['tokens'])
        for sent in json_main['sentences']]
    if not src_sent_tokens:
        return None, None, None
    else:
        tgt_sent_tokens = [
        list(t['word'].lower() for t in sent['tokens'])
        for sent in json_abs['sentences']]

        with open(f_tag, 'r') as f:
            json_tag = json.load(f)
        tag_tokens = []
        tag_tags = []
        sent_lengths = [len(val) for val in src_sent_tokens]
        count = 0
        offset = 0
        temp_doc_len = len(json_tag)
        while offset < temp_doc_len:
            present_sent_len = sent_lengths[count]
            sent_tokens = json_tag[offset:offset + present_sent_len]
            try:
                assert [val.lower() for _, val in sent_tokens] == src_sent_tokens[count]
            except AssertionError as e:
                print('src:', src_sent_tokens[count])
                print('tag:', [val.lower() for _, val in sent_tokens])
        #assert [val.lower() for _, val in sent_tokens] == src_sent_tokens[count]
            offset += present_sent_len
            try:
                assert offset <= temp_doc_len
            except AssertionError as e:
                print('not match')
                return None, None, None
        #tag_tokens.append([val.lower() for _, val in sent_tokens])
            temp=[]
            for val, t in sent_tokens:
                if ' ' in t:
                    s = t.split()
                    align_tag = len(s)*[val]
                    temp += align_tag
                else:
                    temp.append(val)
            tag_tags.append(temp)
            count += 1
    #assert tag_tokens == src_sent_tokens

        tags = tag_tags
        src = [clean(' '.join(tokens)).split() for tokens in src_sent_tokens]
        for i, val in enumerate(src):
            assert len(val) == len(tags[i])

        tgt = [clean(' '.join(tokens)).split() for tokens in tgt_sent_tokens]
        return src, tgt, tags

def load_xml(p):
    tree = ET.parse(p)
    root = tree.getroot()
    title, byline, abs, paras = [], [], [], []
    title_node = list(root.iter('hedline'))
    if (len(title_node) > 0):
        try:
            title = [p.text.lower().split() for p in list(title_node[0].iter('hl1'))][0]
        except:
            print(p)

    else:
        return None, None
    byline_node = list(root.iter('byline'))
    byline_node = [n for n in byline_node if n.attrib['class'] == 'normalized_byline']
    if (len(byline_node) > 0):
        byline = byline_node[0].text.lower().split()
    abs_node = list(root.iter('abstract'))
    if (len(abs_node) > 0):
        try:
            abs = [p.text.lower().split() for p in list(abs_node[0].iter('p'))][0]
        except:
            print(p)

    else:
        return None, None
    abs = ' '.join(abs).split(';')
    abs[-1] = abs[-1].replace('(m)', '')
    abs[-1] = abs[-1].replace('(s)', '')

    for ww in nyt_remove_words:
        abs[-1] = abs[-1].replace('(' + ww + ')', '')
    abs = [p.split() for p in abs]
    abs = [p for p in abs if len(p) > 2]

    for doc_node in root.iter('block'):
        att = doc_node.get('class')
        # if(att == 'abstract'):
        #     abs = [p.text for p in list(f.iter('p'))]
        if (att == 'full_text'):
            paras = [p.text.lower().split() for p in list(doc_node.iter('p'))]
            break
    if (len(paras) > 0):
        if (len(byline) > 0):
            paras = [title + ['[unused3]'] + byline + ['[unused4]']] + paras
        else:
            paras = [title + ['[unused3]']] + paras

        return paras, abs
    else:
        return None, None

def tokenize_allenai_datasets(args):
    root_data_dir = os.path.abspath(args.raw_path)
    tokenized_data_dir = os.path.abspath(args.save_path)
    meta_path = os.path.join(root_data_dir, 'metadata.jsonl')
    new_meta_path = os.path.join(root_data_dir, 'PMC.csv')
    pmc_dir = os.path.join(root_data_dir, 'document_parses', 'pmc_json')
    txt_dir = os.path.join(root_data_dir, 'document_parses', 'txt_json')
    
    files_count_real = 0
    no_path_counter = 0

    # make directories for saving data if they don't already exist
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    if not os.path.exists(tokenized_data_dir):
        os.makedirs(tokenized_data_dir)
    
    print('... Loading PMC data from {}'.format(pmc_dir))

    # read in csv containing metadata about files
    df = pd.read_csv(meta_path, sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
    print('Number of files before removing papers without abstract: {}'.format(df.shape[0]))
    
    # skip papers without abstract
    df = df[~pd.isnull(df.abstract)]
    print('Number of files after removing papers without abstract: {}'.format(df.shape[0]))
    
    # drop duplicates
    df['title_lower'] = df.title.str.lower()
    df= df.drop_duplicates(subset='title_lower').drop(columns='title_lower')
    len_before = df.shape[0]
    print('Number of files once articles deduplicated: \t{}'.format(len_before)) # 56341 
    
    start = time.time()
    print('... (1) Processing files into readable .txt format for tokenizer into path: {}...'.format(txt_dir))
    
    write_head = False

    # write out new csv containing files we use in our dataset
    with open(new_meta_path, 'w') as f:
        w = csv.writer(f)
        for i,row in tqdm(df.iterrows(),total=df.shape[0]):
                
            # read in file if available
            pid = row['pmcid']
            try:
                pubtime = row['publish_time']
            except KeyError: 
                pubtime = row['year']
            # pubtime = datetime.strptime(row['publish_time'], '%Y-%m-%d').timestamp()
            ppath = os.path.join(pmc_dir, '{}.xml.json'.format(pid))
            if not os.path.isfile(ppath):
                no_path_counter +=1
                continue
            with open(ppath, 'r') as fi:
                json_dict = json.load(fi)
            
            # preprocess / clean file
            cleaned_dict = clean_json(json_dict)
            tpath = os.path.join(txt_dir, '{}-{}.txt'.format(pubtime, pid))
            tpath_abs = os.path.join(txt_dir, '{}-{}.abs.txt'.format(pubtime, pid))
            
            # write out main text and abstract 
            with open(tpath, 'w') as fil:
                fil.write(cleaned_dict['text'])
            with open(tpath_abs, 'w') as fil:
                fil.write(row['abstract'])
            files_count_real += 1
            
            # write csv row
            cleaned_dict['abstract'] = row['abstract']
            if cleaned_dict['title'] == 'NA':
                cleaned_dict['title'] = row['title']
            if not write_head:
                w.writerow(cleaned_dict.keys())
                write_head = True       
            w.writerow(cleaned_dict.values())

    end = time.time()
    print('Real count for files with abstract: {} ({}%)'.format(files_count_real,files_count_real / len_before * 100))
    print('... Ending (1), time elapsed {}'.format(end - start))

    print("Preparing to tokenize %s to %s..." % (root_data_dir, tokenized_data_dir))
    num_files_to_tokenize = 0
    # make IO list file
    print("Making list of files to tokenize...")
    with open('mapping_for_corenlp.txt', 'w') as fi:
        for fname in os.listdir(txt_dir):
            fpath = os.path.join(txt_dir, fname)
            fi.write('{}\n'.format(fpath))
            num_files_to_tokenize+=1

    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
               '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
               'json', '-outputDirectory', tokenized_data_dir]

    print("Tokenizing %i files in %s and saving in %s..." % (num_files_to_tokenize, txt_dir, tokenized_data_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")
    

    # Check that the tokenized data directory contains the same number of files as the original directory
    num_orig = len(os.listdir(txt_dir))
    num_tokenized = len(os.listdir(tokenized_data_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized data directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                tokenized_data_dir, num_tokenized, root_data_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (root_data_dir, tokenized_data_dir))
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
    
    dirs = ['sample']

    for idx, dir in enumerate(dirs):
        files_count_real = 0
        tokenized_data_dir = os.path.join(os.path.abspath(args.save_path),dir)
        source_txt_file = os.path.join(root_data_dir, '{}.jsonl'.format(dir))
        txt_dir = os.path.join(root_data_dir, 'txt_json', dir)

        # make directories for saving data if they don't already exist
        if not os.path.exists(txt_dir):
            os.makedirs(txt_dir)
        if not os.path.exists(tokenized_data_dir):
            os.makedirs(tokenized_data_dir)
        
        print('... Loading PMC data from {}'.format(root_data_dir))

        # read in txt file with raw data
        df = pd.read_json(path_or_buf=source_txt_file, lines=True)
        len_before = df.shape[0]
        
        start = time.time()
        print('... (1) Processing pubmed files into readable .txt format for tokenizer into path: {}...'.format(txt_dir))
        
        # write out new csv containing files we use in our dataset
        pid = 0
        labels = []
        for i,row in tqdm(df.iterrows(),total=df.shape[0]):
                
            # read in pubmed file if available
        
            # preprocess / clean file
            try:
               label =  row['label']
            except KeyError:
                label = []
            cleaned_text = clean_abstract(row['text'])
            tpath = os.path.join(txt_dir, '{}.txt'.format(pid))
            tpath_abs = os.path.join(txt_dir, '{}.abs.txt'.format(pid))
            # preprocess/ clean abstract
            abstract = clean_abstract(row['summary'])
            #print(row)
            labels.append(label)
            #print('text:', cleaned_text)
            #print('summary:', abstract)
        
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
    
        print('Real count for files with abstract: {} ({}%)'.format(files_count_real,files_count_real / len_before * 100))
        print('... Ending (1), time elapsed {}'.format(end - start))

        print("Preparing to tokenize %s to %s..." % (root_data_dir, tokenized_data_dir))
        num_files_to_tokenize = 0
        # make IO list file
        print("Making list of files to tokenize...")
        with open('mapping_for_corenlp.txt', 'w') as fi:
            for fname in os.listdir(txt_dir):
                fpath = os.path.join(txt_dir, fname)
                fi.write('{}\n'.format(fpath))
                num_files_to_tokenize+=1

        command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
                '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
                'json', '-outputDirectory', tokenized_data_dir]

        print("Tokenizing %i files in %s and saving in %s..." % (num_files_to_tokenize, txt_dir, tokenized_data_dir))
        subprocess.call(command)
        print("Stanford CoreNLP Tokenizer has finished.")
        os.remove("mapping_for_corenlp.txt")
        

        # Check that the tokenized data directory contains the same number of files as the original directory
        num_orig = len(os.listdir(txt_dir))
        num_tokenized = len(os.listdir(tokenized_data_dir))
        #if num_orig != num_tokenized:
        #    raise Exception(
        #        "The tokenized data directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
        #            tokenized_data_dir, num_tokenized, root_data_dir, num_orig))
        #print("Successfully finished tokenizing %s to %s.\n" % (root_data_dir, tokenized_data_dir))
        shutil.rmtree(txt_dir)

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


class RoBertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        #BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.sep_token = '<s>'
        self.cls_token = '</s>'
        self.pad_token = '<pad>'
        self.tgt_bos = '<s>'
        self.tgt_eos = '</s>'
        self.tgt_sent_split = '<s>'
        self.sep_vid = self.tokenizer.convert_tokens_to_ids(self.sep_token)
        self.cls_vid = self.tokenizer.convert_tokens_to_ids(self.cls_token)
        self.pad_vid = self.tokenizer.convert_tokens_to_ids(self.pad_token)
        self.tgt_bos_vid = self.tokenizer.convert_tokens_to_ids(self.tgt_bos)
        self.tgt_eos_vid = self.tokenizer.convert_tokens_to_ids(self.tgt_eos)
        self.tgt_sent_split_vid = self.tokenizer.convert_tokens_to_ids(self.tgt_sent_split)

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
        #_segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        #segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        #segments_ids = []
        #for i, s in enumerate(segs):
        #    if (i % 2 == 0):
        #        segments_ids += s * [0]
        #    else:
        #        segments_ids += s * [1]
        segments_ids = [0]*len(src_subtoken_idxs)
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]

        tgt_subtokens_str = '<s>' + ' </s>'.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt))) for tt in tgt]) + ' </s>'
        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt

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

class PubmedData():
    def __init__(self, args):
        self.args = args
        model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
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
        #print(src_subtokens)
        #print(len(segments_ids),len(src_subtoken_idxs))
        assert len(segments_ids)==len(src_subtoken_idxs)
        #segments_ids = [0]*len(src_subtoken_idxs)
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]

        tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt))) for tt
             in tgt]) + ' [unused1]'
        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt

class BioBertData():
    def __init__(self, args):
        self.args = args
        model_name = 'dmis-lab/biobert-v1.1'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
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
        assert len(segments_ids)==len(src_subtoken_idxs)
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]

        tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt))) for tt
             in tgt]) + ' [unused1]'
        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt

class PicoAdapterData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        #BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.sep_token = '<s>'
        self.cls_token = '</s>'
        self.pad_token = '<pad>'
        self.tgt_bos = '<s>'
        self.tgt_eos = '</s>'
        self.tgt_sent_split = '<s>'
        self.mask_token = '<mask>'
        self.sep_vid = self.tokenizer.convert_tokens_to_ids(self.sep_token)
        self.cls_vid = self.tokenizer.convert_tokens_to_ids(self.cls_token)
        self.pad_vid = self.tokenizer.convert_tokens_to_ids(self.pad_token)
        self.tgt_bos_vid = self.tokenizer.convert_tokens_to_ids(self.tgt_bos)
        self.tgt_eos_vid = self.tokenizer.convert_tokens_to_ids(self.tgt_eos)
        self.tgt_sent_split_vid = self.tokenizer.convert_tokens_to_ids(self.tgt_sent_split)
        self.mask_vid = self.tokenizer.convert_tokens_to_ids(self.mask_token)

    def preprocess(self, src, tag, is_test=False):

        idxs = []
        new_src = []
        new_tag = []
        for i, d in enumerate(src):
            new_src.append([])
            new_tag.append([])
            for j, s in enumerate(d):
                #assert len(s)==len(tag[i][j]),(len(s),len(tag[i][j]))
                if len(s) > self.args.min_src_ntokens_per_sent:
                    new_src[-1].append(s[:self.args.max_src_ntokens_per_sent])
                    new_tag[-1].append(tag[i][j][:self.args.max_src_ntokens_per_sent])
        #print(len(new_tag))
    
        src_filt = [d[:self.args.max_src_nsents][:] for d in new_src if self.args.min_src_nsents < len(d)]
        tag_filt = [d_tag[:self.args.max_src_nsents][:] for d_tag in new_tag if self.args.min_src_nsents < len(d_tag)]
        print(len(src_filt), len(src_filt[0]))

        src_txt = []
        # trans = str.maketrans("", "", string.punctuation + "‘" + "’" + "‐" + '‑' + '”')
        # chin_trans = str.maketrans("", "", punctuation)
        # zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
        # anotrans = str.maketrans("", "", "’")
        for d in src_filt:
            temp = []
            for sent in d:
                temp_s = sent
                for index, each_token in enumerate(sent):
                    if each_token.startswith("http"):
                        temp_s[index] = "http"
                    # elif len(each_token) > 1:
                    #     temp_s[index] = each_token.translate(trans)
                    #     temp_s[index] = temp_s[index].translate(chin_trans)
                    if temp_s[index] == "":
                        temp_s[index] = "."
                    if each_token.startswith("www"):
                        temp_s[index] = "www"
                    if zhPattern.search(each_token):
                        temp_s[index] = "<unk>"
                        # temp_s[index] = temp_s[index].translate(anotrans)
                temp.append(' '.join(temp_s))
            src_txt.append(temp)
        #src_txt = [' '.join(sent) for d in src for sent in d]
        #print(len(src_txt))

        text = [' {} {} '.format(self.sep_token, self.cls_token).join(d) for d in src_txt]
        #print(len(text))
        #text = ['{} '.format(self.cls_token) + d + ' {}'.format(self.sep_token) for d in text]
        #print(len(text))
        tags = []
        for i, t in enumerate(tag_filt):
            temp = []
            temp.append('O')
            for tag_list in t:
                for tag in tag_list:
                    temp.append(tag)
                temp.append('O')
                temp.append('O')
            temp = temp[:-2]
            temp.append('O')
            tags.append(temp)
            #assert len(text[i].split())==len(temp), (i, len(text[i].split()),len(temp))

        src_encoding = self.tokenizer(text,truncation=True,padding=True)

        src_subtoken_idxs = src_encoding['input_ids']
        src_subtoken_origin_idxs = src_subtoken_idxs
        #print(len(src_subtoken_idxs), len(src_subtoken_idxs[0]))
        src_subtokens = [self.tokenizer.convert_ids_to_tokens(idx) for idx in src_subtoken_idxs]
        #print(text[3].split()[:10])
        #print(src_subtokens[3][:50])
        tag_align = []
        for i, subtoken in enumerate(src_subtokens):
            print(len(subtoken))
            aligned_labels = ["O"] * len(subtoken)
            head = 0
            count = 0
            subtoken[1] = "Ġ" + subtoken[1] 
            temp_src = text[i].split("</s>")
            temp = " ".join([val for val in subtoken if val != '<pad>'])
            temp = temp.split("</s>")
            #print("last temp:", temp[-1])
            for j, temp_sent in enumerate(temp):
                if j < len(temp)-1:
                    temp_s = temp_sent.split()
                    temp_sent_filt = [filt for filt in temp_s if filt.startswith("Ġ")]
                    if len(temp_sent_filt)-1!=len(temp_src[j].split()):
                        print(i,j,len(temp_sent_filt)-1,len(temp_src[j].split()), temp_s,temp[j+1], temp_src[j].split())
            #assert len(tags[i])==len([val for val in subtoken if val.startswith("Ġ") and val !='<pad>']),(i, len(tags[i]),len([val for val in subtoken if val.startswith("Ġ") and val !='<pad>']))
            for j, each_str in enumerate(subtoken):
                #print(i, each_str, head, count)
                if each_str != "<pad>":
                    if j==0 or "Ġ" in each_str:
                        aligned_labels[head] = tags[i][count]
                        count += 1
                        head += 1
                    else:
                        aligned_labels[head] = tags[i][count-1]
                        head += 1
                else:
                    break
            tag_align.append(aligned_labels)
        #print(head, len(src_subtokens), count, len(tags))
        #src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        tag_dict = {'O':0, "I-INT":1, "I-PAR": 2, "I-OUT": 3}
        tag_id = []
        for tag_list in tag_align:
            temp = []
            for tag in tag_list:
                temp.append(int(tag_dict[tag]))
            tag_id.append(temp)
        mask_label = []
        for i, tag_list in enumerate(tag_align):
            temp = []
            for j,tag in enumerate(tag_list):
                if tag == "O":
                    temp.append(0.0)
                else:
                    src_subtoken_idxs[i][j] = self.mask_vid
                    temp.append(1.0)
            #print(len(temp))
            mask_label.append(temp)
        data = []
        for i in range(len(mask_label)):
            data.append({"src": src_subtoken_idxs[i], "tag": tag_id[i], "mask": mask_label[i], "src_orig":src_subtoken_origin_idxs[i]})
        return data


class PicoBertAdapterData():
    def __init__(self, args):
        self.args = args
        from transformers import BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",do_lower_case=True)

        # BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        #self.unused_token = '[unused0]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]
        self.mask_token = '[MASK]'
        self.mask_vid = self.tokenizer.convert_tokens_to_ids(self.mask_token)

    def preprocess(self, src, tag, is_test=False):

        idxs = []
        new_src = []
        new_tag = []
        for i, d in enumerate(src):
            new_src.append([])
            new_tag.append([])
            for j, s in enumerate(d):
                # assert len(s)==len(tag[i][j]),(len(s),len(tag[i][j]))
                if len(s) > self.args.min_src_ntokens_per_sent:
                    new_src[-1].append(s[:self.args.max_src_ntokens_per_sent])
                    new_tag[-1].append(tag[i][j][:self.args.max_src_ntokens_per_sent])
        # print(len(new_tag))

        src_filt = [d[:self.args.max_src_nsents][:] for d in new_src if self.args.min_src_nsents < len(d)]
        tag_filt = [d_tag[:self.args.max_src_nsents][:] for d_tag in new_tag if self.args.min_src_nsents < len(d_tag)]
        #print (tag_filt[0], len(tag_filt))
        print(len(src_filt), len(src_filt[0]), src_filt[0][0])
        src_txt = []
        # trans = str.maketrans("", "", string.punctuation+"‘"+"’"+"‐"+'‑'+'”')
        # chin_trans = str.maketrans("", "", punctuation)
        # zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
        #anotrans = str.maketrans("", "", "’")
        for d in src_filt:
            temp = []
            for sent in d:
                temp_s = sent
                for index, each_token in enumerate(sent):
                    if each_token.startswith("http"):
                        temp_s[index] = "http"
                    # elif len(each_token) > 1:
                    #     temp_s[index] = each_token.translate(trans)
                    #     temp_s[index] = temp_s[index].translate(chin_trans)
                    if temp_s[index] == "":
                        temp_s[index] = "[UNK]"
                    if each_token.startswith("www"):
                        temp_s[index] = "www"
                    if zhPattern.search(each_token):
                        temp_s[index] = "[UNK]"
                        #temp_s[index] = temp_s[index].translate(anotrans)
                temp.append(' '.join(temp_s))
            src_txt.append(temp)

        # src_txt = [' '.join(sent) for d in src for sent in d]
        # print(len(src_txt))

        text = [' {} {} '.format(self.sep_token, self.cls_token).join(d) for d in src_txt]
        # print(len(text))
        #text = ['{} '.format(self.cls_token) + d + ' {}'.format(self.sep_token) for d in text]
        # print(len(text))
        tags = []
        for i, t in enumerate(tag_filt):
            temp = []
            temp.append('O')
            for tag_list in t:
                for tag in tag_list:
                    temp.append(tag)
                temp.append('O')
                temp.append('O')
            temp = temp[:-2]
            temp.append('O')
            tags.append(temp)
            #assert len(text[i].split()) == len(temp), (i, text[i].split(), len(text[i].split()), len(temp))

        #print("equal:", len(tags[0]), len(text[0].split()))
        src_encoding = self.tokenizer(text, truncation=True, padding=True)
        src_subtoken_idxs = src_encoding['input_ids']
        src_subtoken_origin_idxs = src_subtoken_idxs
        src_token_type_id = src_encoding['token_type_ids']
        #print(len(src_subtoken_idxs), len(src_subtoken_idxs[0]))
        #print (len(tags[0]), len(src_subtoken_idxs[0]), len(src_token_type_id[0]))
        src_subtokens = [self.tokenizer.convert_ids_to_tokens(idx) for idx in src_subtoken_idxs]
        tag_align = []
        #print(src_subtokens[0])
        #print("debug one:", len(src_subtokens), len(tags))
        for i, subtoken in enumerate(src_subtokens):
            aligned_labels = ["O"] * len(subtoken)
            count = 0
            # print(len(subtoken))
            temp_src = text[i].split("[CLS]")
            temp = " ".join([val for val in subtoken if val != '[PAD]'])
            temp = temp.split("[CLS]")
            for j, temp_sent in enumerate(temp):
                if 0 < j:
                    temp_s = temp_sent.split()
                    temp_sent_filt = [filt for filt in temp_s if not filt.startswith("##")]
                    if len(temp_sent_filt)!=len(temp_src[j-1].split()):
                        print(len(temp_sent_filt),len(temp_src[j-1].split()))
                        print("sub:", temp_sent_filt)
                        print("src:", temp_src[j-1].split())
                        print("\n")
            print(i, len(tags[i]),len([val for val in subtoken if not val.startswith("##") and val !='[PAD]']))
            for j, each_str in enumerate(subtoken):
                # print(i, each_str, head, count)
                if each_str != "[PAD]":
                    if each_str.startswith("##"):
                        aligned_labels[j] = tags[i][count - 1]
                    else:
                        #print(each_str, len(tags[i]), count)
                        #try:
                        aligned_labels[j] = tags[i][count]
                        #except Exception:
                        #    pass
                        count += 1
                else:
                    break
            tag_align.append(aligned_labels)
        # print(head, len(src_subtokens), count, len(tags))
        # src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        tag_dict = {'O': 0, "I-INT": 1, "I-PAR": 2, "I-OUT": 3}
        tag_id = []
        for tag_list in tag_align:
            temp = []
            for tag in tag_list:
                temp.append(int(tag_dict[tag]))
            tag_id.append(temp)
        mask_label = []
        for i, tag_list in enumerate(tag_align):
            temp = []
            for j, tag in enumerate(tag_list):
                if tag == "O":
                    temp.append(0.0)
                else:
                    src_subtoken_idxs[i][j] = self.mask_vid
                    temp.append(1.0)
            #print(len(temp))
            mask_label.append(temp)
        print(len(mask_label))
        data = []
        for i in range(len(mask_label)):
            data.append({"src": src_subtoken_idxs[i], "tag": tag_id[i], "mask": mask_label[i], "token_type_ids":src_token_type_id[i],
                         "src_orig":src_subtoken_origin_idxs[i]})
        return data

class PicoPubmedBertAdapterData():
    def __init__(self, args):
        self.args = args
        model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]
        self.mask_token = '[MASK]'
        self.mask_vid = self.tokenizer.convert_tokens_to_ids(self.mask_token)

    def preprocess(self, src, tag, is_test=False):

        idxs = []
        new_src = []
        new_tag = []
        for i, d in enumerate(src):
            new_src.append([])
            new_tag.append([])
            for j, s in enumerate(d):
                # assert len(s)==len(tag[i][j]),(len(s),len(tag[i][j]))
                if len(s) > self.args.min_src_ntokens_per_sent:
                    new_src[-1].append(s[:self.args.max_src_ntokens_per_sent])
                    new_tag[-1].append(tag[i][j][:self.args.max_src_ntokens_per_sent])
        # print(len(new_tag))

        src_filt = [d[:self.args.max_src_nsents][:] for d in new_src if self.args.min_src_nsents < len(d)]
        tag_filt = [d_tag[:self.args.max_src_nsents][:] for d_tag in new_tag if self.args.min_src_nsents < len(d_tag)]
        print(len(src_filt), len(src_filt[0]), src_filt[0][0])
        src_txt = []
        # trans = str.maketrans("", "", string.punctuation + "‘" + "’" + "‐" + '‑' + '”')
        # chin_trans = str.maketrans("", "", punctuation)
        # zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
        # anotrans = str.maketrans("", "", "’")
        for d in src_filt:
            temp = []
            for sent in d:
                temp_s = sent
                for index, each_token in enumerate(sent):
                    if each_token.startswith("http"):
                        temp_s[index] = "http"
                    # elif len(each_token) > 1:
                    #     temp_s[index] = each_token.translate(trans)
                    #     temp_s[index] = temp_s[index].translate(chin_trans)
                    if temp_s[index] == "":
                        temp_s[index] = "[UNK]"
                    if each_token.startswith("www"):
                        temp_s[index] = "www"
                    if zhPattern.search(each_token):
                        temp_s[index] = "[UNK]"
                        # temp_s[index] = temp_s[index].translate(anotrans)
                temp.append(' '.join(temp_s))
            src_txt.append(temp)

        # src_txt = [' '.join(sent) for d in src for sent in d]
        # print(len(src_txt))

        text = [' {} {} '.format(self.sep_token, self.cls_token).join(d) for d in src_txt]
        # print(len(text))
        #text = ['{} '.format(self.cls_token) + d + ' {}'.format(self.sep_token) for d in text]
        # print(len(text))
        tags = []
        for i, t in enumerate(tag_filt):
            temp = []
            temp.append('O')
            for tag_list in t:
                for tag in tag_list:
                    temp.append(tag)
                temp.append('O')
                temp.append('O')
            temp = temp[:-2]
            temp.append('O')
            tags.append(temp)
            #assert len(text[i].split()) == len(temp), (i, text[i].split(), len(text[i].split()), len(temp))

        src_encoding = self.tokenizer(text, truncation=True, padding=True, max_length=512)
        src_subtoken_idxs = src_encoding['input_ids']
        src_token_type_id = src_encoding['token_type_ids']
        src_subtoken_origin_idxs = src_subtoken_idxs
        # print(len(src_subtoken_idxs), len(src_subtoken_idxs[0]))
        # print (len(tags[0]), len(src_subtoken_idxs[0]), len(src_token_type_id[0]))
        src_subtokens = [self.tokenizer.convert_ids_to_tokens(idx) for idx in src_subtoken_idxs]
        tag_align = []
        # print(src_subtokens[0])
        # print("debug one:", len(src_subtokens), len(tags))
        for i, subtoken in enumerate(src_subtokens):
            aligned_labels = ["O"] * len(subtoken)
            count = 0
            # print(len(subtoken))
            temp_src = text[i].split("[CLS]")
            temp = " ".join([val for val in subtoken if val != '[PAD]'])
            temp = temp.split("[CLS]")
            for j, temp_sent in enumerate(temp):
                if 0 < j:
                    temp_s = temp_sent.split()
                    temp_sent_filt = [filt for filt in temp_s if not filt.startswith("##")]
                    if len(temp_sent_filt) != len(temp_src[j - 1].split()):
                        print(len(temp_sent_filt), len(temp_src[j - 1].split()))
                        print("sub:", temp_sent_filt)
                        print("src:", temp_src[j - 1].split())
                        print("\n")
            print(i, len(tags[i]), len([val for val in subtoken if not val.startswith("##") and val != '[PAD]']))
            for j, each_str in enumerate(subtoken):
                # print(i, each_str, head, count)
                if each_str != "[PAD]":
                    if each_str.startswith("##"):
                        aligned_labels[j] = tags[i][count - 1]
                    else:
                        # print(each_str, len(tags[i]), count)
                        # try:
                        aligned_labels[j] = tags[i][count]
                        # except Exception:
                        #    pass
                        count += 1
                else:
                    break
            tag_align.append(aligned_labels)
        # print(head, len(src_subtokens), count, len(tags))
        # src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        tag_dict = {'O': 0, "I-INT": 1, "I-PAR": 2, "I-OUT": 3}
        tag_id = []
        for tag_list in tag_align:
            temp = []
            for tag in tag_list:
                temp.append(int(tag_dict[tag]))
            tag_id.append(temp)
        mask_label = []
        for i, tag_list in enumerate(tag_align):
            temp = []
            for j, tag in enumerate(tag_list):
                if tag == "O":
                    temp.append(0.0)
                else:
                    src_subtoken_idxs[i][j] = self.mask_vid
                    temp.append(1.0)
            # print(len(temp))
            mask_label.append(temp)
        data = []
        for i in range(len(mask_label)):
            data.append({"src": src_subtoken_idxs[i], "tag": tag_id[i], "mask": mask_label[i],
                         "token_type_ids": src_token_type_id[i],"src_orig":src_subtoken_origin_idxs[i]})
        return data

class PicoBioBertAdapterData():
    def __init__(self, args):
        self.args = args
        model_name = 'dmis-lab/biobert-v1.1'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]
        self.mask_token = '[MASK]'
        self.mask_vid = self.tokenizer.convert_tokens_to_ids(self.mask_token)

    def preprocess(self, src, tag, is_test=False):

        idxs = []
        new_src = []
        new_tag = []
        for i, d in enumerate(src):
            new_src.append([])
            new_tag.append([])
            for j, s in enumerate(d):
                # assert len(s)==len(tag[i][j]),(len(s),len(tag[i][j]))
                if len(s) > self.args.min_src_ntokens_per_sent:
                    new_src[-1].append(s[:self.args.max_src_ntokens_per_sent])
                    new_tag[-1].append(tag[i][j][:self.args.max_src_ntokens_per_sent])
        # print(len(new_tag))

        src_filt = [d[:self.args.max_src_nsents][:] for d in new_src if self.args.min_src_nsents < len(d)]
        tag_filt = [d_tag[:self.args.max_src_nsents][:] for d_tag in new_tag if self.args.min_src_nsents < len(d_tag)]
        print(len(src_filt), len(src_filt[0]), src_filt[0][0])
        src_txt = []
        # trans = str.maketrans("", "", string.punctuation + "‘" + "’" + "‐" + '‑' + '”')
        # chin_trans = str.maketrans("", "", punctuation)
        # zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
        # anotrans = str.maketrans("", "", "’")
        for d in src_filt:
            temp = []
            for sent in d:
                temp_s = sent
                for index, each_token in enumerate(sent):
                    if each_token.startswith("http"):
                        temp_s[index] = "http"
                    # elif len(each_token) > 1:
                    #     temp_s[index] = each_token.translate(trans)
                    #     temp_s[index] = temp_s[index].translate(chin_trans)
                    if temp_s[index] == "":
                        temp_s[index] = "[UNK]"
                    if each_token.startswith("www"):
                        temp_s[index] = "www"
                    if zhPattern.search(each_token):
                        temp_s[index] = "[UNK]"
                        # temp_s[index] = temp_s[index].translate(anotrans)
                temp.append(' '.join(temp_s))
            src_txt.append(temp)

        # src_txt = [' '.join(sent) for d in src for sent in d]
        # print(len(src_txt))

        text = [' {} {} '.format(self.sep_token, self.cls_token).join(d) for d in src_txt]
        # print(len(text))
        #text = ['{} '.format(self.cls_token) + d + ' {}'.format(self.sep_token) for d in text]
        # print(len(text))
        tags = []
        for i, t in enumerate(tag_filt):
            temp = []
            temp.append('O')
            for tag_list in t:
                for tag in tag_list:
                    temp.append(tag)
                temp.append('O')
                temp.append('O')
            temp = temp[:-2]
            temp.append('O')
            tags.append(temp)
            #assert len(text[i].split()) == len(temp), (i, text[i].split(), len(text[i].split()), len(temp))

        src_encoding = self.tokenizer(text, truncation=True, padding=True, max_length=512)
        src_subtoken_idxs = src_encoding['input_ids']
        src_token_type_id = src_encoding['token_type_ids']
        src_subtoken_origin_idxs = src_subtoken_idxs
        # print(len(src_subtoken_idxs), len(src_subtoken_idxs[0]))
        # print (len(tags[0]), len(src_subtoken_idxs[0]), len(src_token_type_id[0]))
        src_subtokens = [self.tokenizer.convert_ids_to_tokens(idx) for idx in src_subtoken_idxs]
        tag_align = []
        # print(src_subtokens[0])
        # print("debug one:", len(src_subtokens), len(tags))
        for i, subtoken in enumerate(src_subtokens):
            aligned_labels = ["O"] * len(subtoken)
            count = 0
            # print(len(subtoken))
            temp_src = text[i].split("[CLS]")
            temp = " ".join([val for val in subtoken if val != '[PAD]'])
            temp = temp.split("[CLS]")
            for j, temp_sent in enumerate(temp):
                if 0 < j:
                    temp_s = temp_sent.split()
                    temp_sent_filt = [filt for filt in temp_s if not filt.startswith("##")]
                    if len(temp_sent_filt) != len(temp_src[j - 1].split()):
                        print(len(temp_sent_filt), len(temp_src[j - 1].split()))
                        print("sub:", temp_sent_filt)
                        print("src:", temp_src[j - 1].split())
                        print("\n")
            print(i, len(tags[i]), len([val for val in subtoken if not val.startswith("##") and val != '[PAD]']))
            for j, each_str in enumerate(subtoken):
                # print(i, each_str, head, count)
                if each_str != "[PAD]":
                    if each_str.startswith("##"):
                        aligned_labels[j] = tags[i][count - 1]
                    else:
                        # print(each_str, len(tags[i]), count)
                        # try:
                        aligned_labels[j] = tags[i][count]
                        # except Exception:
                        #    pass
                        count += 1
                else:
                    break
            tag_align.append(aligned_labels)
        # print(head, len(src_subtokens), count, len(tags))
        # src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        tag_dict = {'O': 0, "I-INT": 1, "I-PAR": 2, "I-OUT": 3}
        tag_id = []
        for tag_list in tag_align:
            temp = []
            for tag in tag_list:
                temp.append(int(tag_dict[tag]))
            tag_id.append(temp)
        mask_label = []
        for i, tag_list in enumerate(tag_align):
            temp = []
            for j, tag in enumerate(tag_list):
                if tag == "O":
                    temp.append(0.0)
                else:
                    src_subtoken_idxs[i][j] = self.mask_vid
                    temp.append(1.0)
            # print(len(temp))
            mask_label.append(temp)
        data = []
        for i in range(len(mask_label)):
            data.append({"src": src_subtoken_idxs[i], "tag": tag_id[i], "mask": mask_label[i],
                         "token_type_ids": src_token_type_id[i],"src_orig":src_subtoken_origin_idxs[i]})
        return data

def format_to_robert(args):
    print('... (5) Converting data to roBERT data... this will take a while')

    datasets = ['train', 'valid', 'test']

    #corpora = [os.path.join(args.raw_path, f) for f in os.listdir(args.raw_path)
    #           if not f.startswith('.') and f.endswith('.json')]
    print("")
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            #print("json_f:", json_f, real_name)
            a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))
        
        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_robert, a_lst):
            pass
        pool.close()
        pool.join()


def format_to_bert(args):
    print('... (5) Converting data to BERT data... this will take a while')

    datasets = ['train', 'valid', 'test']

    # corpora = [os.path.join(args.raw_path, f) for f in os.listdir(args.raw_path)
    #           if not f.startswith('.') and f.endswith('.json')]
    print("")
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            # print("json_f:", json_f, real_name)
            a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))

        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_bert, a_lst):
            pass
        pool.close()
        pool.join()

def format_to_pubmed_bert(args):
    print('... (5) Converting data to pubmed BERT data... this will take a while')

    datasets = ['train', 'valid', 'test']

    # corpora = [os.path.join(args.raw_path, f) for f in os.listdir(args.raw_path)
    #           if not f.startswith('.') and f.endswith('.json')]
    print("")
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            # print("json_f:", json_f, real_name)
            a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))

        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_pubmed_bert, a_lst):
            pass
        pool.close()
        pool.join()

def format_to_bio_bert(args):
    print('... (5) Converting data to BioBERT data... this will take a while')

    datasets = ['train', 'valid', 'test']

    # corpora = [os.path.join(args.raw_path, f) for f in os.listdir(args.raw_path)
    #           if not f.startswith('.') and f.endswith('.json')]
    print("")
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            # print("json_f:", json_f, real_name)
            a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))

        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_bio_bert, a_lst):
            pass
        pool.close()
        pool.join()

def format_to_pico_adapter_robert(args):
    print('... (5) Converting data to pico adapter data... this will take a while')

    datasets = ['train', 'valid', 'test']

    # corpora = [os.path.join(args.raw_path, f) for f in os.listdir(args.raw_path)
    #           if not f.startswith('.') and f.endswith('.json')]
    print("")
    for corpus_type in datasets:
        json_file = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            json_file.append(json_f)
            # print("json_f:", json_f, real_name)
        save_path = pjoin(args.save_path, corpus_type+'.padpter.pt')
        is_test = corpus_type == 'test'
        if (os.path.exists(save_path)):
            logger.info('Ignore %s' % save_path)
            return
        pico_adapter = PicoAdapterData(args)
        logger.info('Processing %s' % json_file)
        data = []
        source = []
        tag = []
        #json_file = [json_file[0]]
        for j in json_file:
            jobs = json.load(open(j))
            for d in jobs:
                source.append(d['src'])
                tag.append(d['tag'])
        #source = source[:100]
        #tag = tag[:100]
        data = pico_adapter.preprocess(source, tag, is_test=is_test)
        logger.info('Processed instances %d' % len(data))
        logger.info('Saving to %s' % save_path)
        torch.save(data, save_path)

def format_to_pico_adapter_bert(args):
    print('... (5) Converting data to pico adapter data... this will take a while')

    datasets = ['train', 'valid', 'test']

    # corpora = [os.path.join(args.raw_path, f) for f in os.listdir(args.raw_path)
    #           if not f.startswith('.') and f.endswith('.json')]
    print("")
    for corpus_type in datasets:
        json_file = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            json_file.append(json_f)
            # print("json_f:", json_f, real_name)
        save_path = pjoin(args.save_path, corpus_type+'.padpter.pt')
        is_test = corpus_type == 'test'
        if (os.path.exists(save_path)):
            logger.info('Ignore %s' % save_path)
            continue
        pico_adapter = PicoBertAdapterData(args)
        logger.info('Processing %s' % json_file)
        data = []
        source = []
        tag = []
        #json_file = [json_file[0]]
        for j in json_file:
            jobs = json.load(open(j))
            for d in jobs:
                source.append(d['src'])
                tag.append(d['tag'])
        #source = source[:100]
        #tag = tag[:100]
        data = pico_adapter.preprocess(source, tag, is_test=is_test)
        logger.info('Processed instances %d' % len(data))
        logger.info('Saving to %s' % save_path)
        torch.save(data, save_path)

def format_to_pico_adapter_pubmed_bert(args):
    print('... (5) Converting data to pico adapter data... this will take a while')

    datasets = ['train', 'valid', 'test']

    # corpora = [os.path.join(args.raw_path, f) for f in os.listdir(args.raw_path)
    #           if not f.startswith('.') and f.endswith('.json')]
    print("")
    for corpus_type in datasets:
        json_file = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            json_file.append(json_f)
            # print("json_f:", json_f, real_name)
        save_path = pjoin(args.save_path, corpus_type+'.padpter.pt')
        is_test = corpus_type == 'test'
        if (os.path.exists(save_path)):
            logger.info('Ignore %s' % save_path)
            return
        pico_adapter = PicoPubmedBertAdapterData(args)
        logger.info('Processing %s' % json_file)
        data = []
        source = []
        tag = []
        #json_file = [json_file[0]]
        for j in json_file:
            jobs = json.load(open(j))
            for d in jobs:
                source.append(d['src'])
                tag.append(d['tag'])
        #source = source[:100]
        #tag = tag[:100]
        data = pico_adapter.preprocess(source, tag, is_test=is_test)
        logger.info('Processed instances %d' % len(data))
        logger.info('Saving to %s' % save_path)
        torch.save(data, save_path)

def format_to_pico_adapter_bio_bert(args):
    print('... (5) Converting data to pico adapter data... this will take a while')

    datasets = ['train', 'valid', 'test']

    # corpora = [os.path.join(args.raw_path, f) for f in os.listdir(args.raw_path)
    #           if not f.startswith('.') and f.endswith('.json')]
    print("")
    for corpus_type in datasets:
        json_file = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            json_file.append(json_f)
            # print("json_f:", json_f, real_name)
        save_path = pjoin(args.save_path, corpus_type+'.padpter.pt')
        is_test = corpus_type == 'test'
        if (os.path.exists(save_path)):
            logger.info('Ignore %s' % save_path)
            return
        pico_adapter = PicoBioBertAdapterData(args)
        logger.info('Processing %s' % json_file)
        data = []
        source = []
        tag = []
        #json_file = [json_file[0]]
        for j in json_file:
            jobs = json.load(open(j))
            for d in jobs:
                source.append(d['src'])
                tag.append(d['tag'])
        data = pico_adapter.preprocess(source, tag, is_test=is_test)
        logger.info('Processed instances %d' % len(data))
        logger.info('Saving to %s' % save_path)
        torch.save(data, save_path)

def _format_to_robert(params):
    corpus_type, json_file, args, save_file = params
    is_test = corpus_type == 'test'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    Robert = RoBertData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    for d in jobs:
        source, tgt = d['src'], d['tgt']
        sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 3)
        if (args.lower):
            source = [' '.join(s).lower().split() for s in source]
            tgt = [' '.join(s).lower().split() for s in tgt]
        b_data = Robert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
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
        source, tgt, label = d['src'], d['tgt'], d['label']
        if args.corpus != "pubmed":
            sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 3)
        else:
            #sent_labels = label
            #if sent_labels == []:
            sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 7)
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

def _format_to_pubmed_bert(params):
    corpus_type, json_file, args, save_file = params
    is_test = corpus_type == 'test'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    pubmed_bert = PubmedData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    for d in jobs:
        source, tgt = d['src'], d['tgt']
        sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 3)
        if (args.lower):
            source = [' '.join(s).lower().split() for s in source]
            tgt = [' '.join(s).lower().split() for s in tgt]
        b_data = pubmed_bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
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

def _format_to_bio_bert(params):
    corpus_type, json_file, args, save_file = params
    is_test = corpus_type == 'test'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bio_bert = BioBertData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    for d in jobs:
        source, tgt = d['src'], d['tgt']
        sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 3)
        if (args.lower):
            source = [' '.join(s).lower().split() for s in source]
            tgt = [' '.join(s).lower().split() for s in tgt]
        b_data = bio_bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
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
    if args.corpus != 'pubmed':
        corpora = sorted([os.path.join(args.raw_path, f) for f in os.listdir(args.raw_path)
                      if not f.startswith('.') and not f.endswith('.abs.txt.json') and not f.endswith('.tag.json')])
        #train_files, valid_files, test_files = [], [], []:
        args_list = []
        for f_main in corpora:
            f_abs_name = '{}.abs.txt.json'.format(os.path.basename(f_main).split('.')[0])
            f_abs = os.path.join(args.raw_path, f_abs_name)
            f_tag_name = '{}.tag.json'.format(os.path.basename(f_main).split('.')[0])
            f_tag = os.path.join(args.raw_path, f_tag_name)
            args_list.append((f_main, f_abs, f_tag, args))
        index_list = list(range(len(args_list)))
        random.shuffle(index_list)
        train_list_id = index_list[:int(len(args_list)*0.75)]
        eval_list_id = index_list[int(len(args_list)*0.75)+1:int(len(args_list)*0.9)]
        test_list_id = index_list[int(len(args_list)*0.9)+1:]
        train_files = [args_list[i] for i in train_list_id]
        valid_files = [args_list[i] for i in eval_list_id]
        test_files = [args_list[i] for i in test_list_id]
    else:
        root_data_dir = os.path.abspath(args.raw_path)
        train_files, valid_files, test_files = [], [], []
        test_txt_path = os.path.join(root_data_dir, 'test_pubmed')
        val_txt_path = os.path.join(root_data_dir, 'val_pubmed')
        train_txt_path = os.path.join(root_data_dir, 'train_pubmed')
        test_corpora = sorted([os.path.join(test_txt_path, f) for f in os.listdir(test_txt_path)
                              if not f.startswith('.') and not f.endswith('.abs.txt.json') and not f.endswith('.tag.json')])
        val_corpora = sorted([os.path.join(val_txt_path, f) for f in os.listdir(val_txt_path)
                               if not f.startswith('.') and not f.endswith('.abs.txt.json') and not f.endswith('.tag.json')])
        train_corpora = sorted([os.path.join(train_txt_path, f) for f in os.listdir(train_txt_path)
                              if not f.startswith('.') and not f.endswith('.abs.txt.json') and not f.endswith('.tag.json')])
        with open(os.path.join(root_data_dir, 'test.pkl'), 'rb') as f:
            test_label = pickle.load(f)
        with open(os.path.join(root_data_dir, 'val.pkl'), 'rb') as f:
            val_label = pickle.load(f)
        with open(os.path.join(root_data_dir, 'train.pkl'), 'rb') as f:
            train_label = pickle.load(f)
        for f_main in test_corpora:
            f_abs_name = '{}.abs.txt.json'.format(os.path.basename(f_main).split('.')[0])
            f_abs = os.path.join(test_txt_path, f_abs_name)
            f_tag_name = '{}.tag.json'.format(os.path.basename(f_main).split('.')[0])
            f_tag = os.path.join(test_txt_path, f_tag_name)
            paper_id = os.path.basename(f_main).split('.')[0]
            label = test_label[int(paper_id)]
            test_files.append((f_main, f_abs, f_tag, args, label))
        for f_main in val_corpora:
            f_abs_name = '{}.abs.txt.json'.format(os.path.basename(f_main).split('.')[0])
            f_abs = os.path.join(val_txt_path, f_abs_name)
            f_tag_name = '{}.tag.json'.format(os.path.basename(f_main).split('.')[0])
            f_tag = os.path.join(val_txt_path, f_tag_name)
            paper_id = os.path.basename(f_main).split('.')[0]
            label = val_label[int(paper_id)]
            valid_files.append((f_main, f_abs, f_tag, args, label))
        for f_main in train_corpora:
            f_abs_name = '{}.abs.txt.json'.format(os.path.basename(f_main).split('.')[0])
            f_abs = os.path.join(train_txt_path, f_abs_name)
            f_tag_name = '{}.tag.json'.format(os.path.basename(f_main).split('.')[0])
            f_tag = os.path.join(train_txt_path, f_tag_name)
            paper_id = os.path.basename(f_main).split('.')[0]
            label = train_label[int(paper_id)]
            train_files.append((f_main, f_abs, f_tag, args, label))

    start = time.time()
    print('... (4) Packing tokenized data into shards...')
    #print('Converting files count: {}'.format(len(corpora)))

    # imap executes in sync multiprocess manner
    # use array and shard_size to save the flow of ordered data
    corporas = {'train': train_files, 'valid': valid_files, 'test': test_files}
    for corpus_type in ['train', 'valid', 'test']:
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
            print('last shard {} saved'.format(shard_count))
            with open(fpath, 'w') as f:
                f.write(json.dumps(dataset))
            dataset = []
            shard_count += 1
    end = time.time()
    print('... Ending (4), time elapsed {}'.format(end - start))

def _format_to_lines(params):
    f_main, f_abs, f_tags, args, label = params
    source, tgt, tag= load_json(f_main, f_abs, f_tags)
    if not source:
        return None
    else:
        return {'src': source, 'tgt': tgt, "tag":tag, "label":label}

def format_xsum_to_lines(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'test', 'valid']

    corpus_mapping = json.load(open(pjoin(args.raw_path, 'XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json')))

    for corpus_type in datasets:
        mapped_fnames = corpus_mapping[corpus_type]
        root_src = pjoin(args.raw_path, 'restbody')
        root_tgt = pjoin(args.raw_path, 'firstsentence')
        # realnames = [fname.split('.')[0] for fname in os.listdir(root_src)]
        realnames = mapped_fnames

        a_lst = [(root_src, root_tgt, n) for n in realnames]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_xsum_to_lines, a_lst):
            if (d is None):
                continue
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []


def _format_xsum_to_lines(params):
    src_path, root_tgt, name = params
    f_src = pjoin(src_path, name + '.restbody')
    f_tgt = pjoin(root_tgt, name + '.fs')
    if (os.path.exists(f_src) and os.path.exists(f_tgt)):
        print(name)
        source = []
        for sent in open(f_src):
            source.append(sent.split())
        tgt = []
        for sent in open(f_tgt):
            tgt.append(sent.split())
        return {'src': source, 'tgt': tgt}
    return None
