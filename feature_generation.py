#!/usr/bin/python 
# coding: utf-8

import sys
import pandas as pd
import numpy as np
import re 
import nltk 
import datetime 
import matplotlib.pyplot as plt
import uuid
import ujson
import sqlite3
import sys
import psycopg2
import shutil
import os
import logging 

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, TruncatedSVD

from nltk.corpus import stopwords
from nltk import SnowballStemmer

def generate_features( OVERLAP_MODE = 'remove', DO_REPEAT_SCORE = 1, DOTEST = 0):
    """
    bool DOTEST: should we run in test mode? 
    """ 

    #### Options ##############################################################

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # parameters for vectorizer 
    ANALYZER = "word" # unit of features are single words rather then phrases of words 
    STRIP_ACCENTS = 'unicode' 
    TOKENIZER = None
    NGRAM_RANGE = (0,2) # Range for n-grams 
    MAX_DF = 0.8  # Exclude words that are contained in more than x percent of documents 

    ###########################################################################

    # read in data 
    test_file = 'data/test.csv'
    train_file = 'data/train.csv'
    
    tr = pd.read_csv(train_file)
    te = pd.read_csv(test_file)

    Ntr = tr.shape[0]
    Nte = te.shape[0]

    if DOTEST: 
        tr = tr.iloc[1:1000]
        te = te.iloc[1:1000]
        MIN_DF = 1.0
        MAX_DF = 1.0
        Ntr = 1000
        Nte = 1000 
    else: 
        MIN_DF = 100/Ntr
    
    logging.info( """
        Options
        OVERLAP_MODE: {}
        DO_REPEAT_SCORE: {}
        DOTEST: {} 

        ANALYZE: {}
        STRIP_ACCENTS: {}
        TOKENIZER: {}
        NGRAM_RANGE: {} 
        MAX_DF: {} 
        MIN_DF: {} 
        """.format( OVERLAP_MODE, DO_REPEAT_SCORE, DOTEST, ANALYZER, \
            STRIP_ACCENTS, TOKENIZER, NGRAM_RANGE, MAX_DF, MIN_DF) ) 

    
    # get list of comments     
    train_comments = list( tr.comment_text )
    test_comments = list( te.comment_text )
    train_ids = tr.id 
    test_ids = te.id 
    
    # preprocess string 
    RE_PREPROCESS = re.compile(r""" \W + # one or more nonword characters
                                    |    # the or operator
                                    \d+  # digits""", re.VERBOSE)
   
    # regexp preprocessing  
    processed_tr_comments = np.array( [ re.sub(RE_PREPROCESS, ' ', comment).lower() for comment in train_comments] )
    processed_te_comments = np.array( [ re.sub(RE_PREPROCESS, ' ', comment).lower() for comment in test_comments] )
                                
    
    tr_vectorizer = CountVectorizer(analyzer=ANALYZER,
                                tokenizer=None, # alternatively tokenize_and_stem but it will be slower 
                                ngram_range=NGRAM_RANGE,
                                stop_words = stopwords.words('english'),
                                strip_accents=STRIP_ACCENTS,
                                min_df = MIN_DF,
                                max_df = MAX_DF)
    
    te_vectorizer = CountVectorizer(analyzer=ANALYZER,
                                tokenizer=None, # alternatively tokenize_and_stem but it will be slower 
                                ngram_range=NGRAM_RANGE,
                                stop_words = stopwords.words('english'),
                                strip_accents=STRIP_ACCENTS,
                                min_df = MIN_DF,
                                max_df = MAX_DF)
    
    # tokenize  
    train_bag_of_words = tr_vectorizer.fit_transform( processed_tr_comments ) 
    test_bag_of_words = te_vectorizer.fit_transform( processed_te_comments )
    
    tr_vocab = tr_vectorizer.get_feature_names()
    te_vocab = te_vectorizer.get_feature_names()
    
    if OVERLAP_MODE: 
        # compute percent overlap in vocabulary 
        overlap = set(tr_vocab).intersection( te_vocab )
    
        # Remove any features that aren't in both training and test 
        train_in_overlap  = [i for i, word in enumerate(tr_vocab) if word in overlap]
        test_in_overlap  = [i for i, word in enumerate(te_vocab) if word in overlap]
        tr_vocab2 = [tr_vocab[i] for i in train_in_overlap]
        te_vocab2 = [te_vocab[i] for i in test_in_overlap]
        train_bag_of_words = train_bag_of_words[:,train_in_overlap]
        test_bag_of_words = test_bag_of_words[:, test_in_overlap]
    
    # Generate repeat score 
    if DO_REPEAT_SCORE: 
        train_numerator = (train_bag_of_words.getnnz(1) + 1) 
        train_denominator = (train_bag_of_words.sum(1) + 1) 
        train_numerator = train_numerator.reshape( train_numerator.shape[0], 1)

        test_numerator = (test_bag_of_words.getnnz(1) + 1) 
        test_denominator = (test_bag_of_words.sum(1) + 1) 
        test_numerator = test_numerator.reshape( test_numerator.shape[0], 1)
    
        train_repeat_score = np.divide( train_numerator, train_denominator)
        test_repeat_score = np.divide( test_numerator, test_denominator)
    
    # Create Feature DataFrame 
    train_feat = pd.DataFrame( train_repeat_score, columns = ['repeat_score'] )
    test_feat = pd.DataFrame( test_repeat_score, columns = ['repeat_score'] )

    train_feat.index = tr.index
    test_feat.index = tr.index

    return train_bag_of_words, test_bag_of_words, train_feat, test_feat 

