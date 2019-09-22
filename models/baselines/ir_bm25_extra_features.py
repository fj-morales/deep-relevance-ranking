#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Reanking BM25 + extra features with linear model


# In[2]:


# Imports 
# For Linear model
# Inspiration from:
# https://opensourceconnections.com/blog/2017/04/01/learning-to-rank-linear-models/
from sklearn.linear_model import LinearRegression
# from math import sin
import numpy as np
from itertools import groupby
import argparse


# REMOVE!!
from utils import *
from eval_utils import *


# In[3]:


# Functions

def load_features(features_file):
    with open(features_file, 'rt') as ff:
        rels = []
        qids = []
        features =  []
        docids = []
        for feature_line in ff:
            cols = feature_line.split(' ')
            rels.append(cols[0])
            qids.append(cols[1].split(':')[1])
            features.append([x.split(':')[1] for x in cols[2:-1]])
            docids.append(cols[-1:][0].split('=')[1].strip('\n'))
    return [np.array(rels, dtype=np.int), 
            qids, 
            np.array(features, dtype=np.float), 
            docids]


# In[4]:


def train_linear_model(train_features_file):
    [rels, qids, features, docids] = load_features(train_features_file)
    
    # Slice info for L2R Linear Model (Deep Relevance Ranking paper)
    # 4 extra features
    extra_features = features[:,-4:]
    # Fitting linear model
    # Ordinary Least Squares Linear Regression 
    linearModel = LinearRegression()
    linearModel.fit(extra_features, rels)
    
#     print(linearModel.coef_)
#     print(linearModel.intercept_)
    return linearModel


# In[5]:


def predict(test_file, linear_model):
    [rels, qids, features, docids] = load_features(test_file)
    extra_features = features[:,-4:]
    
    predictions = linear_model.predict(extra_features)
    print(len(predictions))
    
    queries_dict = {}
    for i, qid in enumerate(qids):
        if qid not in queries_dict.keys():
            queries_dict[qid] = [[qid, docids[i], predictions[i]]]
        else:
#             print(qid, docids[i], predictions[i])
            queries_dict[qid].append([qid, docids[i], predictions[i]])
    
    # Sort predictions
    {qid:value.sort(key=lambda x: x[2], reverse=True) for (qid, value) in queries_dict.items()}
    # Add ranking number
    {qid:[x.append(value.index(x) + 1) for x in value] for (qid, value) in queries_dict.items()}

    return queries_dict


# In[6]:


def write_predictions_run_file(predictions_dict, filename):
    with open(filename, 'wt') as f_out:
        for qid, value in predictions_dict.items():
            [f_out.write(x[0] + ' Q0 ' +  x[1] + ' ' + str(x[3]) + ' ' + str(x[2])[0:7] + ' linearModel\n') for x in value]


class fakeParser:
    def __init__(self):
        self.dataset = 'bioasq' 
#         self.build_index = True
        self.build_index = None
        self.data_spli = 'all'
        self.fold = '1'
        self.gen_features = True
#         self.gen_features = None
            


# Main
if __name__ == "__main__":
## System inputs, main variable options

    parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
    parser.add_argument('--dataset',   type=str, help='')
    parser.add_argument('--data_split',   type=str, help='')
    parser.add_argument('--fold', type=str,   help='')
    
    args=parser.parse_args()
#     args = fakeParser()

    dataset = args.dataset
    workdir = './' + dataset + '_dir/'
    create_dir(workdir)
    
    
    if (not args.fold or args.dataset == 'bioasq'):
        args.fold = ['']
    elif args.fold == 'all':
        args.fold = ['1','2','3','4','5']
    else:
        args.fold = [args.fold]
    
    
    for f in args.fold:
        
        fold = f # '1'
        
        if args.dataset == 'bioasq':
            fold_dir = workdir
        else:
            fold_dir = workdir + 's' + fold + '/'

    #     features_file = './bioasq_dir/bioasq.dev_features'

        if args.dataset == 'bioasq':
            train_features_file = './bioasq_dir/bioasq_train_features'
    #         val_features_file = './bioasq_dir/bioasq_dev_features'
            test_features_file = './bioasq_dir/bioasq_test_features'

            run_linear_model_file = workdir + 'run_' + dataset + '_linearModel_test_filtered'
            qrels_file = workdir + dataset + '_test_qrels'
            run_bm25_file = workdir + 'run_bm25_' + dataset + '_test_filtered'

        elif args.dataset == 'robust':
            train_features_file = fold_dir+ 'robust_train_s' + fold + '_features'
    #         val_features_file = fold_dir+ 'robust_dev_s' + fold + '_features'
            test_features_file = fold_dir + 'robust_test_s' + fold + '_features'
            run_linear_model_file = fold_dir + 'run_robust_linearModel_test_s' + fold
            qrels_file = fold_dir + 'robust_test_s' + fold + '_qrels'
            run_bm25_file = fold_dir + 'run_bm25_robust_test_s' + fold

        # train model
        linear_model = train_linear_model(train_features_file)

        # predict
        ranked_dict = predict(test_features_file, linear_model)

        # In[10]:

        write_predictions_run_file(ranked_dict, run_linear_model_file)

        # In[17]:

        # BM25+Extra features linear model
        trec_eval_command = '../../eval/trec_eval'
        linear_model_results = eval(trec_eval_command, qrels_file, run_linear_model_file)
        linear_model_results['model'] = 'bm25+extra'
        linear_model_results['model_file'] = run_linear_model_file
        print('Linear model results: \n', linear_model_results)
        
        # BM25 (default, vanilla)  model
        trec_eval_command = '../../eval/trec_eval'
        bm25_results = eval(trec_eval_command, qrels_file, run_bm25_file)
        bm25_results['model'] = 'bm25+extra'
        bm25_results['model_file'] = run_bm25_file
        print('BM25 (default) results: \n', bm25_results)
        

