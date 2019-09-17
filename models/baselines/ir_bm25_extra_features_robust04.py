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


# REMOVE!!
from eval_utils import *


# In[3]:


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
#     extra_features = features
    
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


# In[7]:


# Classes


# In[8]:


# Main
if __name__ == "__main__":
## System inputs, main variable options

#     dataset = sys.argv[1] # 'bioasq'
#     workdir = './' + dataset + '_dir/'
#     data_split = sys.argv[2] # 'test'

#     features_file = './bioasq_dir/bioasq.dev_features'
    workdir = './robust_dir/'
    fold = 's1'
    fold_dir = workdir + fold + '/'
    
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)
    
    train_features_file = fold_dir+ 'rob04.train.' + fold + '_features'
    test_features_file = fold_dir + 'rob04.test.' + fold + '_features'
    
    # train model
    linear_model = train_linear_model(train_features_file)
    
# In[9]:


# predict
ranked_dict = predict(test_features_file, linear_model)


# In[10]:


run_linear_model_file = fold_dir + 'run_rob04_linearModel.test.' + fold
write_predictions_run_file(ranked_dict, run_linear_model_file)


# In[11]:


trec_eval_command = '../../eval/trec_eval'
qrels_file = fold_dir + 'rob04.test.' + fold + '_qrels'
eval(trec_eval_command, qrels_file, run_linear_model_file)


# In[12]:


trec_eval_command = '../../eval/trec_eval'

run_bm25_file = fold_dir + 'run_bm25_rob04.test.' + fold 
eval(trec_eval_command, qrels_file, run_bm25_file)


# In[13]:


linear_model.intercept_

