#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %load ir_baseline.py
#!/usr/bin/env python

# In[ ]:


# import pickle
import json
# import gzip
import os
import subprocess
# import numpy as np
# import multiprocessing
# import re 
# import csv
# import torch
import sys
# import shutil
# import random
import argparse
# import argparse

# import uuid
# import datetime
# import time

# import bz2
# import pandas as pd
# # import dbmanager  as dbmanager
# from os.path import join

## My libraries

import utils
# import bioasq_corpus_parser
import query_parser
from join_split_files import join_files


# In[ ]:


def load_queries(queries_file):
    with open(queries_file, 'rb') as input_file:
        query_data = json.load(input_file)
        return query_data['questions']

# Functions

def features_dict(features_file):
    with open(features_file, 'rt') as f_file:
        feat_dict = {}
        for line in f_file:
            qid = line.split(' ')[1].split(':')[1]
            if qid in feat_dict.keys():
                feat_dict[qid].append(line)
            else:
                feat_dict[qid] = [line]
    return feat_dict

def save_features(all_features_dict,qids_list, features_split_file):
    with open(features_split_file, 'wt') as out_f:
        for qid in qids_list:
            to_write = all_features_dict[qid]
            out_f.write("".join(to_write))


class Index:
    def __init__(self, ir_toolkit_location, parameter_file_location):
        self.ir_toolkit_location = ir_toolkit_location
        self.parameter_file_location = parameter_file_location
#     def build(self, ir_tool_params):
    def build(self):
        
#         utils.create_dir(self.index_location)
    #     index_loc_param = '--indexPath=' + index_loc
        stopwords_file = './stopwords'
        build_index_command = self.ir_toolkit_location + 'buildindex/IndriBuildIndex'
        toolkit_parameters = [
                                build_index_command,
                                self.parameter_file_location,
                                stopwords_file
                                ]

        print(toolkit_parameters)

        proc = subprocess.Popen(toolkit_parameters,stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
        (out, err) = proc.communicate()
        print(out.decode("utf-8"))
        print('Index error: ', err)
        if err == None:
            return 'Ok'


# In[ ]:


class Query:
    def __init__(self, ir_toolkit_location, query_file, query_parameter_file, run_filename):
        self.ir_toolkit_location = ir_toolkit_location
        self.query_file = query_file
        self.query_parameter_file = query_parameter_file
        self.run_filename = run_filename
        
#     def build(self, ir_tool_params):
    def run(self):
        
#         utils.create_dir(self.index_location)
    
        stopwords_file = './stopwords'
        query_command = self.ir_toolkit_location + 'runquery/IndriRunQuery'
        toolkit_parameters = [
                                query_command,
                                self.query_file,
                                self.query_parameter_file,
                                stopwords_file]
        print(toolkit_parameters)
        with open(self.run_filename, 'wt') as rf:
#             proc = subprocess.Popen(toolkit_parameters,stdin=subprocess.PIPE, stdout=rf, stderr=subprocess.STDOUT, shell=False)
            proc = subprocess.Popen(toolkit_parameters,stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
            proc2 = subprocess.Popen(['grep', '^.*[ ]Q0[ ]'],stdin=proc.stdout, stdout=rf, stderr=subprocess.STDOUT, shell=False)
            (out, err)= proc2.communicate()

#             print(out.decode('utf-8'))
#             print('Run error: ', err)
            if err == None:
                pass
#                 return 'Ok'


# In[ ]:


def generate_features_params(params, feat_param_file):
    with open(params[0], 'rt') as q_trec_f:
        trec_lines = q_trec_f.readlines()
    
    with open(feat_param_file, 'wt') as f_out:
        for line in trec_lines[:-1]:
            f_out.write(line)
        f_out.write('<index>' + params[1] + '</index>\n')    
        f_out.write('<outFile>' + params[2] + '</outFile>\n')    
        f_out.write('<rankedDocsFile>' + params[3] + '</rankedDocsFile>\n')    
        f_out.write('<qrelsFile>' + params[4] + '</qrelsFile>\n')    
        f_out.write('<stemmer>' + params[5] + '</stemmer>\n') # This should not be here!, Fix GenerateExtraFeatures.cpp to read from index manifest
        f_out.write('</parameters>\n')    
         
        
class GenerateExtraFeatures:
    def __init__(self, ir_toolkit_location, feat_param_file):
        self.ir_toolkit_location = ir_toolkit_location
        self.feat_param_file = feat_param_file
        self.log_file = feat_param_file + '_run.log'
        
#     def build(self, ir_tool_params):
    def run(self):
        
        features_command = self.ir_toolkit_location + 'L2R-features/GenerateExtraFeatures'
        toolkit_parameters = [
                                features_command,
                                self.feat_param_file]
        print(toolkit_parameters)
        with open(self.log_file, 'wt') as rf:
            proc = subprocess.Popen(toolkit_parameters,stdin=subprocess.PIPE, stdout=rf, stderr=subprocess.STDOUT, shell=False)
#             proc = subprocess.Popen(toolkit_parameters,stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
            
            (out, err)= proc.communicate()
#             print(out.decode('utf-8'))
            print(err)
        print('Features file generated. Log: ', self.log_file)
            
        

def eval(trec_eval_command, qrel, qret):
    
    metrics = '-m map -m P.20 -m ndcg_cut.20'
    toolkit_parameters = [
                            trec_eval_command,
                            '-m',
                            'map',
                            '-m',
                            'P.20',
                            '-m',
                            'ndcg_cut.20',
                            qrel,
                            qret]

#     print(toolkit_parameters)

    proc = subprocess.Popen(toolkit_parameters, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
    (out, err) = proc.communicate()
#     print(out.decode("utf-8"))
#    print('Run error: ', err)
    if err == None:
        pass
#         print('No errors')
    return out.decode("utf-8")


# In[ ]:



class fakeParser:
    def __init__(self):
        self.dataset = 'robust' 
        self.data_split = 'test'
#         self.data_split = 'train'
#         self.data_split = 'dev'
        self.build_index = False
        self.fold = '1'
        self.gen_features_flag = False
        

if __name__ == "__main__":
    
    
# #     ir_toolkit_location = sys.argv[1] # '../indri/'

#     # create dataset files dir
    
    parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
    parser.add_argument('--dataset',   type=str, help='')
    parser.add_argument('--data_split',   type=str, help='')
    parser.add_argument('--buildindex', action='store_true')
    parser.add_argument('--fold', type=str,   help='')
    parser.add_argument('--gen_features', action='store_true')
    
    
#     args=parser.parse_args()
    args = fakeParser()
    
    
    gen_features_flag = args.gen_features_flag
    dataset = args.dataset # 
    workdir = './' + dataset + '_dir/'
    data_split =  args.data_split# 'test'
    fold = args.fold
    
    fold_dir = workdir + 's' + fold + '/'
    
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)
    
    try:
        build_index_flag = args.build_index # True
    except:
        build_index_flag = False
    
#     # generate corpus files to index
    
    pool_size = 40 # scales very well when server is not used
#     # Get all filenames
    
#     # Options
#     data_dir = '/ssd/francisco/pubmed19-test/'
    data_dir = '/ssd/francisco/deep-relevance-ranking/robust04_data/collection/'
    
    
    to_index_dir =  workdir + dataset + '_corpus/'
    index_dir = workdir + dataset + '_indri_index'

    ir_toolkit_location = '../../../indri-l2r/'
    trec_eval_command = '../../eval/trec_eval'
    parameter_file_location = workdir + dataset + '_index_param_file'

    print(fold_dir)
#     utils.create_dir(fold_dir)
    
    if build_index_flag == True: 
        
        utils.create_dir(index_dir)

        index_data = Index(ir_toolkit_location, parameter_file_location)
        index_data.build() # time consuming

    
#     # Generate qrels and qret
    
    queries_file = '../../robust04_data/split_' + fold + '/rob04.' +  data_split + '.s' + fold + '.json'

    print(queries_file)
    
    prefix = queries_file.split('/')[-1].strip('.json')
    filename_prefix = fold_dir + prefix
    
    print(filename_prefix)
    
    

    #     print(filename_prefix)
    
    trec_query_file = filename_prefix + '_trec_query'
    qrels_file = filename_prefix + '_qrels'
    print(trec_query_file)
    print(qrels_file)
    query_parser.query_parser(queries_file, trec_query_file, qrels_file) # fast
    

    # Run query
    run_filename = fold_dir + 'run_bm25_' + prefix
    query_parameter_file = workdir + dataset + '_query_params'
    

    bm25_query = Query(ir_toolkit_location, trec_query_file, query_parameter_file, run_filename)
    bm25_query.run() # fast
    
    

    # Eval
    eval(trec_eval_command, qrels_file, run_filename)    
    
    

    # Generate feature param file for all queries
    # Only generate if not existent!
    
#     trec_query_all_file = workdir + 'rob04.trec_queries.json'
#     qrels_all_file = workdir + 'rob04_qrels_all'
#     run_filename_all = workdir + 'run_bm25_rob04.all'
    
    gen_features_param_file = workdir + 'rob04' + '_gen_features_params'
    out_features_file = workdir + 'rob04' + '_features'
    
    if gen_features_flag:
    
        [all_dict_queries_file, run_filename_all, qrels_all_file, trec_query_all_file] = join_files()


        features_params =[
            trec_query_all_file,
            index_dir,
            out_features_file,
            run_filename_all,
            qrels_all_file,
            'krovetz', # This should not be here!, Fix GenerateExtraFeatures.cpp to read from index manifest
        ]
        print(features_params)
        print(gen_features_param_file)

        generate_features_params(features_params, gen_features_param_file)

        # Generate L2R features 

        feature_generator = GenerateExtraFeatures(ir_toolkit_location, gen_features_param_file)

        feature_generator.run()


# In[2]:


q_file = queries_file


query_list = load_queries(queries_file)
qid_list = [q['id'] for q in query_list] 

feat_dic = features_dict(out_features_file)

out_features_file = filename_prefix + '_features'

save_features(feat_dic, qid_list, out_features_file)

