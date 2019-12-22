# IR query preprocessor

#!/usr/bin/env python
# coding: utf-8

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

# import argparse

# import uuid
# import datetime
# import time

# import bz2
import argparse
# import pandas as pd
# # import dbmanager  as dbmanager
# from os.path import join

## My libraries

import eval_utils
import ir_utils
import bioasq_corpus_parser
# import bioasq_query_parser
import query_parser
# from ir_utils import *



class Query:
    def __init__(self, ir_toolkit_location, query_file, query_parameter_file, run_filename, stopwords_file):
        self.ir_toolkit_location = ir_toolkit_location
        self.query_file = query_file
        self.query_parameter_file = query_parameter_file
        self.run_filename = run_filename
        self.stopwords_file = stopwords_file
        
#     def build(self, ir_tool_params):
    def run(self):
        
#         ir_utils.create_dir(self.index_location)
    
        query_command = self.ir_toolkit_location + 'runquery/IndriRunQuery'
        toolkit_parameters = [
                                query_command,
                                self.query_file,
                                self.query_parameter_file,
                                self.stopwords_file]
        print(toolkit_parameters)
        with open(self.run_filename, 'wt') as rf:
#             proc = subprocess.Popen(toolkit_parameters,stdin=subprocess.PIPE, stdout=rf, stderr=subprocess.STDOUT, shell=False)
            proc = subprocess.Popen(toolkit_parameters,stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
            proc2 = subprocess.Popen(['grep', '^.*[ ]Q0[ ]'],stdin=proc.stdout, stdout=rf, stderr=subprocess.STDOUT, shell=False)
            (out, err)= proc2.communicate()

#             print('Run error: ', err)
            if err == None:
                pass
#                 return 'Ok'


# Specific for BioASQ
# for BioASQ only

def get_doc_year(corpus_dir):
    doc_year_files = [os.path.join('./',root, name)
             for root, dirs, files in os.walk(corpus_dir)
             for name in files
             if all(y in name for y in ['year'])]
#     print(doc_year_files)
    
    doc_years_dict = {}
    for doc_year in doc_year_files:
        with open(doc_year, 'rt') as dy_f:
            dy_dict = json.load(dy_f)
            doc_years_dict.update(dy_dict)
    return doc_years_dict


# In[ ]:


# Specific for BioASQ
# for BioASQ only

def filter_year(run_filename, run_filename_filtered, doc_years_dict):
    if 'train' in run_filename:
        filter_year = 2015
    else:
        filter_year = 2016
    with open(run_filename, 'rt') as rf:
        run_dict = {}
        doc_rank = 0
        for line in rf:

            elem = line.split(' ')
            q_id = elem[0]
            doc = elem[2]
            
            doc_score = elem[4]
            doc_year = doc_years_dict[doc]            
            if int(doc_year) <= filter_year:
                if q_id in run_dict:
                    doc_rank += 1
                    if doc_rank == 100:
                        pass
#                         print('orig: ', elem[3])                        
                    if doc_rank > 100:
                        continue
                    docu = [q_id, doc, doc_rank, doc_score, doc_year]
                    run_dict[q_id].append(docu)
                else:
                    if doc_rank < 100:
                        try: 
#                             print('previous: ',  docu)
                            pass
                        except: 
                            pass
                    doc_rank = 1
                    docu = [q_id, doc, doc_rank, doc_score, doc_year]
                    run_dict[q_id] = [docu]

        with open(run_filename_filtered, 'wt') as filter_f:
            for key, value in run_dict.items():
                for val in value:
                    filter_f.write(val[0] + ' Q0 ' + val[1] + ' ' + str(val[2]) + ' ' + str(val[3]) + ' indri\n')


class fakeParser:
    def __init__(self):
        self.dataset = 'bioasq' 
        self.data_split = 'test'
#         self.data_split = 'train'
#         self.data_split = 'dev'
#         self.build_index = True
        self.build_index = None
        self.fold = '1'
        self.gen_features = True
#         self.gen_features = None
        

if __name__ == "__main__":
    
    
# #     ir_toolkit_location = sys.argv[1] # '../indri/'

#     # create dataset files dir
    
    parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
    parser.add_argument('--dataset',   type=str, help='')
    parser.add_argument('--data_split',   type=str, help='')
    parser.add_argument('--pool_size',   type=int, help='')
#     parser.add_argument('--build_index', action='store_true')
    parser.add_argument('--fold', type=str,   help='')
#     parser.add_argument('--gen_features', action='store_true')
    
    
    args=parser.parse_args()
#     args = fakeParser()
    ir_toolkit_location = '../../../indri-l2r/'
    
    dataset = args.dataset
    workdir = './' + dataset + '_dir/'
    to_index_dir =  workdir + dataset + '_corpus/'
    ir_utils.create_dir(workdir)
    confdir = './' + dataset + '_config/'
    parameter_file_location = confdir + dataset + '_index_param_file'
    stopwords_file = confdir + 'stopwords'
    
    
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
        ir_utils.create_dir(fold_dir)
    #     # Generate qrels and qret

        if args.data_split == 'all':
            data_splits = ['train', 'dev', 'test']
        else:
            data_splits = [args.data_split]

            
        for data_split in data_splits:
                
            if args.dataset == 'bioasq':
                queries_file = '../../bioasq_data/bioasq.' + data_split + '.json'
                prefix = dataset + '_' + data_split

            elif args.dataset == 'robust':
                queries_file = '../../robust04_data/split_' + fold + '/rob04.' +  data_split + '.s' + fold + '.json'
                prefix = dataset + '_' + data_split + '_s' + fold 


#             prefix = queries_file.split('/')[-1].strip('.json')
#             filename_prefix = workdir + prefix
            filename_prefix = fold_dir + prefix


        #     print(filename_prefix)

            trec_query_file = filename_prefix + '_trec_query'
            qrels_file = filename_prefix + '_qrels'

        #     bioasq_query_parser.query_parser(queries_file, trec_query_file, qrels_file) # fast
            query_parser.query_parser(queries_file, trec_query_file, qrels_file) # fast

             # Run query

            run_filename = fold_dir + 'run_bm25_' + prefix
#             run_filename = workdir + 'run_bm25_' + prefix
            query_parameter_file = confdir + dataset + '_query_params'


            bm25_query = Query(ir_toolkit_location, trec_query_file, query_parameter_file, run_filename, stopwords_file)
            bm25_query.run() # fast


            # BIOASQ: Filter docus by year
            if args.dataset == 'bioasq':
                doc_years_dict = get_doc_year(to_index_dir)
                run_filename_filtered = run_filename + '_filtered'
                filter_year(run_filename, run_filename_filtered, doc_years_dict)


        # #     # Eval
        #     results = eval(trec_eval_command, qrels_file, run_filename_filtered)    
        #     print(results)

            