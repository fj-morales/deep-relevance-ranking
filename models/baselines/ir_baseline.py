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
# import pandas as pd
# # import dbmanager  as dbmanager
# from os.path import join

## My libraries

import utils
import bioasq_corpus_parser
import bioasq_query_parser
# from utils import *


# In[ ]:


class Index:
    def __init__(self, ir_toolkit_location, parameter_file_location):
        self.ir_toolkit_location = ir_toolkit_location
        self.parameter_file_location = parameter_file_location
#     def build(self, ir_tool_params):
    def build(self):
        
#         utils.create_dir(self.index_location)
    #     index_loc_param = '--indexPath=' + index_loc
        stopwords_file = './stopwords'
        build_index_command = self.ir_toolkit_location + 'bin/IndriBuildIndex'
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
        query_command = self.ir_toolkit_location + 'bin/IndriRunQuery'
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

            print('Run error: ', err)
            if err == None:
                return 'Ok'


# In[ ]:


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

    print(toolkit_parameters)

    proc = subprocess.Popen(toolkit_parameters, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
    (out, err) = proc.communicate()
    print(out.decode("utf-8"))
    print('Run error: ', err)
    if err == None:
        return 'Ok'


# In[ ]:


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


# In[ ]:


if __name__ == "__main__":
    
    
# #     ir_toolkit_location = sys.argv[1] # '../indri/'
# #     parameter_file_location = sys.argv[2] # './bioasq_index_param_file'

#     # create dataset files dir
    
    
    
    dataset = sys.argv[1] # 'bioasq'
    workdir = './' + dataset + '_dir/'
    split = sys.argv[2] # 'test'
    
    try:
        build_index_flag = sys.argv[3] # True
    except:
        build_index_flag = False
    
#     # generate corpus files to index
    
    pool_size = 40 # scales very well when server is not used
#     # Get all filenames
    
#     # Options
#     data_dir = '/ssd/francisco/pubmed19-test/'
    data_dir = '/ssd/francisco/pubmed19/'
    
    
    to_index_dir =  workdir + dataset + '_corpus/'
    index_dir = workdir + dataset + '_indri_index'

    ir_toolkit_location = '../../../indri/'
    trec_eval_command = '../../eval/trec_eval'
    parameter_file_location = workdir + 'bioasq_index_param_file'

    
#     # Generate query files
    
    if build_index_flag == True: 
        
        
        utils.create_dir(to_index_dir)
        utils.create_dir(index_dir)

        # Parse Pubmed (BioASQ) dataset
        bioasq_corpus_parser.corpus_parser(data_dir, to_index_dir, pool_size) # time consuming

        index_data = Index(ir_toolkit_location, parameter_file_location)
        index_data.build() # time consuming

    
#     # Generate qrels and qret
    
    queries_file = '../../bioasq_data/bioasq.' + split + '.json'

    prefix = queries_file.split('/')[-1].strip('.json')
    filename_prefix = workdir + prefix
    
#     print(filename_prefix)
    
    trec_query_file = filename_prefix + '_trec_query'
    qrels_file = filename_prefix + '_qrels'
    
    bioasq_query_parser.query_parser(queries_file, trec_query_file, qrels_file) # fast
    
#     # Run query
    

    run_filename = workdir + 'run_' + prefix
    query_parameter_file = workdir + dataset + '_query_params'
    

    bm25_query = Query(ir_toolkit_location, trec_query_file, query_parameter_file, run_filename)
    bm25_query.run() # fast
    
    # BIOASQ: Filter docus by year
    doc_years_dict = get_doc_year(to_index_dir)
    run_filename_filtered = run_filename + '_filtered'
    filter_year(run_filename, run_filename_filtered, doc_years_dict)
#     # Eval
    eval(trec_eval_command, qrels_file, run_filename_filtered)    

