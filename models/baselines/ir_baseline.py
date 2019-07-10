#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
# import sys
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


# In[2]:


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


# In[3]:


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


# In[4]:


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


# In[5]:


if __name__ == "__main__":
    
#     ir_toolkit_location = sys.argv[1] # '../indri/'
#     parameter_file_location = sys.argv[2] # './bioasq_index_param_file'

    # create dataset files dir
    
    dataset = 'bioasq'
    workdir = './' + dataset + '_dir/'
    
    # generate corpus files to index
    
    pool_size = 40 # scales very well when server is not used
    # Get all filenames
    
    # Options
#     data_dir = '/ssd/francisco/pubmed19-test/'
    data_dir = '/ssd/francisco/pubmed19/'
    
    
    to_index_dir =  workdir + dataset + '_corpus/'
    index_dir = workdir + dataset + '_indri_index'
    utils.create_dir(to_index_dir)
    utils.create_dir(index_dir)
    
    
    
    
    bioasq_corpus_parser.corpus_parser(data_dir, to_index_dir, pool_size)

    

    # Generate query files
    
    
    ir_toolkit_location = '../../../indri/'
    trec_eval_command = '../../eval/trec_eval'
    parameter_file_location = workdir + 'bioasq_index_param_file'
    
    
    
    
    
    index_data = Index(ir_toolkit_location, parameter_file_location)
    index_data.build()
    
    
    
    
    
    # Generate qrels and qret
    
    queries_file = '../../bioasq_data/bioasq.test.json'
    
    prefix = queries_file.split('/')[-1].strip('.json')
    filename_prefix = workdir + prefix
    
    print(filename_prefix)
    
    trec_query_file = filename_prefix + '_trec_query'
    qrels_file = filename_prefix + '_qrels'
    
    bioasq_query_parser.query_parser(queries_file, trec_query_file, qrels_file)
    
    # Run query
    

    run_filename = workdir + 'run_' + prefix
    query_parameter_file = workdir + dataset + '_query_params'
    

    bm25_query = Query(ir_toolkit_location, trec_query_file, query_parameter_file, run_filename)
    bm25_query.run()
    

    # Eval
    eval(trec_eval_command, qrels_file, run_filename)


# In[9]:


# Specific for BioASQ
# for BioASQ only

def rfile_dict(run_filename):
    with open(run_filename, 'rt') as rf:
        run_dict = {}
        for line in rf:
            elem = line.split(' ')
            q_id = elem[0]
            doc = elem[2]
            doc_rank = elem[3]
            doc_score = elem[4]
            docu = [q_id, doc, doc_rank, doc_score]
            if q_id in run_dict:
                run_dict[q_id].append(docu)
            else:
                run_dict[q_id] = [docu]
        return run_dict


# In[10]:


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


# In[11]:


dyears = get_doc_year(to_index_dir)


# In[12]:


len(dyears)

