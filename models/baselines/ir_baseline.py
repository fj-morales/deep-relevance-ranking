#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import pickle
# import json
# import gzip
# import os
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
        rf = open(self.run_filename, 'wt') 
        proc = subprocess.Popen(toolkit_parameters,stdin=subprocess.PIPE, stdout=rf, stderr=subprocess.STDOUT, shell=False)
        (out, err) = proc.communicate()
#         print(out.decode("utf-8"))
#         print('Run error: ', err)
#         if err == None:
#             return 'Ok'


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


if __name__ == "__main__":
    
#     ir_toolkit_location = sys.argv[1] # '../indri/'
#     parameter_file_location = sys.argv[2] # './bioasq_index_param_file'

    # create dataset files dir
    
    dataset = 'bioasq'
    workdir = dataset + '_dir/'
    
    # generate corpus files to index
    
    pool_size = 40 # scales very well when server is not used
    # Get all filenames
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

