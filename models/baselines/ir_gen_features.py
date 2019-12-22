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
from ir_utils import load_queries
import bioasq_corpus_parser
# import bioasq_query_parser
import query_parser
# from ir_utils import *
from join_split_files import join_files



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
    print(features_split_file)
    with open(features_split_file, 'wt') as out_f:
        for qid in qids_list:
            to_write = all_features_dict[qid]
            out_f.write("".join(to_write))


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
    parser.add_argument('--fold', type=str,   help='')
    
    args=parser.parse_args()
#     args = fakeParser()
    ir_toolkit_location = '../../../indri-l2r/'
    
    dataset = args.dataset
    workdir = './' + dataset + '_dir/'
    to_index_dir =  workdir + dataset + '_corpus/'
    index_dir = workdir + dataset + '_indri_index'
    ir_utils.create_dir(workdir)
    confdir = './' + dataset + '_config/'
    parameter_file_location = confdir + dataset + '_index_param_file'
    stopwords_file = confdir + 'stopwords'
    
    
    if (not args.fold or args.dataset == 'bioasq'):
        args.fold = ['']
    elif args.fold == 'all':
        args.fold = ['1','2','3','4','5']
#         args.fold = ['1']
    else:
        args.fold = [args.fold]
    
    # Generate all features for robust
    # Later, divide them according to fold and data_split
    if args.dataset == 'robust':
        pass
        gen_features_param_file = workdir + dataset + '_gen_features_params'
        out_features_file = workdir + dataset + '_features'

        [all_dict_queries_file, run_filename_all, qrels_all_file, trec_query_all_file] = join_files()
        
        features_params =[
                trec_query_all_file,
                index_dir,
                out_features_file,
                run_filename_all,
                qrels_all_file,
                'krovetz', # This should not be here!, Fix GenerateExtraFeatures.cpp to read from index manifest
        ]   
        
        generate_features_params(features_params, gen_features_param_file)

        # Generate L2R features 
        feature_generator = GenerateExtraFeatures(ir_toolkit_location, gen_features_param_file)
        feature_generator.run()
        
        # Get all features in a dict
        feat_dic = features_dict(out_features_file)
    
    
    # Get features for every fold and data_split
    
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
#                 prefix = queries_file.split('/')[-1].strip('.json')
                prefix = dataset + '_' + data_split
                filename_prefix = workdir + prefix
                trec_query_file = filename_prefix + '_trec_query'
                run_filename = workdir + 'run_bm25_' + prefix
                run_filename_filtered = run_filename + '_filtered'
                qrels_file = filename_prefix + '_qrels'
                gen_features_param_file = filename_prefix + '_gen_features_params'
                out_features_file = filename_prefix + '_features'
                features_params =[
                    trec_query_file,
                    index_dir,
                    out_features_file,
                    run_filename_filtered,
                    qrels_file,
                    'krovetz', # This should not be here!, Fix GenerateExtraFeatures.cpp to read from index manifest
                ]    
                generate_features_params(features_params, gen_features_param_file)

                    # Generate L2R features 

                feature_generator = GenerateExtraFeatures(ir_toolkit_location, gen_features_param_file)
                feature_generator.run()
                

            elif args.dataset == 'robust':
                queries_file = '../../robust04_data/split_' + fold + '/rob04.' +  data_split + '.s' + fold + '.json'
#                 q_file = queries_file
#                 prefix = queries_file.split('/')[-1].strip('.json')
                prefix = dataset + '_' + data_split + '_s' + fold 
                filename_prefix = fold_dir + prefix

                query_list = load_queries(queries_file)
                qid_list = [q['id'] for q in query_list] 

#                 print(filename_prefix)
                out_features_split_file = filename_prefix + '_features'

                print(len(feat_dic))
                print(len(qid_list))
                
                save_features(feat_dic, qid_list, out_features_split_file)
        
        
        
