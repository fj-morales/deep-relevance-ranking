
# coding: utf-8

# In[1]:


# %load ir_lmart.py
#!/usr/bin/env python

# In[1]:


# Stronger baseline: Listwise L2R - LambdaMART
# Hyperparameter optimziation HPonsteroids requires Python 3!


# In[2]:


# Imports
import os
import subprocess
import sys
from functools import partial
import multiprocessing
import numpy as np
import pickle
import json

# REMOVE!!
from eval_utils import *

# HPO
from hpo_utils import *

from utils import *

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

import logging
logging.basicConfig(level=logging.WARNING)

# HPO server and stuff

# import logging
# logging.basicConfig(level=logging.WARNING)

import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
from hpbandster.optimizers import RandomSearch as RS
from hpbandster.examples.commons import MyWorker


# In[2]:


# In[3]:


# Functions
def generate_run_file(pre_run_file, run_file):
    
    with open(pre_run_file, 'rt') as input_f:
        pre_run = input_f.readlines()
#         print(type(pre_run))
    with open(run_file, 'wt') as out_f:
        for line in pre_run:
            out_f.write(line.replace('docid=','').replace('indri', 'lambdaMART'))
        


# In[4]:


# Classes
class L2Ranker:
    def __init__(self, ranklib_location, params, normalization=[]):
        self.ranklib_location = ranklib_location
        # Works with Oracle JSE
        # java version "1.8.0_211"
        # Java(TM) SE Runtime Environment (build 1.8.0_211-b12)
        # Java HotSpot(TM) 64-Bit Server VM (build 25.211-b12, mixed mode)
        self.params = params
        self.ranker_command = ['java', '-jar', ranklib_location + 'RankLib-2.12.jar']
        self.normalization = normalization
        self.save_model_file = ''
        
#     def build(self, ir_tool_params):
    def train(self, train_data_file, save_model_file, hpo_config):
        self.save_model_file = save_model_file
        self.log_file = save_model_file + '.log'
        self.hpo_config= hpo_config
        toolkit_parameters = [
                                *self.ranker_command, # * to unpack list elements
                                '-train',
                                train_data_file,
                                *self.normalization,
                                *self.params,
                                '-leaf', 
                                str(self.hpo_config['n_leaves']),
                                '-shrinkage',
                                str(self.hpo_config['learning_rate']),
                                '-tree', # Oner regression tree per boosted iteration
                                str(self.hpo_config['n_trees']),
                                '-save',
                                self.save_model_file   
                            ] 
        
#         print(toolkit_parameters)
        with open(self.log_file, 'wt') as rf:
            proc = subprocess.Popen(toolkit_parameters,stdin=subprocess.PIPE, stdout=rf, stderr=subprocess.PIPE, shell=False)
#         proc = subprocess.Popen(toolkit_parameters,stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
            
        (out, err)= proc.communicate()
#         print(out.decode('utf-8').splitlines())
#         print(out)

        if err == b'':
            print('Model saved: ', self.save_model_file)
        else:
#             print('error:', err, type(err))
            print('Something went wrong on training, see log: ', self.log_file)
            
  

    def gen_run_file(self, test_data_file, run_file):
        pre_run_file = run_file.replace('run_', 'pre_run_', 1)
        toolkit_parameters = [
                                *self.ranker_command, # * to unpack list elements
                                '-load',
                                self.save_model_file,
                                *self.normalization,
                                '-rank',
                                test_data_file,
                                '-indri',
                                pre_run_file     
                            ] 
        
#         print(toolkit_parameters)
        with open(self.log_file, 'at') as rf:
            proc = subprocess.Popen(toolkit_parameters,stdin=subprocess.PIPE, stdout=rf, stderr=subprocess.STDOUT, shell=False)
#         proc = subprocess.Popen(toolkit_parameters,stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
            
        (out, err)= proc.communicate()
#         print(out.decode('utf-8').splitlines())
#         print(out)
        print(err)
    
        print(run_file)
        print(pre_run_file)
        
        generate_run_file(pre_run_file, run_file)
        
#         print('Run model saved: ', run_file)


# In[5]:


# try:
#     import keras
#     from keras.datasets import mnist
#     from keras.models import Sequential
#     from keras.layers import Dense, Dropout, Flatten
#     from keras.layers import Conv2D, MaxPooling2D
#     from keras import backend as K
# except:
#     raise ImportError("For this example you need to install keras.")

# try:
#     import torchvision
#     import torchvision.transforms as transforms
# except:
#     raise ImportError("For this example you need to install pytorch-vision.")


# In[3]:


class fakeParser:
    def __init__(self):
        self.min_budget = 2 
        self.max_budget = 4
        self.n_iterations = 4 
        self.n_workers =4
        self.dataset = 'bioasq' 
        self.data_split = 'test'
#         self.data_split = 'train'
#         self.data_split = 'dev'
#         self.build_index = True
        self.build_index = None
        self.fold = '1'
        self.gen_features = True
#         self.gen_features = None
        
# In[6]:


# In[3]:


# In[4]:


# Main
if __name__ == "__main__":
    
    # Options and variables
    
    parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
    parser.add_argument('--min_budget',   type=float, help='Minimum budget used during the optimization.',    default=2)
    parser.add_argument('--max_budget',   type=float, help='Maximum budget used during the optimization.',    default=4)
    parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=500)
    parser.add_argument('--n_workers', type=int,   help='Number of workers to run in parallel.', default=5)
    
    parser.add_argument('--dataset',   type=str, help='')
    parser.add_argument('--data_split',   type=str, help='')
    parser.add_argument('--pool_size',   type=int, help='')
#     parser.add_argument('--build_index', action='store_true')
    parser.add_argument('--fold', type=str,   help='')
#     parser.add_argument('--gen_features', action='store_true')

#     args=parser.parse_args()
    args = fakeParser()
    
#     dataset = sys.argv[1] # 'bioasq'
#     workdir = './' + dataset + '_dir/'
#     data_split = sys.argv[2] # 'test'

    dataset = args.dataset
    workdir = './' + dataset + '_dir/'
    confdir = './' + dataset + '_config/'
    data_split =  args.data_split
    fold = args.fold
        
    ranklib_location = '../../../ranklib/'
    
    
    if (not args.fold or args.dataset == 'bioasq'):
        args.fold = ['']
    elif args.fold == 'all':
        args.fold = ['1','2','3','4','5']
#         args.fold = ['1']
    else:
        args.fold = [args.fold]
        
    for f in args.fold:
        
        fold = f # '1'
        
        if args.dataset == 'bioasq':
            
  
            
            fold_dir = workdir
            qrels_val_file = fold_dir + dataset + '_' + 'dev' + '_qrels'
            dataset_fold = dataset 
            
            train_data_file = fold_dir + dataset + '_' + 'train'  + '_features'
            val_data_file = fold_dir + dataset + '_' + 'dev' +  '_features'
            test_data_file = fold_dir + dataset + '_' + 'test' + '_features'  
            
        elif args.dataset == 'robust':
            
            fold_dir = workdir + 's' + fold + '/'
            qrels_val_file = fold_dir + dataset + '_' + 'dev' + '_' + fold + '_qrels'
            dataset_fold = dataset + '_' + fold
            
            train_data_file = fold_dir + dataset + '_' + 'train' + '_' + fold + '_features'
            val_data_file = fold_dir + dataset + '_' + 'dev' + '_' + fold + '_features'
            test_data_file = fold_dir + dataset + '_' + 'test' + '_' + fold + '_features'

        l2r_model = 'lmart'

        enabled_features_file = confdir + dataset + '_' + l2r_model + '_enabled_features' # dont'f change to fold_dir!

    #     print(enabled_features_file)
        # Train L2R model: LambdaMART
        # Parameters 

        n_leaves = '10'
        learning_rate = '0.1'
        n_trees = '1000'
        hpo_params = {'n_leaves': n_leaves, 'learning_rate': learning_rate, 'n_trees': n_trees}



        metric2t = 'MAP' # 'MAP, NDCG@k, DCG@k, P@k, RR@k, ERR@k (default=ERR@10)'

        ranker_type = '6' # LambdaMART

        # normalization: Feature Engineering?
    #     norm_params = ['-norm', 'zscore'] # 'sum', 'zscore', 'linear'

        norm_params = ['-norm', 'zscore'] # 'sum', 'zscore', 'linear'

        l2r_params = [
            '-validate',
            val_data_file,
            '-ranker',
            ranker_type,
            '-metric2t',
            metric2t,
            '-feature',
            enabled_features_file
        ]

        # Run train

        lmart_model = L2Ranker(ranklib_location, l2r_params)
    #     lmart_model = L2Ranker(ranklib_location, l2r_params, norm_params)


# In[5]:


trec_eval_command = '../../eval/trec_eval'


params_list = [
    train_data_file,
    val_data_file,
    fold_dir,
    lmart_model,
    qrels_val_file
]

def eval_hpo(params_list, hpo_params):
    
    train_data_file = params_list[0]
    val_data_file = params_list[1]
    fold_dir = params_list[2]
    lmart_model = params_list[3]
    qrels_val_file = params_list[4]
    
    hpo_params_suffix = 'nl' + str(hpo_params['n_leaves']) + 'lr' + str(hpo_params['learning_rate']) + 'nt' + str(hpo_params['n_trees'])
    
    save_model_file = fold_dir + dataset_fold + '_lmart_' + hpo_params_suffix + '_model' 
    
    lmart_model.train(train_data_file, save_model_file, hpo_params)
    run_file = fold_dir + 'run_' + dataset_fold + '_lmart_' + hpo_params_suffix
    
#   lmart_model.gen_run_file(test_data_file, run_file)

    lmart_model.gen_run_file(val_data_file, run_file)
    
#     print(qrels_val_file)
#     print(run_file)
    eval_results = eval(trec_eval_command, qrels_val_file, run_file)
    eval_results.update(lmart_model.hpo_config)
    eval_results['lmart_model'] = lmart_model
    return eval_results


# In[6]:


def find_best_dev_model(best_model_params_file, random_iterations = 5000):
#     random_search = 'yes'
    
    if random_search == 'yes':
        ## Heavy random search
        brange = np.arange(0.1,1,0.05)
        krange = np.arange(0.1,4,0.1)
        N_range = np.arange(5,500,1) # num of docs
        M_range = np.arange(5,500,1) # num of terms
        lamb_range = np.arange(0,1,0.1) # weights of original query

        ## Light random search
#         brange = [0.2]
#         krange = [0.8]
#         N_range = np.arange(1,50,2)
#         M_range = np.arange(1,50,2)
#         lamb_range = np.arange(0,1,0.2)
        
        h_param_ranges = [brange, krange, N_range, M_range, lamb_range]
        params = get_random_params(h_param_ranges, random_iterations)

    else:
        brange = [0.2]
        krange = [0.8]
        N_range = [11]
        M_range = [10]
        lamb_range = [0.5]
       
        params = [[round(b,3), round(k,3), round(N,3), round(M,3), round(Lambda,3)] 
                  for b in brange for k in krange for N in N_range for M in M_range for Lambda in lamb_range]
   
    print('# Params: ', len(params)) 
    pool_size = 20
#     print(len(params))
    pool = multiprocessing.Pool(processes=pool_size,
                                initializer=start_process,
                                )

#     pool_outputs = pool.map(bm25_computing, params)
    

    pool_outputs = pool.map_async(bm25_computing, params)
#     pool_outputs.get()
    ###

    
    ##
    
    
    pool.close() # no more tasks
    while (True):
        if (pool_outputs.ready()): break
        remaining = pool_outputs._number_left
#         remaining2 = remaining1
#         remaining1 = pool_outputs._number_left
        if remaining%10 == 0:
            print("Waiting for", remaining, "tasks to complete...")
            time.sleep(2)
        
      
    pool.join()  # wrap up current tasks
    pool_outputs.get()
    params_file = './best_ir_model/' + dataset_name_ext + '_' + 'bm25_rm3_' + split + '_hparams.pickle'
    pickle.dump(pool_outputs.get(), open(params_file, "wb" ) )
    print('Total parameters: ' + str(len(pool_outputs.get())))
    best_model_params = max(pool_outputs.get(), key=lambda x: x[5])
    
    best_model_dict = {
        'b': best_model_params[0],
        'k': best_model_params[1],
        'N': best_model_params[2],
        'M': best_model_params[3],
        'Lambda': best_model_params[4],
        'random_iterations': random_iterations,
        'map': best_model_params[5],
        'p_20': best_model_params[6],
        'ndcg_20': best_model_params[7]
        
    }
    best_model_dict = {k:str(v) for k, v in best_model_dict.items()} # everything to string
    
    with open(best_model_params_file, 'wt') as best_model_f:
        json.dump(best_model_dict, best_model_f)


# In[7]:


def start_process():
    print( 'Starting', multiprocessing.current_process().name)


# In[8]:


def eval_multi_hpo(params_list, hpo_params_list, pool_size):
   
    eval_hpo_partial = partial(eval_hpo, params_list)

    pool = multiprocessing.Pool(processes=pool_size,
                                initializer=start_process,
                                )

    pool_outputs = pool.map_async(eval_hpo_partial, hpo_params_list)
    pool.close() # no more tasks
    pool.join()  # wrap up current tasks
    print('Total parameters: ' + str(len(pool_outputs.get())))
    return pool_outputs.get()


# In[9]:


hpo_method = 'rs'

random_iterations = 20 # these are outside parameters

nleaves_range = np.arange(1,51,1)
lrate_range = np.arange(0.1,1,0.1)
ntrees_range = np.arange(1,51,1)

h_param_ranges = [nleaves_range, lrate_range, ntrees_range]

if hpo_method == 'rs':
    h_params = get_random_params(h_param_ranges, random_iterations)
elif hpo_method == 'gs':
    h_params = get_grid_search_params(h_param_ranges)

hpo_params_list = [{'n_leaves': x[0], 'learning_rate': x[1], 'n_trees': x[2]} for x in h_params]

print(len(hpo_params_list))


# In[10]:


# hpo_params_list = [hpo_params]
pool_size = 5

hpo_results = eval_multi_hpo(params_list, hpo_params_list, pool_size)


# In[11]:


best_model_hpo = max(hpo_results, key=lambda x: x['map'])
best_model_hpo['random_iterations'] = random_iterations
best_model_hpo['data_split'] = 'validation'
best_lmart_model = best_model_hpo.pop('lmart_model')
best_model_hpo = {k: str(v) for k, v in best_model_hpo.items()}

hpo_results_file = fold_dir +  dataset + '_' + fold + '_' + l2r_model + '_hpo_results.pickle'
best_model_hpo_file = fold_dir +  'best_' + dataset + '_' + fold + '_' + l2r_model + '_hparams.json'

pickle.dump(hpo_results, open(hpo_results_file, "wb" ) )


# In[12]:


hpo_results


# In[13]:


qrels_test_file = fold_dir + dataset + '_' + 'test_' + fold + '_qrels'

run_file = fold_dir + 'run_' + dataset + '_' + fold + '_test_lmart_best'

best_lmart_model.gen_run_file(test_data_file, run_file)

eval_results = eval(trec_eval_command, qrels_test_file, run_file)


# In[14]:


eval_results['data_split'] = 'test'
eval_results

best_model_hpo['test'] = eval_results

with open(best_model_hpo_file, 'wt') as best_model_f:
    json.dump(best_model_hpo, best_model_f)

