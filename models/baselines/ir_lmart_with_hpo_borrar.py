#!/usr/bin/env python
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

# REMOVE!!
from ir_utils import *

# HPO

from hpo import *

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
        self.log_file = self.params[-1:][0] + '.log'
        self.ranker_command = ['java', '-jar', ranklib_location + 'RankLib-2.12.jar']
        self.normalization = normalization
        self.save_model_file = ''
        
#     def build(self, ir_tool_params):
    def train(self, train_data_file, save_model_file, config):
        self.save_model_file = save_model_file
        toolkit_parameters = [
                                *self.ranker_command, # * to unpack list elements
                                '-train',
                                train_data_file,
                                *self.normalization,
                                *self.params,
                                '-leaf', 
                                str(config['n_leaves']),
                                '-shrinkage',
                                str(config['learning_rate']),
                                '-tree', # Oner regression tree per boosted iteration
                                str(config['n_trees']),
                                '-save',
                                self.save_model_file   
                            ] 
        
#         print(toolkit_parameters)
        with open(self.log_file, 'wt') as rf:
            proc = subprocess.Popen(toolkit_parameters,stdin=subprocess.PIPE, stdout=rf, stderr=subprocess.STDOUT, shell=False)
#         proc = subprocess.Popen(toolkit_parameters,stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
            
        (out, err)= proc.communicate()
#         print(out.decode('utf-8').splitlines())
#         print(out)
#         print(err)
        print('Model saved: ', self.save_model_file)
            
  

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
#         print(err)

        
        generate_run_file(pre_run_file, run_file)
        

class fakeParser:
    def __init__(self):
        self.min_budget = 2 
        self.max_budget = 4
        self.n_iterations = 4 
        self.n_workers =4
        
# In[6]:


# In[3]:


# Main
if __name__ == "__main__":
    
    # Options and variables
    
    parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
    parser.add_argument('--min_budget',   type=float, help='Minimum budget used during the optimization.',    default=2)
    parser.add_argument('--max_budget',   type=float, help='Maximum budget used during the optimization.',    default=4)
    parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=500)
    parser.add_argument('--n_workers', type=int,   help='Number of workers to run in parallel.', default=5)

#     args=parser.parse_args()
    args = fakeParser()
    
#     dataset = sys.argv[1] # 'bioasq'
#     workdir = './' + dataset + '_dir/'
#     data_split = sys.argv[2] # 'test'

    dataset = 'bioasq'
    workdir = './' + dataset + '_dir/'
    data_split =  'train'
    k_fold = 's1' 
    ranklib_location = '../../../ranklib/'
    
#     train_data_file = './bioasq_dir/bioasq.trai_features_reduced'
#     val_data_file = './bioasq_dir/bioasq.dev_features_reduced'
#     test_data_file = './bioasq_dir/bioasq.test_features_reduced'
    
    train_data_file = './bioasq_dir/bioasq.trai_features'
    val_data_file = './bioasq_dir/bioasq.dev_features'
    test_data_file = './bioasq_dir/bioasq.test_features'
    
    l2r_model = '_lmart_'
    


    
    enabled_features_file = workdir + dataset + l2r_model + 'enabled_features'
    
#     print(enabled_features_file)
    # Train L2R model: LambdaMART
    # Parameters 
    
#     n_leaves = '10'
#     learning_rate = '0.1'
#     n_trees = '1000'
#     hpo_params = [n_leaves, learning_rate, n_trees]
    
    
    
    metric2t = 'MAP' # 'MAP, NDCG@k, DCG@k, P@k, RR@k, ERR@k (default=ERR@10)'
    
    ranker_type = '6' # LambdaMART
    
    # normalization: Feature Engineering?
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
    
#     lmart_model = L2Ranker(ranklib_location, l2r_params)
#     lmart_model = L2Ranker(ranklib_location, l2r_params, norm_params)
    


# In[7]:


#     lmart_model.train(train_data_file, hpo_params)


# In[8]:


#     lmart_model.gen_run_file(test_data_file, run_file)


# In[9]:


trec_eval_command = '../../eval/trec_eval'
qrels_val_file = './bioasq_dir/bioasq.dev_qrels'
#     eval(trec_eval_command, qrels_file, './run_l2linear')


# In[ ]:





# In[10]:


#     # HPO 

    
    
# #     worker = HpoWorker(run_id='0')
#     cs = worker.get_configspace()

#     config = cs.sample_configuration().get_dictionary()
    
        
# #     pre_run_file = workdir + 'pre_run_' + dataset + l2r_model
    
#     run_file = workdir + 'run_' + dataset + l2r_model
    
#     print(config)
#     res = worker.compute(config=config)
#     print(res['loss'])


# In[11]:


# Start a nameserver (see example_3)
NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
NS.start()


# In[12]:


workers=[]
for i in range(args.n_workers):
    worker = HpoWorker( nameserver='127.0.0.1',run_id='example1', id=i)
    worker.run(background=True)
    workers.append(worker)


# In[13]:


# Run an optimizer (see example_2)
bohb = BOHB(  configspace = worker.get_configspace(),
                      run_id = 'example1', 
                      min_budget = args.min_budget, max_budget = args.max_budget
               )
res = bohb.run(n_iterations = args.n_iterations, min_n_workers = args.n_workers)


# In[14]:


# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
bohb.shutdown(shutdown_workers=True)
NS.shutdown()


# In[15]:


# Step 5: Analysis
# Each optimizer returns a hpbandster.core.result.Result object.
# It holds informations about the optimization run like the incumbent (=best) configuration.
# For further details about the Result object, see its documentation.
# Here we simply print out the best config and some statistics about the performed runs.
id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()


# In[16]:


all_runs = res.get_all_runs()


# In[17]:


# In[4]:


print('Best found configuration:', id2config[incumbent]['config'])
print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
print('A total of %i runs where executed.' % len(res.get_all_runs()))
print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in all_runs])/args.max_budget))
print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in all_runs])/args.max_budget))
print('The run took  %.1f seconds to complete.'%(all_runs[-1].time_stamps['finished'] - all_runs[0].time_stamps['started']))


# In[18]:


#     qrels_test_file = './bioasq_dir/bioasq.test_qrels'
#     run_val_file = './this.file'
#     lmart_model = res['info']
#     lmart_model.gen_run_file(test_data_file, run_val_file)
#     eval(trec_eval_command, qrels_test_file, run_val_file)


# In[10]:


id2config

