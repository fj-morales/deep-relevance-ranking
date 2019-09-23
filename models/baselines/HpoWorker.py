import random
import itertools
import os

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


from hpbandster.examples.commons import MyWorker
import copy

from eval_utils import *



def get_train_budget_data_file(budget, qid_list, train_data_file):
    # Budget is percentage of training data: 
    # min_budget = 10%
    # max_budget = 100%
    if (int(budget) <= len(qid_list)):
        len_queries = len(qid_list)
    
#         print('total budget:', len_queries)
#         print('allocated budget:', budget)
        train_budget_queries_file = train_data_file + '_budget' + str(budget)
        if not os.path.exists(train_budget_queries_file):
            #                         print('queries lenght\n:',len_queries)
            with open(train_data_file, 'rt') as f_in:
                with open(train_budget_queries_file, 'wt') as budget_file_out:
                    
                    for query_feature in f_in:                        

                        qid = query_feature.split()[1].split(':')[1]
#                         print(qid)
                        if qid in qid_list[0:budget]:
                            
                            budget_file_out.write(query_feature)
                    return train_budget_queries_file
        else:
#             print("File already exists")
            return train_budget_queries_file                
    else:
        print('Budget is outside the limits (10% < b < 100%): ', budget)
        return 'THIS IS NOT WORKING'

# def get_train_budget_data_file(budget, qid_list, train_data_file):
#     # Budget is percentage of training data: 
#     # min_budget = 10%
#     # max_budget = 100%
#     if (int(budget) <= 100 or int(budget) >= 10):
#         len_queries = len(qid_list)
        
#         budgeted_queries = round(len_queries * (budget / 100))
#         print('total budget:', len_queries)
#         print('allocated budget:', budgeted_queries)
#         train_budget_queries_file = train_data_file + '_budget' + str(budget)
#         if not os.path.exists(train_budget_queries_file):
#             #                         print('queries lenght\n:',len_queries)
#             with open(train_data_file, 'rt') as f_in:
#                 with open(train_budget_queries_file, 'wt') as budget_file_out:
                    
#                     for query_feature in f_in:                        

#                         qid = query_feature.split()[1].split(':')[1]
# #                         print(qid)
#                         if qid in qid_list[0:budgeted_queries]:
                            
#                             budget_file_out.write(query_feature)
#                     return train_budget_queries_file
#         else:
#             print("File already exists")
#             return train_budget_queries_file                
#     else:
#         print('Budget is outside the limits (10% < b < 100%): ', budget)
#         return 'THIS IS NOT WORKING'    
    

class HpoWorker(Worker):
    def __init__(self, model_instance, save_model_prefix, run_file_prefix, budget_train_features_file, qid_list, trec_eval_command, 
                 qrels_val_file, tickets, **kwargs):
            super().__init__(**kwargs)
            self.save_model_prefix = save_model_prefix
            self.run_file_prefix = run_file_prefix
            self.model = copy.deepcopy(model_instance)
            self.budget_train_features_file = budget_train_features_file
            self.qid_list = qid_list
            self.trec_eval_command = trec_eval_command
            self.qrels_val_file = qrels_val_file
            self.tickets = tickets
    def compute(self, config, budget, *args, **kwargs):
    
            """
            Simple example for a compute function using a feed forward network.
            It is trained on the MNIST dataset.
            The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
            """
            
            budget = int(budget)
            
            my_ticket = None
            
            while my_ticket is None:

                try:
                    this_key = list(self.tickets.keys())[0]
                    my_ticket = self.tickets.pop(this_key)

                except:
                    pass
            
            
#             print('My ticket: ', my_ticket, '\n')
#             print('Tickets left: ', len(self.tickets), '\n')
            
            
            
            # Train model with config parameters
                
            #     pre_run_file = workdir + 'pre_run_' + dataset + l2r_model
            
            config['learning_rate'] = round(config['learning_rate'],2)
            n_l = config['n_leaves']
            l_r = config['learning_rate']
            n_t = config['n_trees']
            
            config_suffix = '_id' + str(my_ticket) + '_budget' + str(budget) + '_leaves' + str(n_l) + '_lr' + str(l_r) + '_n' + str(n_t)
            
            self.save_model_file = self.save_model_prefix + config_suffix
            
            self.run_val_file = self.run_file_prefix + config_suffix
            
#             print('Type class of budget variable: ', type(budget))
#             print(self.qid_list)
            
            budget_train_features_file = get_train_budget_data_file(budget, self.qid_list, self.budget_train_features_file)
            
#             lmart_model = L2Ranker(ranklib_location, l2r_params, norm_params)
#             print('budget_file:', budget_train_features_file)
        
            self.model.train(budget_train_features_file, self.save_model_file, config)
        
            val_data_file = self.model.params[1]
            
            self.model.gen_run_file(val_data_file, self.run_val_file)
            
            # Evaluate Model
            
            val_results = eval(self.trec_eval_command, self.qrels_val_file, self.run_val_file)
            
            val_map = float(val_results['map'])
#             print('Aqui pedi imprimir map\n', val_map, config, '\n')
        

            #import IPython; IPython.embed()
            return ({
                    'loss': 1 - val_map, # remember: HpBandSter always minimizes!
#                     'info': {'model': lmart_model}
                    'info': val_results
#                     'info': 'oh my god' 
            })


    @staticmethod
    def get_configspace():
            """
            It builds the configuration space with the needed hyperparameters.
            It is easily possible to implement different types of hyperparameters.
            Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
            :return: ConfigurationsSpace-Object
            """
            cs = CS.ConfigurationSpace()

            n_leaves = CSH.UniformIntegerHyperparameter('n_leaves', lower=1, upper=100, default_value=10, q=5, log=False)
            learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=0.01, upper=0.5, default_value=0.1, q=0.01, log=False)
            n_trees = CSH.UniformIntegerHyperparameter('n_trees', lower=100, upper=2000, default_value=1000, q=50 ,log=False)
            
#             n_leaves = CSH.UniformIntegerHyperparameter('n_leaves', lower=10, upper=11, default_value=10, log=False)
#             learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=0.1, upper=0.2, default_value=0.1, q=0.1, log=False)
#             n_trees = CSH.UniformIntegerHyperparameter('n_trees', lower=1000, upper=1001, default_value=1000, q=1, log=False)

            
            cs.add_hyperparameters([n_leaves, learning_rate, n_trees])

            return cs

if __name__ == "__main__":
    worker = KerasWorker(run_id='0')
    cs = worker.get_configspace()

    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(config=config, budget=1, working_directory='.')
    print(res)