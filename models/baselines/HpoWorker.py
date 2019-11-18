import random
import itertools
import os

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

from ir_utils import *

# HPO server and stuff

# import logging
# logging.basicConfig(level=logging.WARNING)

import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres


from hpbandster.examples.commons import MyWorker
import copy

from eval_utils import *
import numpy as np

import numpy as np

# model

from ir_lmart import *

import logging
logging.basicConfig(level=logging.WARNING)

# def get_train_budget_data_file(budget, qid_list, train_data_file):
#     # Budget is percentage of training data: 
#     # min_budget = 10%
#     # max_budget = 100%
#     if (int(budget) <= len(qid_list)):
#         len_queries = len(qid_list)
    
# #         print('total budget:', len_queries)
# #         print('allocated budget:', budget)
#         train_budget_queries_file = train_data_file + '_budget' + str(budget)
#         if not os.path.exists(train_budget_queries_file):
#             #                         print('queries lenght\n:',len_queries)
#             with open(train_data_file, 'rt') as f_in:
#                 with open(train_budget_queries_file, 'wt') as budget_file_out:
                    
#                     for query_feature in f_in:                        

#                         qid = query_feature.split()[1].split(':')[1]
# #                         print(qid)
#                         if qid in qid_list[0:budget]:
                            
#                             budget_file_out.write(query_feature)
#                     return train_budget_queries_file
#         else:
# #             print("File already exists")
#             return train_budget_queries_file                
#     else:
#         print('Budget is outside the limits (10% < b < 100%): ', budget)
#         return 'THIS IS NOT WORKING'

def get_train_budget_data_file(budget, qid_list, train_data_file):
    # Budget is percentage of training data: 
    # min_budget = 10%
    # max_budget = 100%
    if (int(budget) <= 100 or int(budget) >= 10):
        len_queries = len(qid_list)
        
        budgeted_queries = round(len_queries * (budget / 100))
        print('total budget:', len_queries)
        print('allocated budget:', budgeted_queries)
        train_budget_queries_file = train_data_file + '_budget' + str(budget)
        if not os.path.exists(train_budget_queries_file):
            #                         print('queries lenght\n:',len_queries)
            with open(train_data_file, 'rt') as f_in:
                with open(train_budget_queries_file, 'wt') as budget_file_out:
                    
                    for query_feature in f_in:                        

                        qid = query_feature.split()[1].split(':')[1]
#                         print(qid)
                        if qid in qid_list[0:budgeted_queries]:
                            
                            budget_file_out.write(query_feature)
                    return train_budget_queries_file
        else:
            print("File already exists")
            return train_budget_queries_file                
    else:
        print('Budget is outside the limits (10% < b < 100%): ', budget)
        return 'THIS IS NOT WORKING'   
    
def compute_one_fold(budget, config, tickets, save_model_prefix, run_file_prefix, run_test_file_prefix, model_instance, 
                     qrels_val_file, qid_list, train_data_file, trec_eval_command, ranklib_location, *args, **kwargs):
    
            """
            Simple example for a compute function using a feed forward network.
            It is trained on the MNIST dataset.
            The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
            """
            
            budget = int(budget)
            
            my_ticket = None
            
            while my_ticket is None:

                try:
                    this_key = list(tickets.keys())[0]
                    my_ticket = tickets.pop(this_key)
                    print('Run ID:' , my_ticket, '\n')
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
            
            config_suffix = 'id' + str(my_ticket) + '_budget' + str(budget) + '_leaves' + str(n_l) + '_lr' + str(l_r) + '_n' + str(n_t)
            
            save_model_file = save_model_prefix + config_suffix
            
            run_val_file = run_file_prefix + config_suffix
            run_test_file = run_test_file_prefix + config_suffix
            
#             print('Type class of budget variable: ', type(budget))
#             print(self.qid_list)
            

            budget_train_features_file = get_train_budget_data_file(budget, qid_list, train_data_file)
            
#             lmart_model = L2Ranker(ranklib_location, l2r_params, norm_params)
#             print('budget_file:', budget_train_features_file)
        
            model = copy.deepcopy(model_instance)
            model.train(budget_train_features_file, save_model_file, config)
        
            val_data_file = model.params[1]
            test_data_file = model.test_data_file
            
            model.gen_run_file(val_data_file, run_val_file)
#             model.gen_run_file(test_data_file, run_test_file)
            
            # Evaluate Model
            qrels_test_file = qrels_val_file.replace('dev', 'test')
            
            val_results = eval(trec_eval_command, qrels_val_file, run_val_file)
            print('Validation results: ', val_results)
#             test_results = eval(trec_eval_command, qrels_test_file, run_test_file)
            
            val_results['model_file'] = save_model_file
#             val_results['test_results'] = test_results
            
            val_map = float(val_results['map'])
#             print('Aqui pedi imprimir map\n', val_map, config, '\n')
        
            print('Metric: ', val_results, '\n')

            #import IPython; IPython.embed()
            return ({
                    'metric': val_map,
                    'info': val_results
            })


class HpoWorker(Worker):
    def __init__(self, dataset, workdir, ranklib_location, norm_params, ranker_type, trec_eval_command, metric2t, tickets, **kwargs):
            super().__init__(**kwargs)
            self.dataset = dataset
            self.workdir = workdir
            self.ranklib_location = ranklib_location
            self.trec_eval_command = trec_eval_command
            self.tickets = tickets  
            self.ranker_type = ranker_type
            self.metric2t = metric2t
            self.norm_params = norm_params
            
#             self.save_model_prefix = save_model_prefix
#             self.run_file_prefix = run_file_prefix
#             self.model = copy.deepcopy(model_instance)
#             self.budget_train_features_file = budget_train_features_file
#             self.qid_list = qid_list
#             self.trec_eval_command = trec_eval_command
#             self.qrels_val_file = qrels_val_file
            
    def compute(self, config, budget, *args, **kwargs):
    
            """
            Simple example for a compute function using a feed forward network.
            It is trained on the MNIST dataset.
            The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
            """

            if self.dataset == 'bioasq':
                folds = ['']
            elif self.dataset == 'robust':
                folds = ['1','2','3','4','5']

            cv_results_dict = {}
            
            for fold in folds:

                if self.dataset == 'bioasq':
                    fold_dir = self.workdir
                    dataset_fold = self.dataset 
                    train_queries_file = '../../bioasq_data/bioasq.' + 'train' + '.json'
                    train_data_file = fold_dir + self.dataset + '_train' + '_features'
                    val_data_file = fold_dir + self.dataset + '_dev' + '_features'
                    test_data_file = fold_dir + self.dataset + '_test' +  '_features'
                    qrels_val_file = fold_dir + self.dataset + '_dev' + '_qrels'
                else:
                    fold_dir = self.workdir + 's' + fold + '/'
                    dataset_fold = self.dataset + '_s' + fold
                    train_queries_file = '../../robust04_data/split_' + fold + '/rob04.' +  'train' + '.s' + fold + '.json'
                    train_data_file = fold_dir + self.dataset + '_train' + '_s' + fold + '_features'
                    val_data_file = fold_dir + self.dataset + '_dev' + '_s' + fold +  '_features'
                    test_data_file = fold_dir + self.dataset + '_test' + '_s' + fold +  '_features'
                    qrels_val_file = fold_dir + self.dataset + '_dev' + '_s' + fold + '_qrels'


                if self.ranker_type == '6':
                    l2r_model = '_lmart_'
                    
                confdir = './' + self.dataset + '_config/'
                enabled_features_file = confdir + self.dataset + l2r_model + 'enabled_features' 

                l2r_params = [
                    '-validate',
                    val_data_file,
                    '-ranker',
                    self.ranker_type,
#                     '9', #Ranknet: 1; Linear L2: 9; LambdaMART: 6
#                     '-L2',
#                     '0',
                    
                    '-metric2t',
                    self.metric2t,
                    '-feature',
                    enabled_features_file
                ]

                # Run train
                lmart_model = L2Ranker(self.ranklib_location, l2r_params, test_data_file, self.norm_params)

                save_model_prefix = fold_dir + dataset_fold + l2r_model

                run_file_prefix = fold_dir + 'run_' + dataset_fold + l2r_model
                run_test_file_prefix = fold_dir + 'run_tests_' + dataset_fold + l2r_model

                # Preparing budgeted train features data 
                query_list = load_queries(train_queries_file)
                qid_list = [q['id'] for q in query_list] 
                len_queries = len(qid_list) 
                

                train_features_file =  fold_dir + self.dataset + '_' + 'train' + '_features'


#                 budget_train_features_file = train_data_file


                # Compute results for one fold
                one_fold_results = compute_one_fold(budget, config, self.tickets, save_model_prefix, run_file_prefix, run_test_file_prefix, lmart_model, 
                                                     qrels_val_file, qid_list, train_data_file, self.trec_eval_command, self.ranklib_location)

                cv_results_dict['s' + fold] = one_fold_results

            cv_mean_metric = round(np.mean([value['metric'] for key,value in cv_results_dict.items()]), 8)
            cv_std_metric = round(np.std([value['metric'] for key,value in cv_results_dict.items()]), 8)
            
#             cv_mean_metric_test = round(np.mean([float(value['info']['test_results']['map']) for key,value in cv_results_dict.items()]), 8)
#             cv_std_metric_test = round(np.std( [float(value['info']['test_results']['map']) for key,value in cv_results_dict.items()]), 8)
            
            cv_results_dict['mean_metric'] = cv_mean_metric
            cv_results_dict['std_metric'] = cv_std_metric
            
#             cv_results_dict['mean_metric_test'] = cv_mean_metric_test
#             cv_results_dict['std_metric_test'] = cv_std_metric_test
            
            
            return ({
                'loss': 1 - cv_mean_metric, # remember: HpBandSter always minimizes!
                'info': cv_results_dict
            })                
            
    @staticmethod
    def get_configspace(default_config,test_mode,leaf,lr,tree):
            """
            It builds the configuration space with the needed hyperparameters.
            It is easily possible to implement different types of hyperparameters.
            Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
            :return: ConfigurationsSpace-Object
            """
            cs = CS.ConfigurationSpace()
            
            # This is the good one!
            n_leaves = CSH.UniformIntegerHyperparameter('n_leaves', lower=5, upper=100, default_value=10, q=5, log=False)
            learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=0.01, upper=0.5, default_value=0.1, q=0.01, log=False)
            n_trees = CSH.UniformIntegerHyperparameter('n_trees', lower=100, upper=2000, default_value=1000, q=50 ,log=False)
            
#             # Increase spectrum: insane values: 5 328 000 000 total configs!!
#             n_leaves = CSH.UniformIntegerHyperparameter('n_leaves', lower=1, upper=200, default_value=10, q=1, log=False)
#             learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=0.001, upper=0.9, default_value=0.1, q=0.001, log=False)
#             n_trees = CSH.UniformIntegerHyperparameter('n_trees', lower=10, upper=3000, default_value=1000, q=1 ,log=False)
            
#             n_leaves = CSH.UniformIntegerHyperparameter('n_leaves', lower=1, upper=2, default_value=2, log=False)
#             learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=0.1, upper=0.2, default_value=0.1, q=0.1, log=False)
#             n_trees = CSH.UniformIntegerHyperparameter('n_trees', lower=1, upper=2, default_value=1, q=1, log=False)

            if default_config:
                n_leaves = CSH.OrdinalHyperparameter('n_leaves', sequence=[10])
                learning_rate = CSH.OrdinalHyperparameter('learning_rate', sequence=[0.1])
                n_trees = CSH.OrdinalHyperparameter('n_trees', sequence=[1000])
            
            if test_mode:
                n_leaves = CSH.OrdinalHyperparameter('n_leaves', sequence=[leaf])
                learning_rate = CSH.OrdinalHyperparameter('learning_rate', sequence=[lr])
                n_trees = CSH.OrdinalHyperparameter('n_trees', sequence=[tree])
            
            
            cs.add_hyperparameters([n_leaves, learning_rate, n_trees])

            return cs

if __name__ == "__main__":
    worker = KerasWorker(run_id='0')
    cs = worker.get_configspace()

    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(config=config, budget=1, working_directory='.')
    print(res)