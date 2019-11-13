import sys
import subprocess
from eval_utils import *

# Functions
def generate_run_file(pre_run_file, run_file):
    
    with open(pre_run_file, 'rt') as input_f:
        pre_run = input_f.readlines()
        print(type(pre_run))
    with open(run_file, 'wt') as out_f:
        for line in pre_run:
            out_f.write(line.replace('docid=','').replace('indri', 'lambdaMART'))


def gen_run_file(ranklib_location, normalization, save_model_file, test_data_file, run_file):
# Works also for testing

    print("Run file: ",run_file)
    ranker_command = ['java', '-jar', ranklib_location + 'RankLib-2.12.jar']
    pre_run_file = run_file.replace('run_', 'pre_run_', 1)
    toolkit_parameters = [
        *ranker_command, # * to unpack list elements
        '-load',
        save_model_file,
        *normalization,
        '-rank',
        test_data_file,
        '-indri',
        pre_run_file     
        ]             

#     print(toolkit_parameters)
    proc = subprocess.Popen(toolkit_parameters,stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
    (out, err)= proc.communicate()
    #         print(out.decode('utf-8').splitlines())
    #         print(out)
    #         print(err)

    generate_run_file(pre_run_file, run_file)

    
    
def test_model(workdir, dataset, ranklib_location, trec_eval_command, normalization, res):
    
    if dataset == 'bioasq':
        folds = ['']
    elif dataset == 'robust':
        folds = ['1','2','3','4','5']

    results_dict = {}

    for fold in folds:

        if dataset == 'bioasq':
            fold_dir = workdir
            dataset_fold = dataset 
            test_data_file = fold_dir + dataset + '_test' +  '_features'
            qrels_test_file = fold_dir + dataset + '_test' + '_qrels'
            test_data_file = fold_dir + dataset_fold + '_test' + '_features'
        else:
            fold_dir = workdir + 's' + fold + '/'
            dataset_fold = dataset + '_s' + fold
            test_data_file = fold_dir + dataset + '_test' + '_s' + fold +  '_features'
            qrels_test_file = fold_dir + dataset + '_test' + '_s' + fold + '_qrels'
            test_data_file = fold_dir + dataset + '_test' + '_s' + fold  + '_features'


        ## Evaluate on test
        
        config_results = res.get_runs_by_id(res.get_incumbent_id())
        
        best_model = config_results[0].info['s' + fold]['info']['model_file']
        
        suffix = best_model.split('/')[-1].split('_')[-3:]
        
        run_test_file = fold_dir + 'run_' + dataset_fold + '_best_lmart_test_' + str(suffix[0]) + '_' + str(suffix[1]) + '_' + str(suffix[2]) 
        
        gen_run_file(ranklib_location, normalization, best_model, test_data_file, run_test_file)
    
        print(trec_eval_command, qrels_test_file, run_test_file)
        test_results = eval(trec_eval_command, qrels_test_file, run_test_file)
        
        results_dict['s' + fold] = test_results
        
    return results_dict
