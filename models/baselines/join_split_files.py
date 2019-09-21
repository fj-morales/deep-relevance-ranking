#!/usr/bin/env python
# coding: utf-8

# In[14]:


# %load join_split_files.py
#!/usr/bin/env python

# In[2]:


import json
import os
from utils import load_queries

def get_filenames(data_dir, word_filter):
    filenames = [os.path.join(root, name)
             for root, dirs, files in os.walk(data_dir)
             if ("test_mp") not in root
             for name in files
#              if (name.endswith(suffix)  & ('split_' in root))
             if (word_filter in name)
           ]
    return filenames

def join_files():

    data_dir = '../../robust04_data/split_1/'
    json_filter = 'json'

    name_list = get_filenames(data_dir, json_filter)
    len(name_list)
#     print(name_list)

    all_queries = []
    for i in name_list:
        all_queries = all_queries + load_queries(i)

    all_queries_dict = {'questions': all_queries}

    workdir = './robust_dir/'
    all_queries_dict_file = workdir + 'robust_all_queries.json'

    with open(all_queries_dict_file,  'wt') as out_f:
        json.dump(all_queries_dict, out_f, indent = 4)

    run_dir = './robust_dir/s1/'
    run_filter = 'run_bm25_robust'
    run_list = get_filenames(run_dir, run_filter)
    len(run_list)
#     print(run_list)

    all_runs = []
    for i in run_list:
        with open(i, 'rt') as in_file:
            run_lines = in_file.readlines()
        all_runs = all_runs + run_lines
    len(all_runs)

    all_run_file = workdir + 'run_bm25_robust_all'
    with open(all_run_file,  'wt') as out_f:
        out_f.write("".join(all_runs))

    # Join qrels

    qrel_filter = '_qrels'
    qrel_list = get_filenames(run_dir, qrel_filter)
    len(qrel_list)
#     print(qrel_list)

    all_qrels = []
    for i in qrel_list:
        with open(i, 'rt') as in_file:
            qrel_lines = in_file.readlines()
        all_qrels = all_qrels + qrel_lines
    len(all_qrels)

    all_qrels_file = workdir + 'robust_qrels_all'
    with open(all_qrels_file,  'wt') as out_f:
        out_f.write("".join(all_qrels))
        
        
    # Join query_trec

    trec_query_filter = '_trec_query'
    trec_query_list = get_filenames(run_dir, trec_query_filter)
    len(trec_query_list)
#     print(qrel_list)

    all_trec_query = []
    for i in trec_query_list:
        with open(i, 'rt') as in_file:
            trec_lines = in_file.readlines()
        all_trec_query = all_trec_query + trec_lines[1:-1]
        
        
    all_trec_query_file = workdir + 'robust_all_trec_query'
    with open(all_trec_query_file,  'wt') as out_f:
        out_f.write('<parameters>\n')
        out_f.write("".join(all_trec_query))
        out_f.write('</parameters>')
        
    return [all_queries_dict_file, all_run_file, all_qrels_file, all_trec_query_file]

