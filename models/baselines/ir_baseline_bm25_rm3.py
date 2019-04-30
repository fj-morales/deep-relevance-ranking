
# coding: utf-8

# ## BM25 + RM3 with Anserini

# In[129]:


import pickle
import json
import gzip
import os
import subprocess
import numpy as np
import multiprocessing
import re 
import shutil
from itertools import islice
import random

import os
import sys
import uuid
import datetime
import time


# In[133]:


def remove_sc(text):
##     text = re.sub('[.,?;*!%^&_+():-\[\]{}]', '', text.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip())
##     text = re.sub('[\[\]{}.,?;*!%^&_+():-]', '', text.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip()) # DeepPaper method
    text = re.sub(r'[^\w\s]',' ',text) # My method
##     text = text.rstrip('.?')
    return text


# In[134]:


def get_pickle_docs(pickle_filename):
    # Pickle to Trectext converter
    with open(pickle_filename, 'rb') as f_in:
        data = pickle.load(f_in)
        if not os.path.exists(baseline_files):
            os.makedirs(baseline_files)
        
        if os.path.exists(corpus_files):
            shutil.rmtree(corpus_files)
            os.makedirs(corpus_files)
        else:
            os.makedirs(corpus_files)

            
        docs = {}
        for key, value in data.items():
            if "pmid" in value.keys():
                doc_code = value.pop('pmid')
            else:
                doc_code = key
                
# Uncomment                 
#             doc = '<DOC>\n' + \
#                   '<DOCNO>' + doc_code + '</DOCNO>\n' + \
#                   '<TITLE>' + value.pop('title') + '</TITLE>\n' + \
#                   '<TEXT>' + value.pop('abstractText') + '</TEXT>\n' + \
#                   '</DOC>\n'
            
            doc = '<DOC>\n' +                   '<DOCNO>' + doc_code + '</DOCNO>\n' +                   '<TITLE>' + remove_sc(value.pop('title')) + '</TITLE>\n' +                   '<TEXT>' + remove_sc(value.pop('abstractText')) + '</TEXT>\n' +                   '</DOC>\n'
            docs[doc_code] = doc
        return docs


# In[135]:


def to_trecfile(docs, filename, compression = 'yes'):
    # Pickle to Trectext converter
    doc_list = []
    if compression == 'yes':
        with gzip.open(filename,'wt', encoding='utf-8') as f_out:
            docus = {}
            for key, value in docs.items():
                f_out.write(value)
    else:
        with open(filename,'wt', encoding='utf-8') as f_out:
            docus = {}
            for key, value in docs.items():
                f_out.write(value)


# In[136]:


# Build corpus index with Anserini
def build_index(index_input, index_loc, log_file):
    if not os.path.exists(index_loc):
            os.makedirs(index_loc) 
#     index_loc_param = '--indexPath=' + index_loc

    anserini_index = anserini_loc + 'target/appassembler/bin/IndexCollection'
    anserini_parameters = [
#                            'nohup', 
                           'sh',
                           anserini_index,
                           '-collection',
                           'TrecCollection',
                           '-generator',
                           'JsoupGenerator',
                           '-threads',
                            '16',
                            '-input',
                           index_input,
                           '-index',
                           index_loc,
                           '-storePositions',
                            '-keepStopwords',
                            '-storeDocvectors',
                            '-storeRawDocs']
#                           ' >& ',
#                           log_file,
#                            '&']



#     anserini_parameters = ['ls',
#                           index_loc]


#     print(anserini_parameters)

    index_proc = subprocess.Popen(anserini_parameters,
            stdout=subprocess.PIPE, shell=False)
    (out, err) = index_proc.communicate()
#     print(out.decode("utf-8"))
#     print(err)


# In[137]:


def generate_queries_file(queries, filename):
    queries_list = []
    queries_dict = {}
    query = {}
    q_dict = {}
    q_trec = {}
    ids_dict = {}
    id_num = 0
    for q in queries:
        str_id = str(id_num)
        id_new = str_id.rjust(15, '0')
#         print(q['body'])
#         text = q['body']
        text = remove_sc(q['body'])
#         print(text)
    
#         text = re.sub(r'[^\w\s]',' ',text)
##     text = text.lower()
##         text = text.rstrip('.?')
    
        q_dict[q['id']] = q['body']
        query['id_new'] = id_new
        query['number'] = q['id']
        query['text'] = '#stopword(' + text + ')'
        queries_list.append(dict(query))
        q_t = '<top>\n\n' +               '<num> Number: ' + id_new + '\n' +               '<title> ' + q['body'] + '\n\n' +               '<desc> Description:' + '\n\n' +               '<narr> Narrative:' + '\n\n' +               '</top>\n\n'
        q_trec[q['id']] = q_t
        ids_dict[str(id_num)] = q['id']
        id_num += 1
    queries_dict['queries'] = queries_list

    with open(filename, 'wt', encoding='utf-8') as q_file:
        json.dump(queries_dict, q_file, indent = 4)
    
    return [q_dict, q_trec, ids_dict]


# In[138]:


def retrieve_docs(q_topics_file, retrieved_docs_file, index_loc, b_val=0.2, k_val=0.8, n_docs=10, n_terms=10, w_ori_q=0.5, hits=100):
    
    anserini_search = anserini_loc + 'target/appassembler/bin/SearchCollection'
#     print(b_val)
    command = [ 
               'sh',
               anserini_search,
               '-topicreader',
                'Trec',
                '-index',
                index_loc,
                '-topics',
                q_topics_file,
                '-output',
                retrieved_docs_file,
                '-bm25',
                '-b',
                str(b_val),
                '-k1',
                str(k_val),
                '-rm3',
                '-rm3.fbDocs',
                str(n_docs),
                '-rm3.fbTerms',
                str(n_terms),
                '-rm3.originalQueryWeight',
                str(w_ori_q),
                '-hits',
                str(hits)
               ]
#     print(command)
#     command = command.encode('utf-8')
    anserini_exec = subprocess.Popen(command, stdout=subprocess.PIPE, shell=False, encoding='utf-8')
    (out, err) = anserini_exec.communicate()
###     print(out)
# ##    print(err)


# In[139]:


# Return top 100 bm25 scored docs, given query and corpus indexed by anserini

def generate_preds_file(retrieved_docs_file, q_dict, ids_dict, hits=100):
    
    with open(retrieved_docs_file, 'rt') as f_in:
        aux_var = -1
        bm25_docs = []
        while aux_var != 0:
            question = {}
            lines_gen = islice(f_in, hits)
            documents = []
            for line in lines_gen:
                id_aux = line.split(' ')[0]
                current_key = ids_dict[id_aux]
                documents.append(line.split(' ')[2])
                
###             print(documents)
            aux_var = len(documents)
            if aux_var == 0: 
                break
# ##            print(aux_var)##
# ##            print(documents)
            question['id'] = current_key
            question['body'] = q_dict[current_key]
            
            if "bioasq" in dataset_name: 
                documents_url = ['http://www.ncbi.nlm.nih.gov/pubmed/' + doc for doc in documents]
                question['documents'] = documents_url
            elif "rob04" in dataset_name:
                question['documents'] = documents
            bm25_docs.append(dict(question))
            
    return bm25_docs        


# In[140]:


# docus = generate_preds_file(retrieved_docs_file, q_dict, ids_dict)


# In[141]:


# len(docus)


# In[142]:




# pkl_files = [ x for x in os.listdir(dataloc) if all(y in x for y in ['docset', '.pkl'])]


# In[143]:





# In[144]:





# In[145]:


# Random grid search sampling

def get_random_params(hyper_params, num_iter):
    random_h_params_list = []
    while len(random_h_params_list) < num_iter:
        random_h_params_set = []
        for h_param_list in hyper_params:
            sampled_h_param = random.sample(list(h_param_list), k=1)
#             print(type(sampled_h_param[0]))
#             print(sampled_h_param[0])
            random_h_params_set.append(round(sampled_h_param[0], 3))
        if not random_h_params_set in random_h_params_list:
            random_h_params_list.append(random_h_params_set)
#             print('Non repeated')
        else:
            print('repeated')
    return random_h_params_list


# In[146]:





# In[147]:



# queries_file = dataloc + q_filename[0]

def load_queries(queries_file):
    with open(queries_file, 'rb') as input_file:
        query_data = json.load(input_file)
        return query_data['questions']





def save_preds(file, preds):
    with open(file, 'wt') as f_out:
        json.dump(preds, f_out, indent=4)
#     print('Predictions file: ' + file + ', done!')





# In[157]:


def start_process():
    print( 'Starting', multiprocessing.current_process().name)


# In[158]:


def extract_question(query):
    question = {}
    question['body'] = query['body']
    question['id'] = query['id']
###     print(query['body'].rstrip('.'))
#     documents = get_bm25_docs(query['body'].rstrip('.'), index_loc)
    documents = get_bm25_docs(query['body'], index_loc)
    if "bioasq" in dataset_name: 
        documents_url = ['http://www.ncbi.nlm.nih.gov/pubmed/' + doc for doc in documents]
        question['documents'] = documents_url
    elif "rob04" in dataset_name:
        question['documents'] = documents
    return dict(question)


# In[161]:


# b = 0.2
# k = 0.8
# retrieved_docs_file = baseline_files + 'bm25_preds_' + dataset_name_ext + '_' + data_split + '_' + 'b' + str(b) + 'k' + str(k) + '.txt'
# retrieve_docs(q_topics_file, retrieved_docs_file, q_dict, index_loc, b_val=0.2, k_val=0.8)


# In[162]:


def format_bioasq2treceval_qrels(bioasq_data, filename):
    with open(filename, 'wt') as f:
        for q in bioasq_data['questions']:
            for d in q['documents']:
                f.write('{0} 0 {1} 1'.format(q['id'], d))
                f.write('\n')

def format_bioasq2treceval_qret(bioasq_data, system_name, filename):
    with open(filename, 'wt') as f:
        for q in bioasq_data['questions']:
            rank = 1
            for d in q['documents']:
                
                sim = (len(q['documents']) + 1 - rank) / float(len(q['documents']))
                f.write('{0} {1} {2} {3} {4} {5}'.format(q['id'], 0, d, rank, sim, system_name))
                f.write('\n')
                rank += 1

def trec_evaluate(qrels_file, qret_file):
    trec_eval_res = subprocess.Popen(
#         ['./trec_eval', '-m', 'all_trec', qrels_file, qret_file],
        ['./trec_eval', 
         '-m', 'map',
         '-m', 'P.20',
         '-m', 'ndcg_cut.20',
         qrels_file, qret_file],
        stdout=subprocess.PIPE, shell=False)

    (out, err) = trec_eval_res.communicate()
    trec_eval_res = out.decode("utf-8")
#     print(trec_eval_res)
#     print(out)
#     print(err)
####     return trec_eval_res.split('\tall\t')
    return trec_eval_res.strip('map').replace('P_20','').replace('ndcg_cut_20','').replace(' ','').replace('\tall\t','').split('\n')[:-1]


# In[163]:


def evaluate(golden_file, predictions_file):

    system_name = predictions_file
    
    with open(golden_file, 'r') as f:
        golden_data = json.load(f)

    with open(predictions_file, 'r') as f:
        predictions_data = json.load(f)

    temp_dir = uuid.uuid4().hex
    qrels_temp_file = '{0}/{1}'.format(temp_dir, 'qrels.txt')
    qret_temp_file = '{0}/{1}'.format(temp_dir, 'qret.txt')

    try:
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        else:
            sys.exit("Possible uuid collision")

        format_bioasq2treceval_qrels(golden_data, qrels_temp_file)
        format_bioasq2treceval_qret(predictions_data, system_name, qret_temp_file)

        results = trec_evaluate(qrels_temp_file, qret_temp_file)
    finally:
#         print('something')
        os.remove(qrels_temp_file)
        os.remove(qret_temp_file)
        os.rmdir(temp_dir)
    return results


# In[164]:


def bm25_computing(params):
    b = params[0]
    k = params[1]
    N = params[2]
    M = params[3]
    Lambda = params[4]
#     b = 0.2
#     k = 0.8
    params_suffix = 'b' + str(b) + 'k' + str(k) + 'N' + str(N) + 'M' + str(M) + 'Lambda' + str(Lambda)

    bm25_preds_file = baseline_files + 'bm25_rm3_preds_' + dataset_name_ext + '_' + data_split + '_' + params_suffix + '.json'
    
    ###     print(bm25_preds_file)
    if os.path.isfile(bm25_preds_file):
        print(bm25_preds_file + "Already exists!!")
#         return
    retrieved_docs_file = baseline_files + 'run_bm25_rm3_preds_' + dataset_name_ext + '_' + data_split + '_' + params_suffix + '.txt'
    #print(b)
    #print(k)
    retrieve_docs(q_topics_file, retrieved_docs_file, index_loc, b, k, N, M, Lambda)
    bm25_preds = {}
    bm25_preds['questions'] = generate_preds_file(retrieved_docs_file, q_dict, ids_dict)

    save_preds(bm25_preds_file, bm25_preds)  
    
    if 'rob04' in dataset_name_ext:
        golden_file = dataloc + 'rob04.' + data_split + '.' + s[0]   + '.json'
    else:
        golden_file = dataloc + dataset_name_ext + '.' + data_split + '.json'
    
    if os.path.exists(golden_file):
#         print('yes, we can evaluate!')    
#         print(golden_file)    
        [map_metric, p_20, ndcg_20] = evaluate(golden_file,bm25_preds_file)
    else:
        print('no, we cannot evaluate  :( !')  
        
#     results = {
#         'b': b,
#         'k': k,
#         'N': N,
#         'M': M,
#         'Lambda': Lambda,
#         'map': map_metric,
#         'p_20': p_20,
#         'ndcg_20': ndcg_20
        
#     }

    results = [
        b,
        k,
        N,
        M,
        Lambda,
        float(map_metric),
        float(p_20),
        float(ndcg_20)
    ]
    os.remove(retrieved_docs_file)
    os.remove(bm25_preds_file)

    return results


# In[165]:


# if __name__ == '__main__':
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
        time.sleep(2)
        print("Waiting for", remaining, "tasks to complete...")
      
    pool.join()  # wrap up current tasks
    pool_outputs.get()
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


# In[166]:


def get_test_metrics(best_model_params_file):
    try:
        with open(best_model_params_file, 'rt') as best_model_in:
            best_dev_params = json.load(best_model_in)
    #         print(best_dev_model_params)
    #         best_dev_params = {k:float(v) for k, v in best_dev_params.items()}
        params = [best_dev_params['b'],
                  best_dev_params['k'],
                  best_dev_params['N'],
                  best_dev_params['M'],
                  best_dev_params['Lambda']
                 ]
        test_results = bm25_computing(params)
        return test_results
    except:
        print('No dev model file. Run Dev model first!')


# In[167]:


# best_model_params_file = baseline_files + dataset_name_ext + '_bm25_rm3_best_model_test.json'
# find_best_dev_model(best_model_params_file, 2)


# In[171]:


if __name__ == '__main__':

    try:
        dataloc = sys.argv[1]
        print(dataloc)
        split = sys.argv[2]
        
    except:
        sys.exit("Provide data location, split, and number of random iterations")
    
    try:
        random_iter = sys.argv[3]
    except: 
        if 'dev' in split:
            print('No number random of random iterations provided. Using default = 5000')
            random_iter = 5000
        elif 'test' in split:
            print('No need for random iterations in test mode.')
            random_iter = 1

    ## Options

    # search best b and k now?
    random_search = 'yes' 
    # random_search = 'no' 

    # build index? 
    build_index_flag = 'yes'
    # build_index_flag = 'no'

    # N of workers for multiprocessing used random_search
    pool_size = 20

    hits = 100
    
    
    # Define paths
#     dataloc = '../../bioasq_data/'
    # dataloc = '../../robust04_data/split_1/'
    baseline_files ='./baseline_files/'
    corpus_files ='./corpus_files/'
    galago_loc='./galago-3.10-bin/bin/'
    anserini_loc = '../../../anserini/'

    ## TREC storage
    trec_storage = '/ssd/francisco/trec_datasets/deep-relevance-ranking/'
    
    
    # Select data split to work with
#     split = "test"
    # split = "dev"
    # split = "train"
    
    
    pkl_files = [os.path.join(root, name)
    for root, dirs, files in os.walk(dataloc)
    for name in files
    if all(y in name for y in ['docset', split, '.pkl'])]
    
    
    # Convert pickle to trectext file format to be processed with galago
    # pkl_file = [s for s in pkl_files if split in s]
    # [output_file, doc_list ]= pickle_to_json(pkl_file[0])
    doc_list = []
    output_files = []
    all_docs = []
    for pkl_file in pkl_files:
    ###     print(pkl_file)
        docs = get_pickle_docs(pkl_file)
        doc_list = doc_list + list(docs.keys())
        all_docs.append(docs)
        out_name = pkl_file.split('/')[-1:][0]
        out_name = re.sub('\.pkl', '', out_name)
        output_file = corpus_files + out_name + '.gz'
        trec_doc_file = trec_storage + out_name
        output_files.append(output_file)
        ### print(out_name)
        to_trecfile(docs, output_file)
        to_trecfile(docs, trec_doc_file, 'no')
    
    data_split = split
    print(data_split)

    if "rob04" in output_files[0]:
        s = re.findall("(s[0-5]).pkl$", pkl_file)
        dataset_name = "rob04"
        dataset_name_ext = dataset_name + '_'+ s[0]
    #     dataset_name_ext = dataset_name 
        gold_file = '../../robust04_data/rob04.' + split +'.json'
    #     with open(gold_file, 'w') as outfile:
    #         json.dump(query_data, outfile, indent = 4)
        print(dataset_name_ext)
    elif "bioasq" in output_file:
        print("bioasq")
        dataset_name = "bioasq"
        dataset_name_ext = dataset_name
    
    
    # Build index
    
    index_loc = baseline_files + 'anserini_index' + '_' + dataset_name_ext + '_' + data_split
    # index_input = output_files
    index_input = corpus_files
    log_file = baseline_files + 'log_index_' + dataset_name_ext + '_' + data_split

    if build_index_flag == 'yes':
        build_index(index_input, index_loc, log_file)

    #     build_index(index_input, index_loc)
    
    q_filename = [ x for x in os.listdir(dataloc) if all(y in x for y in [dataset_name +'.'+ data_split, '.json'])]
    
    query_files = [os.path.join(root, name)
             for root, dirs, files in os.walk(dataloc)
             for name in files
             if all(y in name for y in [dataset_name +'.'+ data_split, '.json'])]
    
    queries = []
    query_data = {}
    for file in query_files:
        queries = queries + load_queries(file)
    # ##    print(queries)
    query_data['questions'] = queries
    
    query_files[0].strip('split_1')
    
    bm25_queries_file = baseline_files + 'bm25_queries_' + dataset_name_ext + '_' + data_split + '.json'
    [q_dict, q_trec, ids_dict]= generate_queries_file(queries,bm25_queries_file)

    q_topic_filename = dataset_name_ext + '_' + 'query_topics'  + '_' + data_split + '.txt'
    q_topics_file = baseline_files + q_topic_filename
    trec_q_topics_file = trec_storage + q_topic_filename

    to_trecfile(q_trec, q_topics_file, compression = 'no')
    to_trecfile(q_trec, trec_q_topics_file, compression = 'no')
    
    
    best_model_params_file = baseline_files + dataset_name_ext + '_bm25_rm3_best_model_dev.json'
    ## best_model_params_file = baseline_files + dataset_name_ext + '_bm25_rm3_best_model_'+ split + '.json'
    ## find_best_dev_model(best_model_params_file, 2)
    if 'dev' in split:
        find_best_dev_model(best_model_params_file, int(random_iter))
    if 'test' in split:
        test_results = get_test_metrics(best_model_params_file)
        print(test_results)
    
    
   

