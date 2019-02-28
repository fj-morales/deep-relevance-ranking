
# coding: utf-8

# ## Converting pkl files to json files to be read in galago tool

# In[53]:


import pickle
import json
import gzip
import os
import subprocess
import numpy as np


# In[78]:


# Define paths
dataloc = '../../bioasq_data/'
# dataloc = '../../robust04_data/split_3/'
baseline_files ='./baseline_files/'
galago_loc='./galago-3.10-bin/bin/'


# In[55]:


def pickle_to_json(pickle_filename):
    # Pickle to Trectext converter
    doc_list = []
    with open(dataloc + pickle_filename, 'rb') as f_in:
        data = pickle.load(f_in)
        if not os.path.exists(baseline_files):
            os.makedirs(baseline_files)
        out_file = baseline_files + pickle_filename[:-4] + '.gz'
        with gzip.open(out_file,'wt', encoding='utf-8') as f_out:
            docu = {}
            for key, value in data.items():
                if "pmid" in value.keys():
                    doc_code = value.pop('pmid')
                else:
                    doc_code = key
                f_out.write('<DOC>\n' + 
                            '<DOCNO>' + doc_code + '</DOCNO>\n' +
                            '<TITLE>' + value.pop('title') + '</TITLE>\n' +
                            '<TEXT>' + value.pop('abstractText') + '</TEXT>\n' + 
                            '</DOC>\n')
                doc_list.append(doc_code)
        return [out_file, doc_list]


# In[56]:


# Build corpus index 
def build_index(index_input, index_loc):
    index_input_param = '--inputPath+' + index_input    
    index_loc_param = '--indexPath=' + index_loc
    print(index_input_param)
    print(index_loc_param)
    if not os.path.exists(index_loc):
            os.makedirs(index_loc) 
    index_proc = subprocess.Popen(
            [galago_loc + 'galago', 'build', '--stemmer+krovetz',
                index_input_param, index_loc_param],
            stdout=subprocess.PIPE, shell=False)
    (out, err) = index_proc.communicate()
    print(out.decode("utf-8"))
    print(err)


# In[57]:


# # Return top 100 bm25 scored docs, given query and corpus indexed by galago
# def get_bm25_docs(query, index_loc):
#     index_loc_param = '--index=' + index_loc   
#     if "'" in query:
#         query_param = '--query="#stopword(' + query.rstrip('.') + ')"' 
#     else:
#         query_param = '--query=\'#stopword(' + query.rstrip('.') + ')\'' 
        
# #     print(query_param)

#     command = galago_loc + 'galago batch-search --verbose=false --requested=100 ' + \
#          index_loc_param + ' --scorer=bm25 --stemmer+krovetz ' + \
#          query_param + ' | cut -d" " -f3'
#     galago_bm25_exec = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
#     (out, err) = galago_bm25_exec.communicate()
#     bm25_documents = out.decode("utf-8")
#     return bm25_documents.splitlines()


# In[75]:


# Return top 100 bm25 scored docs, given query and corpus indexed by galago
def get_bm25_docs(query, index_loc, b_val, k_val):
    query = query.lower()
    index_loc_param = '--index=' + index_loc  
    b=' --b=' + str(b_val)
    k=' --k=' + str(k_val)
    if "'" in query:
        query_param = '--query="#stopword(' + query.rstrip('.') + ')"' 
    else:
        query_param = '--query=\'#stopword(' + query.rstrip('.') + ')\'' 

    command = galago_loc + 'galago batch-search --verbose=false --requested=100 ' +          index_loc_param + ' --scorer=bm25' +          b +          k +          ' --stemmer+krovetz ' +          query_param + ' | cut -d" " -f3'
#     print(command)
    galago_bm25_exec = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (out, err) = galago_bm25_exec.communicate()
    bm25_documents = out.decode("utf-8")
    return bm25_documents.splitlines()


# In[59]:


# # ## Testing (remove)

# import numpy as np
# index_loc = '/home/fmorales/msc_project/not-a-punching-bag/reproduction/deep-relevance-ranking/models/baselines/baseline_files/index_bioasq_test'
# # # q = 'Has \"RNA interference\" been awarded Nobel prize.'
# # # q = 'Describe Wellens\' Syndrome.'
# # # q = 'Can the Micro-C XL method achieve mononucleosome resolution?'
# # # q = 'What is the role of the UBC9 enzyme in the protein sumoylation pathway?'
# # # q = "Has \"RNA interference\" been awarded Nobel prize?"
# q = "What is the role of gamma-secreatase complex in Alzheimer's Disease?"
# # q = "List diseases associated with the  Dopamine Receptor D4 (DRD4)."
# # "List the classical symptoms of the Moschcowitz syndrome (Thrombotic thrombocytopenic purpura)."
# # q = "List the classical symptoms of the Moschcowitz syndrome (Thrombotic thrombocytopenic purpura)."

# b_range = np.arange(0.4, 0.9, 0.1)
# k_range = np.arange(1,2.5,0.2)
# for b in b_range:
#     for k in k_range:
#         print(round(b,1),round(k,1))
#         valor = get_bm25_docs(q, index_loc, round(b,1), round(k,1))
#         print(valor)


# In[60]:


pkl_files = [ x for x in os.listdir(dataloc) if all(y in x for y in ['docset', '.pkl'])]


# In[61]:


pkl_files


# In[62]:


# Convert pickle to trectext file formar to be processed with galago
for pkl_file in pkl_files[0:1]:
    [output_file, doc_list ]= pickle_to_json(pkl_file)


# In[63]:


if "dev" in output_file:
    print("dev")
    data_split = "dev"
elif "test" in output_file:
    print("test")
    data_split = "test"
elif "train" in output_file:
    print("train")
    data_split = train

    
if "rob04" in output_file:
    print("rob04")
    dataset_name = "rob04"
elif "bioasq" in output_file:
    print("bioasq")
    dataset_name = "bioasq"


# In[64]:


index_loc = baseline_files + 'index' + '_' + dataset_name + '_' + data_split
index_input = output_file
build_index(index_input, index_loc)


# In[65]:


q_filename = [ x for x in os.listdir(dataloc) if all(y in x for y in [dataset_name +'.'+ data_split, '.json'])]


# In[66]:


q_filename


# In[67]:


queries_file = dataloc + q_filename[0]
with open(queries_file, 'rb') as input_file:
    query_data = json.load(input_file)


# In[68]:


def save_preds(file, preds):
    with open(file, 'wt') as f_out:
        json.dump(preds, f_out, indent=4)
    print('Predictions file: ' + file + ', done!')


# In[69]:


print(index_loc)


# In[ ]:


defaults = 'yes'
if defaults == 'yes':
    brange = [0.75]
    krange = [1.2]
else:
    brange = np.arange(0.2,1,0.1)
    krange = np.arange(0.5,2,0.1)
for b in brange:
    b = round(b,2)
    for k in krange:
        k = round(k,2)
        bm25_preds = {}
        questions = []
        question = {}
        for query in query_data['questions']:
            question['body'] = query['body']
            question['id'] = query['id']
        #     print(query['body'].rstrip('.'))
        #     documents = get_bm25_docs(query['body'].rstrip('.'), index_loc)
            documents = get_bm25_docs(query['body'], index_loc, b, k)
            if "bioasq" in dataset_name: 
                documents_url = ['http://www.ncbi.nlm.nih.gov/pubmed/' + doc for doc in documents]
                question['documents'] = documents_url
            elif "rob04" in dataset_name:
                question['documents'] = documents
            questions.append(dict(question))
        bm25_preds_file = baseline_files + 'bm25_preds_' + 'b' + str(b) + 'k' + str(k) + '_'+ dataset_name + '_'+ data_split + '.json'
        bm25_preds['questions'] = questions
        save_preds(bm25_preds_file, bm25_preds)    


# ## Experiments (remove safely)

# In[71]:


# from random import sample
# random_preds = {}
# questions = []
# question = {}
# for query in query_data['questions']:
#     question['body'] = query['body']
#     question['id'] = query['id']
# #     print(query['body'].rstrip('.'))
# #     documents = get_bm25_docs(query['body'].rstrip('.'), index_loc)
#     documents = sample(doc_list, 100)
#     if "bioasq" in dataset_name: 
#         documents_url = ['http://www.ncbi.nlm.nih.gov/pubmed/' + doc for doc in documents]
#         question['documents'] = documents_url
#     elif "rob04" in dataset_name:
#         question['documents'] = documents
#     questions.append(dict(question))
    
# random_preds['questions'] = questions


# In[72]:


# random_preds_file = baseline_files + 'random_preds.' + dataset_name + data_split + '.json'
# with open(random_preds_file, 'wt') as f_out:
#     json.dump(random_preds, f_out, indent=4)


# In[73]:


# len(doc_list)

