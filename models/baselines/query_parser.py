#!/usr/bin/env python
# coding: utf-8

# In[31]:


import datetime
import xml.etree.ElementTree as ET
import gzip
import os
import subprocess
import json

import multiprocessing
import re 
import sys
# sys.path.append('qra_cod')
import shutil

# my modules
import ir_utils
from ir_utils import load_queries


# In[32]:





# In[33]:


def trec_queries(queries):
    
    qrels_list = []
    q_trec = {}
    for q in queries:
#         print(q['body'])
#         text = q['body']
        text = ir_utils.remove_sc(q['body'])
#         print(text)
    
#         text = re.sub(r'[^\w\s]',' ',text)
##     text = text.lower()
##         text = text.rstrip('.?')
        q_t = '<query>\n' + '<number>' + q['id'] + '</number>\n' + '<text>' + text + '</text>\n' + '</query>\n'
        q_trec[q['id']] = q_t
        
        docs = q['documents']
        
        for doc in docs:
            doc = doc.split('/')[-1]# strip url information
            qrel_string = q['id'] + ' 0 ' + doc + ' 1\n'
            qrels_list.append(qrel_string)
    
    return [q_trec, qrels_list]


# In[34]:


def save_trec_query(trec_queries, trec_query_file):
    with open(trec_query_file, 'wt', encoding='utf-8') as q_file:
        q_file.write('<parameters>\n')
        for key, value in trec_queries.items():
            q_file.write(value)
        q_file.write('</parameters>')


# In[35]:


def save_qrels(qrels, qrels_file):
    with open(qrels_file, 'wt', encoding='utf-8') as q_file:
        q_file.writelines(qrels)


# In[36]:


def query_parser(queries_file, trec_query_file, qrels_file):
    
    queries = load_queries(queries_file)
    [q_trec, qrels] = trec_queries(queries)
    
    save_trec_query(q_trec, trec_query_file)
    save_qrels(qrels, qrels_file)

