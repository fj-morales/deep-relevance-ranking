#!/usr/bin/env python
# coding: utf-8

# ## Pubmed coverter

# In[1]:


import datetime
import xml.etree.ElementTree as ET
import gzip
import os
import subprocess

import multiprocessing
import re 
import sys
# sys.path.append('qra_cod')
import shutil


# In[2]:


# print(datetime.datetime.now())


# In[3]:


def get_filenames(data_dir):
    filenames = [os.path.join(root, name)
             for root, dirs, files in os.walk(data_dir)
             if ("test_mp") not in root
             for name in files
             if name.endswith('.xml')
           ]
    return filenames


# In[4]:


def make_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
        os.makedirs(folder)
    else:
        os.makedirs(folder)


# In[5]:


def xml_to_trec(root):
    
    trec_docs = {}
    for MedlineCitation in root.iter('MedlineCitation'):
        doc = {}
        try:
            PMID = MedlineCitation.find('PMID').text
            title = MedlineCitation.find('Article/ArticleTitle').text
            abstractText = MedlineCitation.find('Article/Abstract/AbstractText').text
        except:
            continue
#         try:
#             pubDate_year = MedlineCitation.find('Article/Journal/JournalIssue/PubDate/Year').text
#         except:
#             pubDate_year = None

        trec_doc = '<DOC>\n' +               '<DOCNO>' + PMID + '</DOCNO>\n' +               '<TITLE>' + title + '</TITLE>\n' +               '<TEXT>' + abstractText + '</TEXT>\n' +               '</DOC>\n'
        trec_docs[PMID] = trec_doc
        
    return trec_docs


# In[17]:


def save_trecfile(docs, filename, compression = 'yes'):
    # Pickle to Trectext converter
    if compression == 'yes':
        with gzip.open(filename,'wt') as f_out:
            
            for key, value in docs.items():
                f_out.write(value.encode('utf8'))
    else:
       # print(filename)
        with open(filename,'wt') as f_out:
            
            for key, value in docs.items():
#                 print(value)
                f_out.write(value.encode('utf8'))
    print(filename, ' ... done!')


# In[7]:


def trec_filename_gen(xml_filename):
    prefix = xml_filename.split('/')[-1:][0].strip('.xml')
    filename = to_index_dir + prefix + '_trec_doc.txt'
    return filename


# In[8]:


def start_process():
    print( 'Starting', multiprocessing.current_process().name)


def pubmed_xml_to_json(xml_file):

    # Read ZIP file
    xml_str = xml_file
#     print('xml_str... done!')
    # Parse to root object
    tree = ET.parse(xml_str)
    root = tree.getroot()
    
#     print('Root xml_str... done!')
    # Convert to Trec docs
    trec_docs = xml_to_trec(root)
    
#     print('Trec_docs... generated!')
    # Gen trec filename
    trec_file = trec_filename_gen(xml_file)
    
#     print('Trec_file... saved!!')
    # Save as TREC (Anserini) doc index input file
    save_trecfile(trec_docs, trec_file, compression = 'no')


# In[16]:


if __name__ == '__main__':

    pool_size = 5
    # Get all filenames
    data_dir = '/ssd/francisco/pubmed19/'
    to_index_dir = './baseline_files/corpus_files/'

    make_folder(to_index_dir)

    pubmed_files = get_filenames(data_dir)

    # assign to the multiprocessing pool 
#     pubmed_xml_to_json(pubmed_files[10])
    
    
    pool = multiprocessing.Pool(processes=pool_size,
                            initializer=start_process,
                            )

#     pool_outputs = pool.map(baseline_computing, params)


    pool.map_async(pubmed_xml_to_json, pubmed_files)

    pool.close() # no more tasks

    pool.join()  # wrap up current tasks
    
 

