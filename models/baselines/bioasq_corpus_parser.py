#!/usr/bin/env python
# coding: utf-8
# python 3
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
    if not os.path.exists(folder):
        os.makedirs(folder)

# In[5]:


def xml_to_trec(root):
    
    trec_docs = {}
    doc_year = {}
    for MedlineCitation in root.iter('MedlineCitation'):
        doc = {}
        try:
            PMID = MedlineCitation.find('PMID').text
            title_obj = MedlineCitation.find('Article/ArticleTitle')
            title = "".join(title_obj.itertext())
            abstractTexts_obj = MedlineCitation.findall('Article/Abstract/AbstractText')
            abstractTexts_iters = [x.itertext() for x in abstractTexts_obj]
            pubDate_year = MedlineCitation.find('Article/Journal/JournalIssue/PubDate/Year').text
#             print(abstractTexts_iters)
            abstractText = [" ".join(x) for x in abstractTexts_iters]
            abstractText = " ".join(abstractText)
            if len(abstractText) == 0:
                #print('length is zero') 
                continue
        except:
            #e = sys.exec_info()[0]
            #print(e)
            continue
#         try:
#             pubDate_year = MedlineCitation.find('Article/Journal/JournalIssue/PubDate/Year').text
#         except:
#             pubDate_year = None

        if (PMID is None) | (title is None) | (abstractText is None) | (pubDate_year is None):
            print('PMID: ',PMID)
            print('title: ',title)
            print('abstractText :' ,abstractText)
            print('All abstract objects: ',MedlineCitation.findall('Article/Abstract/AbstractText'))
            print('pubDate_year :' ,pubDate_year)
            break
        

        trec_doc = '<DOC>\n' + '<DOCNO>' + PMID + '</DOCNO>\n' + '<TITLE>' + title + '</TITLE>\n' + '<TEXT>' + abstractText + '</TEXT>\n' + '<YEAR>' + pubDate_year + '</YEAR>\n' + '</DOC>\n'
        trec_docs[PMID] = trec_doc
        doc_year[PMID] = pubDate_year
    print(len(doc_year))
    return [trec_docs, doc_year] 



# In[17]:


def save_trecfile(docs, filename, compression = 'yes'):
    # Pickle to Trectext converter
    print('Preparing to save:', filename)
    if compression == 'yes':
        with gzip.open(filename,'wt') as f_out:
            for key, value in docs.items():
                f_out.write(value.encode('utf8'))
    else:
        with open(filename,'wt', encoding='utf-8') as f_out:
#         with open(filename,'wt') as f_out: # python 2.7
            for key, value in docs.items():
                # print(value)
#                 f_out.write(value.encode('utf8')) # python 2.7
                f_out.write(value)
    print(filename, ' ... done!')
    
# In[7]:


def trec_filename_gen(xml_filename):
    prefix = xml_filename.split('/')[-1:][0].strip('.xml')
    to_index_dir = './bioasq_dir/bioasq_corpus/' # TODO Fix, this should be a variable, not hardcoded
    filename = to_index_dir + prefix + '_trec_doc.txt'
    return filename


# In[8]:


def start_process():
    print( 'Starting', multiprocessing.current_process().name)


def pubmed_xml_to_json(xml_file):

#     print('Trec_docs... generated!')
    # Gen trec filename
    trec_file = trec_filename_gen(xml_file)
    doc_year_file = trec_file.split('_')[0] + '_doc_year'

    if os.path.exists(trec_file):
#         pass
        print(trec_file, ' Already exists... skip!')
    else:
        print(trec_file, ' processing...')
    	# Read ZIP file
        xml_str = xml_file
        print('xml_str... done!')
        # Parse to root object
        tree = ET.parse(xml_str)
        root = tree.getroot()

        print('Root xml_str... done!')
        # Convert to Trec docs
        [trec_docs, doc_year]= xml_to_trec(root)
        print(len(trec_docs))


        #     print('Trec_file... saved!!')
        # Save as TREC (Anserini) doc index input file
        save_trecfile(trec_docs, trec_file, compression = 'no')
        with open(doc_year_file, 'wt') as doc_year_f:
            json.dump(doc_year, doc_year_file, indent=4)

# In[16]:


def corpus_parser(data_dir, to_index_dir, pool_size):

#     pool_size = 25
#     data_dir = '/ssd/francisco/pubmed19/'
#     to_index_dir = './bioasq_dir/bioasq_corpus/' # TODO Fix, pass to multiprocessing!

    make_folder(to_index_dir)

    pubmed_files = get_filenames(data_dir)
	

    # assign to the multiprocessing pool 
    
    
    pool = multiprocessing.Pool(processes=pool_size,
                            initializer=start_process,
                            )

#     pool_outputs = pool.map(baseline_computing, params)


    pool.map_async(pubmed_xml_to_json, pubmed_files)

    pool.close() # no more tasks

    pool.join()  # wrap up current tasks
    
 

