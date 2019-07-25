#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import re
# import nltk
import os

# import bioasq_corpus_parser
import utils
import multiprocessing


# In[2]:


def get_filenames(data_dir):
    filenames = [os.path.join(root, name)
             for root, dirs, files in os.walk(data_dir)
             for name in files
             if "DS_Store" not in name
           ]
    return filenames


# In[3]:


def start_process():
    pass
#     print( 'Starting', multiprocessing.current_process().name)


# In[4]:


def process_file(file):
    
    filename = file.split('/')[-1:][0]
    
    outdir = to_index_dir + '/'.join(file.split('/')[-2:-1]) + '/'
#     print(outdir)
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    file_out = outdir + filename
    
    open_tags = ['<H3>', '<HT>', '<TEXT>', '<HEADLINE>']
    close_tags = ['</H3>', '</HT>', '</TEXT>', '</HEADLINE>']
    try:
        with open(file, 'rt', encoding = "ISO-8859-1") as input_f, open(file_out, 'wt', encoding = "utf-8") as out_f:

            lines  = []
    #         i = 0
            open_tag = False
            for line in input_f:
    #             i += 1
    #             if i> 200:
    #                 break
                if any(tag in line for tag in close_tags):
                    out_f.write(line)
                    open_tag = False
    #                 print(line)
                    continue

                elif any(tag in line for tag in open_tags):
                    open_tag = True
                    out_f.write(line)
    #                 print(line)
                    continue

                if open_tag:
    #                 print('change')
                    line = utils.remove_sc(line) + '\n'

                out_f.write(line)
                print('Saved :', file_out)    
    except:
        print('error processing file :', file)    


# In[5]:


def corpus_parser(data_dir, to_index_dir, pool_size):
    pass


# In[7]:


if __name__ == "__main__":
    
    data_dir = '/ssd2/francisco/robust_corpus/'
    
    
    to_index_dir = './robust_dir/robust_corpus/'
    
    pool_size = 10
    
    corpus_files = get_filenames(data_dir)
    
#     corpus_files = corpus_files[0:10]
    
    print(len(corpus_files))
    
    pool = multiprocessing.Pool(processes=pool_size,
                            initializer=start_process,
                            )

#     pool_outputs = pool.map(baseline_computing, params)


    pool.map_async(process_file, corpus_files)

    pool.close() # no more tasks

    pool.join()  # wrap up current tasks

