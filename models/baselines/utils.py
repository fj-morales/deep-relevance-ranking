
# coding: utf-8

# In[1]:


import os
import shutil
import re 
import nltk


# In[2]:


# Utils


# In[3]:


# Create dir function
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def destroy_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path) 
        
# In[4]:



clean = lambda t: re.sub('[,?;*!%^&_+():-\[\]{}]', ' ', t.replace('"', ' ').replace('/', ' ').replace('\\', ' ').replace("'", ' ').replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('-', ' ').replace('.', '').replace('&hyph;', ' ').replace('&blank;', ' ').strip().lower())


def remove_sc(text):

    text = re.sub(r'[^\w\s]',' ',text) # My method
#     text = clean(text) # Their method

    return text

def remove_sc2(text):

#     text = re.sub(r'[^\w\s]',' ',text) # My method
    text = clean(text) # Their method

    return text

# def remove_sc(text):
# ##     text = re.sub('[.,?;*!%^&_+():-\[\]{}]', '', text.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip())
# ##     text = re.sub('[\[\]{}.,?;*!%^&_+():-]', '', text.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip()) # DeepPaper method
#     text = re.sub(r'[^\w\s]',' ',text) # My method
# ##     text = text.rstrip('.?')
#     return text


# Tokenization