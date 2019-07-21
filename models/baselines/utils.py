
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
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path) 


# In[4]:




def remove_sc(text):
##     text = re.sub('[.,?;*!%^&_+():-\[\]{}]', '', text.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip())
##     text = re.sub('[\[\]{}.,?;*!%^&_+():-]', '', text.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip()) # DeepPaper method
    text = re.sub(r'[^\w\s]',' ',text) # My method
##     text = text.rstrip('.?')
    return text


# Tokenization