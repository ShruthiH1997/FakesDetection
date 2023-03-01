#!/usr/bin/env python
# coding: utf-8

# In[51]:


import os
import pandas as pd
import re
import glob
import tqdm


# In[52]:


def clean(text):
    
    # Remove newline characters and unnecessary whitespaces
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove headers
    # text = re.sub(r'==.*?==', '', text)
    # text = re.sub(r'=.*?=', '', text)
    text = re.sub(r'=+[^=]+=+', '', text)

    return text


# ### Preprocess and Save

# In[53]:


if not os.path.exists('./data_cleaned'):
    os.mkdir('./data_cleaned')


# In[60]:


for file in glob.glob('data/*.tok'):
    with open(file, 'r') as f:
        filename = file.split('/')[-1].replace('.tok', '')
        data = f.read().split('< start_bio >')
        pd.DataFrame([clean(para) for para in data if clean(para)]).to_csv(f'./data_cleaned/{filename}.clean', index=False, header=False)
        

print("Processing finished!")

# In[ ]:




