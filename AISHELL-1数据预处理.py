#!/usr/bin/env python
# coding: utf-8

# ## 预处理方式一
# 你可以使用自己的数据集来训练模型。你的数据集需要包含至少以下3个文件：
#  - train.index
#  - dev.index
#  - labels.json
# 
# train.index和dev.index为索引文件，表示音频文件和标注的对应关系，应具有如下的简单格式：
# ```text
# /path/to/audio/file0.wav,数据预处理
# /path/to/audio/file1.wav,小时不识月
# ...
# ```
# 
# labels.gz是pkle文件，应包含数据集标注中出现过的所有字符，表示为一个list数组。其中开头首字符必须是无效字符（可任意指定，不和其他字符重复就行），预留给CTC作为blank label;建议索引0为'_'，索引28位' '
# ```text
# [
#    '_', // 第一个字符表示CTC空字符，可以随便设置，但不要和其他字符重复。
#    '小',
#    '时',
#    '不',
#    '识',
#    '月',
#    ...
# ]
# ```

# ## 预处理方式二
# train.index和dev.index为索引文件，表示音频文件和标注的对应关系，应具有如下的简单格式：
# ```text
# /path/to/audio/file0.wav,数据 预 处理
# /path/to/audio/file1.wav,小时 不识 月
# ...
# ```
# 
# labels.gz是pkle文件，应包含数据集标注中出现过的所有字符，表示为一个list数组。其中开头首字符必须是无效字符（可任意指定，不和其他字符重复就行），预留给CTC作为blank label;建议索引0为'_'，索引28位' '
# ```text
# [
#    '_', // 第一个字符表示CTC空字符，可以随便设置，但不要和其他字符重复。
#    '小时',
#    '不识',
#    '月',
#     '预',
#     '处理'
#    ...
# ]
# ```
# **注：如果是方式二处理，则data.py中MASRDataset类读取数据后处理的方式要有所改动**<br>
# 原始数据集AISHELL-1已经给我们分好词，也可以自行用jieba分词

# In[1]:


import os
import re
import joblib
import librosa
import torch
import wave
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from random import shuffle


# ## 读取wav文件

# In[21]:


train_path_dir = 'data_aishell/wav/train/'
dev_path_dir = 'data_aishell/wav/dev/'


# In[22]:


train_files_path = []
dev_files_path = []
def recur_train(rootdir):
    for root, dirs, files in tqdm(os.walk(rootdir)):
        for file in files:
            if 'DS_Store' in file:
                continue
            train_files_path.append(os.path.join(root,file))
        for dir in dirs:
            recur_train(dir)
def recur_dev(rootdir):
    for root, dirs, files in tqdm(os.walk(rootdir)):
        for file in files:
            if 'DS_Store' in file:
                continue
            dev_files_path.append(os.path.join(root,file))
        for dir in dirs:
            recur_dev(dir)
recur_train(train_path_dir)
recur_dev(dev_path_dir)
# wav_paths = [x for x in all_files_path if 'wav' in x]


# In[25]:


print('train_files_path len:', len(train_files_path))
print('dev_files_path len:', len(dev_files_path))
all_files_path = train_files_path+dev_files_path


# In[26]:


print(len(all_files_path))
all_files_path


# ## 读取transcript文件

# In[110]:


#读取transcript文件，处理成字典形式{'BAC009S0002W0122': '而对楼市成交抑制作用最大的限购'}


# In[27]:


_d = {}
with open('data_aishell/transcript/aishell_transcript_v0.8.txt', encoding='utf-8') as f:
    data = f.readlines()
    for i in tqdm(data):
        k, v = re.split('\s+', i, 1)
        _d[k.strip()] = v.replace('\n','').replace('\t','').replace(' ','')


# ## 生成train.index, dev.index和labels.gz三个文件

# In[29]:


res_train = []
for file in tqdm(train_files_path):
    file_name = file.split('/')[-1][:-4]
    if file_name in _d:
        res_train.append((file, _d[file_name]))
res_dev = []
for file in tqdm(dev_files_path):
    file_name = file.split('/')[-1][:-4]
    if file_name in _d:
        res_dev.append((file, _d[file_name]))


# In[31]:


all_words = list(set(''.join([v for v in _d.values()])))
all_words = ['_'] + all_words[:27] + [' '] + all_words[27:]


# In[43]:


len(all_words)


# In[36]:


pd.DataFrame(res_train).to_csv('data_aishell/train.index',index=False,header=None)
pd.DataFrame(res_dev).to_csv('data_aishell/dev.index',index=False,header=None)
joblib.dump(all_words, 'data_aishell/labels.gz')


# In[13]:


# #读取看看
# with open('data_aishell/train.index') as f:
#     idx = f.readlines()
# idx = [x.strip().split(",", 1) for x in idx]

# all_words = joblib.load('data_aishell/labels.gz')


