from transformers import AutoTokenizer
# import os
import torch
import numpy as np
# from sklearn.model_selection import train_test_split
from collections import defaultdict
import pandas as pd
import json
from collections import defaultdict
from datasets import Dataset, DatasetDict
import joblib




def preprocess_data(datapath, tokenizer, binarizer):
    source = []
    labels = []

    with open(datapath, 'r') as f:
        for line in f.readlines():
            line = json.loads(line)
            source.append(tokenizer.encode(line['token'].strip().lower(), truncation=True))
            labels.append(line['label'])
    
    # Use the binarizer directly — consistent with how train/test split was originally encoded
    one_hot_labels = binarizer.transform(labels).tolist()

    
    # --- Create HuggingFace Dataset ---
    full_dataset = Dataset.from_dict({
        "input_ids": source,          # list of token id lists (variable length)
        "labels": one_hot_labels,     # list of one-hot int lists
    })

    return full_dataset

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    binarizer = joblib.load('/Users/fdp54928/Documents/GitHub Repositories/contrastive-htc/data/panet/binarizer.pkl')

    # panet_dict => mapping from label id to label name
    with open('/Users/fdp54928/Documents/GitHub Repositories/contrastive-htc/data/panet/iri_to_label_dict.json') as f:
        panet_dict = json.load(f)

    # subclass_map => parent to children mapping
    path = '/Users/fdp54928/Documents/GitHub Repositories/contrastive-htc/data/panet/'
    subclass_map = {}
    with open(f"{path}panet.taxonomy", 'r', encoding='utf-8') as f:
        for line in f:
            # Strip the newline and split by tabs
            parts = line.strip().split('\t')
            if parts:
                parent = parts[0]
                # The rest of the elements (if any) are the children
                children = parts[1:]
                subclass_map[parent] = children

    train_dataset = preprocess_data('/Users/fdp54928/Documents/GitHub Repositories/contrastive-htc/data/panet/train_raw.json', tokenizer, binarizer)
    test_dataset = preprocess_data('/Users/fdp54928/Documents/GitHub Repositories/contrastive-htc/data/panet/test_raw.json', tokenizer, binarizer)
    eval_dataset = preprocess_data('/Users/fdp54928/Documents/GitHub Repositories/contrastive-htc/data/panet/eval_raw.json', tokenizer, binarizer)
    full_dataset = preprocess_data('/Users/fdp54928/Documents/GitHub Repositories/contrastive-htc/data/panet/full_data_raw.json', tokenizer, binarizer)

    
    value_dict = {i: tokenizer.encode(panet_dict[v].lower(), add_special_tokens=False)
            for i, v in enumerate(binarizer.classes_)}
    
    panet_idx_dict = {v: i for i, v in enumerate(binarizer.classes_)}
    hiera = defaultdict(set)

    for i, v in enumerate(binarizer.classes_):
        for child in subclass_map[v]:
            if child in panet_idx_dict:
                hiera[i].add(panet_idx_dict[child])

    
    torch.save(value_dict, 'processed_data/bert_value_dict.pt')
    torch.save(hiera, 'processed_data/slot.pt')
    
    
    train, test, val = [], [], []

    train = list(range(len(train_dataset)))
    val = list(range(len(train), len(eval_dataset)+len(train)))
    test = list(range(len(train)+len(val), len(test_dataset)+len(train)+len(val)))

    torch.save({'train': train, 'val': val, 'test': test}, 'processed_data/split.pt')


    full_dataset.save_to_disk('/Users/fdp54928/Library/CloudStorage/OneDrive-Nexus365/GitHub Repositories/contrastive-htc/data/panet/processed_data')

    # dataset = DatasetDict({
    #     "train": train_dataset,
    #     "test": test_dataset,
    #     "eval": eval_dataset,
    # })



    

