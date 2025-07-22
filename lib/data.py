# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets

    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process c4 dataset
# def get_c4(nsamples, seed, seqlen, tokenizer):
#     # Load train and validation datasets
#     traindata = load_dataset('json', data_files='en/c4-train.00000-of-01024.json.gz', split='train')
#     valdata = load_dataset('json', data_files='en/c4-validation.00000-of-00008.json.gz', split='train')
    
#     # Generate samples from training set
#     random.seed(seed)
#     trainloader = []
#     for _ in range(nsamples):
#         while True:
#             i = random.randint(0, len(traindata) - 1)
#             trainenc = tokenizer(traindata[i]['text'], return_tensors='pt', truncation=True, max_length=seqlen)
#             if trainenc.input_ids.shape[1] > seqlen:
#                 break
#         i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
#         j = i + seqlen
#         inp = trainenc.input_ids[:, i:j]
#         tar = inp.clone()
#         tar[:, :-1] = -100
#         trainloader.append((inp, tar))




#     # Prepare validation dataset
#     valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt', truncation=True, max_length=seqlen)


#     valenc = valenc.input_ids[:, :(256 * seqlen)]
#     valenc = TokenizerWrapper(valenc)
#     return trainloader, valenc


# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    print("[INFO] Loading training and validation datasets...")
    traindata = load_dataset('json', data_files='en/c4-train.00000-of-01024.json.gz', split='train')
    valdata = load_dataset('json', data_files='en/c4-validation.00000-of-00008.json.gz', split='train')
    print("[INFO] Datasets loaded successfully.")
    
    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    print(f"[INFO] Starting to generate {nsamples} training samples...")
    
    for sample_idx in range(nsamples):
        print(f"[DEBUG] Generating sample {sample_idx + 1}/{nsamples}")
        while True:
            i = random.randint(0, len(traindata) - 1)
            print(f"[DEBUG] Tokenizing sample from index {i}")
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            print(f"[DEBUG] Tokenized input shape: {trainenc.input_ids.shape}")
            if trainenc.input_ids.shape[1] > seqlen:
                print("[DEBUG] Sequence is long enough, proceeding to slicing.")
                break
            else:
                print("[DEBUG] Sequence too short, retrying...")

        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        print(f"[DEBUG] Sample {sample_idx + 1} added. inp shape: {inp.shape}, tar shape: {tar.shape}")

    print("[INFO] All training samples generated successfully.")
    
    print("[INFO] Preparing validation dataset...")
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt', truncation=True, max_length=seqlen)
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    print(f"[INFO] Validation dataset shape: {valenc.data.shape}")
    
    return trainloader, valenc


# Load and process ptb dataset
def get_ptb(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc
    

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    if "ptb" in name:
        return get_ptb(nsamples, seed, seqlen, tokenizer)
