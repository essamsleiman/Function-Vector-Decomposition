import pandas as pd
# import matplotlib.pyplot as plt
from transformers import pipeline
from numpy.linalg import norm
import random

random.seed(10)

import os, re, json
import torch, numpy as np

import sys
# sys.path.append('../')
torch.set_grad_enabled(False)

from src.utils.extract_utils import get_mean_head_activations, compute_universal_function_vector
from src.utils.intervention_utils import fv_intervention_natural_text, function_vector_intervention
from src.utils.model_utils import load_gpt_model_and_tokenizer
from src.utils.prompt_utils import load_dataset, word_pairs_to_prompt_data, create_prompt
from src.utils.eval_utils import decode_to_vocab, sentence_eval

# model_name = 'gpt2-xl'


dataset = load_dataset('professor_teacher', seed=0)

# Sample ICL example pairs, and a test word
dataset = load_dataset('professor_teacher')
word_pairs = dataset['train'][:5]
test_pair = dataset['test'][15]

print("test_pair: ======-----", test_pair['output'].strip())
prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=test_pair, prepend_bos_token=True)
sentence = create_prompt(prompt_data)
print("ICL prompt:\n", repr(sentence), '\n\n')

zeroshot_prompt_data = word_pairs_to_prompt_data({'input':[], 'output':[]}, query_target_pair=test_pair, prepend_bos_token=True, shuffle_labels=True)
zeroshot_sentence = create_prompt(zeroshot_prompt_data)
print("Zero-Shot Prompt:\n", repr(zeroshot_sentence))

# Check model's ICL answer

print("Input Sentence:", repr(sentence), '\n')
print(f"Input Query: {repr(test_pair['input'])}, Target: {repr(test_pair['output'])}\n")


# unbias_dataset = load_dataset('unbiased_dataset', seed=0)
unbias_dataset = load_dataset('gender_bias', seed=0)
