import pandas as pd
# import matplotlib.pyplot as plt
from transformers import pipeline
from numpy.linalg import norm
import random
import pickle

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
from src.utils.eval_utils import decode_to_vocab, sentence_eval, get_probability, is_highest_probability
from utils import compute_and_cache_FV, get_bias, get_score, avg
# model_name = 'gpt2-xl'
model_name = 'meta-llama/llama-2-7b-hf'
model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name)

profession_labels = ['accountant', 'architect', 'attorney', 'chiropractor', 'comedian', 'composer', 'dentist', 'dietitian', 'dj', 'filmmaker', 'interior_designer', 'journalist', 'model', 'nurse', 'painter', 'paralegal', 'pastor', 'personal_trainer', 'photographer', 'physician', 'poet', 'professor', 'psychologist', 'rapper', 'software_engineer', 'surgeon', 'teacher', 'yoga_teacher']

# Load Datasets
dataset = load_dataset('professor_teacher', seed=0)
female_dataset = load_dataset('female_vector', seed=0)
male_dataset = load_dataset('male_vector', seed=0)
profession_dataset = load_dataset('profession_prediction_1', seed=0)

# Build Function Vectors from datasets
FV, top_heads = compute_and_cache_FV(dataset, model, model_config, tokenizer, cache_filename='FV.pkl')
FV_female, top_heads_female = compute_and_cache_FV(female_dataset, model, model_config, tokenizer, cache_filename='FV_female.pkl')
FV_male, top_heads_male = compute_and_cache_FV(male_dataset, model, model_config, tokenizer, cache_filename='FV_male.pkl')
FV_profession, top_heads_profession = compute_and_cache_FV(profession_dataset, model, model_config, tokenizer, cache_filename='FV_profession.pkl')


total_prob_gender_zero_shot_bias_male = []
total_prob_gender_zero_shot_bias_female = []
total_prob_gender_icl_bias_male = []
total_prob_gender_icl_bias_female = []
total_prob_gender_fv_bias_male = []
total_prob_gender_fv_bias_female = []
total_prob_icl_fv_bias_male = []
total_prob_icl_fv_bias_female = []
total_prob_gender_icl_fv_minus_fvunb_bias_male = []
total_prob_gender_icl_fv_minus_fvunb_bias_female = []
total_prob_gender_icl_fv_plus_fvunb_bias_male = []
total_prob_gender_icl_fv_plus_fvunb_bias_female = []
total_prob_gender_fv_unbias_male = []
total_prob_gender_fv_unbias_female = []

total_prob_gender_fv_plus_fvunb_profession = []
total_prob_gender_zero_shot_plus_fvunb_profession = []

total_prob_gender_fv_plus_fvunb_profession_classify = []
total_prob_gender_zero_shot_plus_fvunb_profession_classify=[]

for n_top_heads in range(10,11):

    # Sample ICL example pairs, and a test word
    word_pairs = dataset['train'][:10]
    word_pairs_profession = profession_dataset['train'][:10]
    
    for test_pair_idx in range(15, 100):
        
        # test-set
        test_pair = dataset['test'][test_pair_idx]
        test_profession = profession_dataset['test'][test_pair_idx]
        test_profession['output'] = str(test_profession['output'])
        
        # Prompts
        prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=test_pair, prepend_bos_token=True)
        prompt_data_profession = word_pairs_to_prompt_data(word_pairs_profession, query_target_pair=test_profession, prepend_bos_token=True)
        zeroshot_prompt_data = word_pairs_to_prompt_data({'input':[], 'output':[]}, query_target_pair=test_pair, prepend_bos_token=True, shuffle_labels=True)
        zeroshot_prompt_data_profession = word_pairs_to_prompt_data({'input':[], 'output':[]}, query_target_pair=test_profession, prepend_bos_token=True, shuffle_labels=True)
        
        sentence = create_prompt(prompt_data)
        sentence_profession = create_prompt(prompt_data_profession)
        zeroshot_sentence = create_prompt(zeroshot_prompt_data)
        zeroshot_sentence_profession = create_prompt(zeroshot_prompt_data_profession)

        # No FV Logits
        clean_logits = sentence_eval(sentence, [test_pair['output']], model, tokenizer, compute_nll=False)
        zero_shot_logits = sentence_eval(zeroshot_sentence, [test_pair['output']], model, tokenizer, compute_nll=False)
        zero_shot_logits_profession = sentence_eval(zeroshot_sentence_profession, [test_profession['output']], model, tokenizer, compute_nll=False)

        # Intervention on the zero-shot prompt at layer i. Layer 21 was found to be the most optimal in the paper
        for i in range(21, 22):
            
            # Use FV as intervention on LLM query
            _, interv_logits_fv = function_vector_intervention(zeroshot_sentence, [test_pair['output']], i, FV, model, model_config, tokenizer)
            _, interv_logits_icl = function_vector_intervention(sentence, [test_pair['output']], i, FV, model, model_config, tokenizer)
            _, interv_logits_both_minus = function_vector_intervention(sentence, [test_pair['output']], i, FV- 1 * (FV_female - FV_male), model, model_config, tokenizer)
            _, interv_logits_both_plus = function_vector_intervention(sentence, [test_pair['output']], i, FV+ 1 * (FV_female - FV_male), model, model_config, tokenizer)
            _, interv_logits_fvunb = function_vector_intervention(sentence, [test_pair['output']], i, 1 * (FV_female - FV_male) , model, model_config, tokenizer)
            _, interv_logits_both_plus_profession = function_vector_intervention(zeroshot_sentence_profession, [test_profession['output']], i, FV+ 1 * (FV_female - FV_male), model, model_config, tokenizer)
            
            # Get token probability of output strings "male" and "female" after intervention
            prob_gender_zero_shot = [get_probability(zero_shot_logits, tokenizer, "male"),  get_probability(zero_shot_logits, tokenizer, "female")]
            prob_gender_icl = [get_probability(clean_logits, tokenizer, "male"),  get_probability(clean_logits, tokenizer, "female")]
            prob_gender_fv = [get_probability(interv_logits_fv, tokenizer, "male"), get_probability(interv_logits_fv, tokenizer, "female")]
            prob_gender_icl_fv = [get_probability(interv_logits_icl, tokenizer, "male"), get_probability(interv_logits_icl, tokenizer, "female")]
            prob_gender_icl_fv_minus_fvunb = [get_probability(interv_logits_both_minus, tokenizer, "male"), get_probability(interv_logits_both_minus, tokenizer, "female")]
            prob_gender_icl_fv_plus_fvunb = [get_probability(interv_logits_both_plus, tokenizer, "male"), get_probability(interv_logits_both_plus, tokenizer, "female"),]
            prob_gender_fvunb = [get_probability(interv_logits_fvunb, tokenizer, "male"), get_probability(interv_logits_fvunb, tokenizer, "female"),]
            
            # calculate the normalized probability of gender tokens
            prob_gender_zero_shot_bias_male, prob_gender_zero_shot_bias_female = get_bias(prob_gender_zero_shot)
            prob_gender_icl_bias_male, prob_gender_icl_bias_female = get_bias(prob_gender_icl)
            prob_gender_fv_bias_male, prob_gender_fv_bias_female = get_bias(prob_gender_fv)
            prob_icl_fv_bias_male, prob_icl_fv_bias_female = get_bias(prob_gender_icl_fv)
            prob_gender_icl_fv_minus_fvunb_bias_male, prob_gender_icl_fv_minus_fvunb_bias_female = get_bias(prob_gender_icl_fv_minus_fvunb)
            prob_gender_icl_fv_plus_fvunb_bias_male, prob_gender_icl_fv_plus_fvunb_bias_female = get_bias(prob_gender_icl_fv_plus_fvunb)
            prob_gender_fv_unbias_male, prob_gender_fv_unbias_female = get_bias(prob_gender_fvunb)
            
            # only store male prob since male bias = female bias bc of normalization
            total_prob_gender_zero_shot_bias_male.append(prob_gender_zero_shot_bias_male)
            total_prob_gender_icl_bias_male.append(prob_gender_icl_bias_male)
            total_prob_gender_fv_bias_male.append(prob_gender_fv_bias_male)
            total_prob_icl_fv_bias_male.append(prob_icl_fv_bias_male)
            total_prob_gender_icl_fv_minus_fvunb_bias_male.append(prob_gender_icl_fv_minus_fvunb_bias_male)
            total_prob_gender_icl_fv_plus_fvunb_bias_male.append(prob_gender_icl_fv_plus_fvunb_bias_male)
            total_prob_gender_fv_unbias_male.append(prob_gender_fv_unbias_male)
            
            # Get probability and classification performance of profession prediction
            profession_word = profession_labels[int(test_profession["output"])]
            
            prob_gender_zero_shot_profession = get_probability(zero_shot_logits_profession, tokenizer, profession_word)
            prob_gender_fv_plus_fvunb_profession = get_probability(interv_logits_both_plus_profession, tokenizer, profession_word)
            prob_gender_zero_shot_profession_classify = is_highest_probability(zero_shot_logits_profession, tokenizer, profession_word, profession_labels)
            prob_gender_fv_plus_fvunb_profession_classify = is_highest_probability(interv_logits_both_plus_profession, tokenizer, profession_word, profession_labels)
            
            # accumulate probability and classification performance of profession prediction
            total_prob_gender_zero_shot_plus_fvunb_profession.append(prob_gender_zero_shot_profession)
            total_prob_gender_fv_plus_fvunb_profession.append(prob_gender_fv_plus_fvunb_profession)
            total_prob_gender_zero_shot_plus_fvunb_profession_classify.append(prob_gender_zero_shot_profession_classify)
            total_prob_gender_fv_plus_fvunb_profession_classify.append(prob_gender_fv_plus_fvunb_profession_classify)
            
    # Gender Bias Calculation
    print("total_prob_gender_zero_shot_bias_male: ", avg(total_prob_gender_zero_shot_bias_male) * 100)
    print("total_prob_gender_icl_bias_male: ", avg(total_prob_gender_icl_bias_male) * 100)
    print("total_prob_gender_fv_bias_male: ", avg(total_prob_gender_fv_bias_male) * 100)
    print("total_prob_icl_fv_bias_male: ", avg(total_prob_icl_fv_bias_male) * 100)
    print("total_prob_gender_icl_fv_minus_fvunb_bias_male: ", avg(total_prob_gender_icl_fv_minus_fvunb_bias_male) * 100)
    print("total_prob_gender_icl_fv_plus_fvunb_bias_male: ", avg(total_prob_gender_icl_fv_plus_fvunb_bias_male) * 100)
    print("total_prob_gender_fv_unbias_male: ", avg(total_prob_gender_fv_unbias_male) * 100)
    
    # Profession classification Performance
    print("total_prob_gender_zero_shot_plus_fvunb_profession: ", avg(total_prob_gender_zero_shot_plus_fvunb_profession) * 100)
    print("total_prob_gender_fv_plus_fvunb_profession: ", avg(total_prob_gender_fv_plus_fvunb_profession) * 100)
    print("total_prob_gender_zero_shot_plus_fvunb_profession_classify: ", avg(total_prob_gender_zero_shot_plus_fvunb_profession_classify) * 100)
    print("total_prob_gender_fv_plus_fvunb_profession_classify: ", avg(total_prob_gender_fv_plus_fvunb_profession_classify) * 100)
    