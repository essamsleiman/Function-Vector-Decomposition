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
from src.utils.eval_utils import decode_to_vocab, sentence_eval, get_probability

model_name = 'gpt2-xl'
# model_name = 'meta-llama/llama-2-7b-hf'
model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name)

def get_bias(inputs):
    # male_bias = inputs[0] / sum(inputs)
    # female_bias = inputs[1] / sum(inputs)
    male_bias = abs(.5 - male_bias)
    female_bias = abs(.5 - female_bias)
    
    return male_bias, female_bias
    
def avg(l):
    return sum(l) / len(l)

def get_score(score, test_sample):
    top1=[]
    top5=[]
    top10=[]
    k_found = 100
    ground_truth =  test_sample.strip().lower()
    for k in range(len(score)):
        # print("score: ", score)
        # print("test_sample: ", test_sample)
        # print("test_sample['output']: ", test_sample['output'])
        # print("score[k]: ", score[k])
        prediction = score[k][0].strip().lower()
        if prediction == ground_truth:
              k_found = k
    if k_found == 0:
        top1.append(1)
        top5.append(1)
        top10.append(1)
    elif 1 <= k_found <=4:
        top1.append(0)
        top5.append(1)
        top10.append(1)
    elif 1<= k_found <=9:
        top1.append(0)
        top5.append(0)
        top10.append(1)
    else:
        top1.append(0)
        top5.append(0)
        top10.append(0)
    
    top_1_accuracy = sum(top1) / len(top1)
    top_5_accuracy = sum(top5) / len(top5)
    top_10_accuracy = sum(top10) / len(top10)
    return top_1_accuracy, top_5_accuracy, top_10_accuracy

dataset = load_dataset('professor_teacher', seed=0)
mean_activations = get_mean_head_activations(dataset, model, model_config, tokenizer)
FV, top_heads = compute_universal_function_vector(mean_activations, model, model_config, n_top_heads=10)

# unbias_dataset = load_dataset('gender_bias', seed=0)
unbias_dataset = load_dataset('gender_unbias', seed=0)
mean_activations_unbias = get_mean_head_activations(unbias_dataset, model, model_config, tokenizer)
FV_unbiased, top_heads_unbiased = compute_universal_function_vector(mean_activations_unbias, model, model_config, n_top_heads=10)



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

for n_top_heads in range(10,11):


    # Sample ICL example pairs, and a test word
    # dataset = load_dataset('professor_teacher')
    word_pairs = dataset['train'][:10]
    avg_score_icl = []
    avg_score_icl_fv = []
    avg_score_fv = []
    avg_score_icl_fv_plus_fvunb = []
    avg_score_icl_fv_minus_fvunb = []
    avg_score_icl_5 = []
    avg_score_icl_fv_5 = []
    avg_score_fv_5 = []
    avg_score_icl_fv_plus_fvunb_5 = []
    avg_score_icl_fv_minus_fvunb_5 = []
    avg_score_icl_10 = []
    avg_score_icl_fv_10 = []
    avg_score_fv_10 = []
    avg_score_icl_fv_plus_fvunb_10 = []
    avg_score_icl_fv_minus_fvunb_10 = []
    for test_pair_idx in range(15, 100):
        test_pair = dataset['test'][test_pair_idx]

        prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=test_pair, prepend_bos_token=True)
        sentence = create_prompt(prompt_data)

        zeroshot_prompt_data = word_pairs_to_prompt_data({'input':[], 'output':[]}, query_target_pair=test_pair, prepend_bos_token=True, shuffle_labels=True)
        zeroshot_sentence = create_prompt(zeroshot_prompt_data)

        # Check model's ICL answer
        clean_logits = sentence_eval(sentence, [test_pair['output']], model, tokenizer, compute_nll=False)
        zero_shot_logits = sentence_eval(zeroshot_sentence, [test_pair['output']], model, tokenizer, compute_nll=False)


        # unbias_dataset = load_dataset('unbiased_dataset', seed=0)

        # Intervention on the zero-shot prompt

        for i in range(21, 22):
        # i = 21
            _, interv_logits_fv = function_vector_intervention(zeroshot_sentence, [test_pair['output']], i, FV, model, model_config, tokenizer)
            _, interv_logits_icl = function_vector_intervention(sentence, [test_pair['output']], i, FV, model, model_config, tokenizer)
            _, interv_logits_both_minus = function_vector_intervention(sentence, [test_pair['output']], i, FV- (FV_unbiased), model, model_config, tokenizer)
            _, interv_logits_both_plus = function_vector_intervention(sentence, [test_pair['output']], i, FV+ (FV_unbiased), model, model_config, tokenizer)


            score_icl = decode_to_vocab(clean_logits, tokenizer, k=15)
            score_fv = decode_to_vocab(interv_logits_fv, tokenizer, k=15)
            score_icl_fv = decode_to_vocab(interv_logits_icl, tokenizer, k=15)
            score_icl_fv_minus_fvunb = decode_to_vocab(interv_logits_both_minus, tokenizer, k=15)
            score_icl_fv_plus_fvunb = decode_to_vocab(interv_logits_both_plus, tokenizer, k=15)
            
            prob_gender_zero_shot = [get_probability(zero_shot_logits, tokenizer, "male"),  get_probability(zero_shot_logits, tokenizer, "female")]
            prob_gender_icl = [get_probability(clean_logits, tokenizer, "male"),  get_probability(clean_logits, tokenizer, "female")]
            prob_gender_fv = [get_probability(interv_logits_fv, tokenizer, "male"), get_probability(interv_logits_fv, tokenizer, "female")]
            prob_icl_fv = [get_probability(interv_logits_icl, tokenizer, "male"), get_probability(interv_logits_icl, tokenizer, "female")]
            prob_gender_icl_fv_minus_fvunb = [get_probability(interv_logits_both_minus, tokenizer, "male"), get_probability(interv_logits_both_minus, tokenizer, "female")]
            prob_gender_icl_fv_plus_fvunb = [get_probability(interv_logits_both_plus, tokenizer, "male"), get_probability(interv_logits_both_plus, tokenizer, "female"),]
            prob_gender_icl_fv_plus_fvunb = [get_probability(interv_logits_both_plus, tokenizer, "male"), get_probability(interv_logits_both_plus, tokenizer, "female"),]
            
            
            top1_icl, top5_icl, top10_icl = get_score(score_icl, test_pair['output'])
            top1_fv, top5_fv, top10_fv = get_score(score_fv, test_pair['output'])
            top1_icl_fv, top5_icl_fv, top10_icl_fv = get_score(score_icl_fv, test_pair['output'])
            top1_icl_fv_minus_fvun, top5_icl_fv_minus_fvun, top10_icl_fv_minus_fvun = get_score(score_icl_fv_minus_fvunb, test_pair['output'])
            top1_icl_fv_plus_fvunb, top5_icl_fv_plus_fvunb, top10_icl_fv_plus_fvunb = get_score(score_icl_fv_plus_fvunb, test_pair['output'])
            

            
            avg_score_icl.append(top1_icl)
            avg_score_fv.append(top1_fv)
            avg_score_icl_fv.append(top1_icl_fv)
            avg_score_icl_fv_plus_fvunb.append(top1_icl_fv_plus_fvunb)
            avg_score_icl_fv_minus_fvunb.append(top1_icl_fv_minus_fvun)
            
            avg_score_icl_5.append(top5_icl)
            avg_score_fv_5.append(top5_fv)
            avg_score_icl_fv_5.append(top5_icl_fv)
            avg_score_icl_fv_plus_fvunb_5.append(top5_icl_fv_plus_fvunb)
            avg_score_icl_fv_minus_fvunb_5.append(top5_icl_fv_minus_fvun)
            
            avg_score_icl_10.append(top10_icl)
            avg_score_fv_10.append(top10_fv)
            avg_score_icl_fv_10.append(top10_icl_fv)
            avg_score_icl_fv_plus_fvunb_10.append(top10_icl_fv_plus_fvunb)
            avg_score_icl_fv_minus_fvunb_10.append(top10_icl_fv_minus_fvun)

            print("======")
            print("avg_score_icl: ", sum(avg_score_icl) / len(avg_score_icl))
            print("avg_score_fv: ", sum(avg_score_fv) / len(avg_score_fv))
            print("avg_score_icl_fv: ", sum(avg_score_icl_fv) / len(avg_score_icl_fv))
            print("avg_score_icl_fv_plus_fvunb: ", sum(avg_score_icl_fv_plus_fvunb) / len(avg_score_icl_fv_plus_fvunb))
            print("avg_score_icl_fv_minus_fvunb: ", sum(avg_score_icl_fv_minus_fvunb) / len(avg_score_icl_fv_minus_fvunb))
            print("======")
            print("avg_score_icl_5: ", sum(avg_score_icl_5) / len(avg_score_icl))
            print("avg_score_fv_5: ", sum(avg_score_fv_5) / len(avg_score_fv))
            print("avg_score_icl_fv_5: ", sum(avg_score_icl_fv_5) / len(avg_score_icl_fv))
            print("avg_score_icl_fv_plus_fvunb_5: ", sum(avg_score_icl_fv_plus_fvunb_5) / len(avg_score_icl_fv_plus_fvunb))
            print("avg_score_icl_fv_minus_fvunb_5: ", sum(avg_score_icl_fv_minus_fvunb_5) / len(avg_score_icl_fv_minus_fvunb))
            print("======")
            print("avg_score_icl_10: ", sum(avg_score_icl_10) / len(avg_score_icl))
            print("avg_score_fv_10: ", sum(avg_score_fv_10) / len(avg_score_fv))
            print("avg_score_icl_fv_10: ", sum(avg_score_icl_fv_10) / len(avg_score_icl_fv))
            print("avg_score_icl_fv_plus_fvunb_10: ", sum(avg_score_icl_fv_plus_fvunb_10) / len(avg_score_icl_fv_plus_fvunb))
            print("avg_score_icl_fv_minus_fvunb_10: ", sum(avg_score_icl_fv_minus_fvunb_10) / len(avg_score_icl_fv_minus_fvunb))
            print("=======================================")
            
            
            
            prob_gender_zero_shot_bias_male, prob_gender_zero_shot_bias_female = get_bias(prob_gender_zero_shot)
            prob_gender_icl_bias_male, prob_gender_icl_bias_female = get_bias(prob_gender_icl)
            prob_gender_fv_bias_male, prob_gender_fv_bias_female = get_bias(prob_gender_fv)
            prob_icl_fv_bias_male, prob_icl_fv_bias_female = get_bias(prob_icl_fv)
            prob_gender_icl_fv_minus_fvunb_bias_male, prob_gender_icl_fv_minus_fvunb_bias_female = get_bias(prob_gender_icl_fv_minus_fvunb)
            prob_gender_icl_fv_plus_fvunb_bias_male, prob_gender_icl_fv_plus_fvunb_bias_female = get_bias(prob_gender_icl_fv_plus_fvunb)
            
            
            total_prob_gender_zero_shot_bias_male.append(prob_gender_zero_shot_bias_male)
            total_prob_gender_zero_shot_bias_female.append(prob_gender_zero_shot_bias_female)
            total_prob_gender_icl_bias_male.append(prob_gender_icl_bias_male)
            total_prob_gender_icl_bias_female.append(prob_gender_icl_bias_female)
            total_prob_gender_fv_bias_male.append(prob_gender_fv_bias_male)
            total_prob_gender_fv_bias_female.append(prob_gender_fv_bias_female)
            total_prob_icl_fv_bias_male.append(prob_icl_fv_bias_male)
            total_prob_icl_fv_bias_female.append(prob_icl_fv_bias_female)
            total_prob_gender_icl_fv_minus_fvunb_bias_male.append(prob_gender_icl_fv_minus_fvunb_bias_male)
            total_prob_gender_icl_fv_minus_fvunb_bias_female.append(prob_gender_icl_fv_minus_fvunb_bias_female)
            total_prob_gender_icl_fv_plus_fvunb_bias_male.append(prob_gender_icl_fv_plus_fvunb_bias_male)
            total_prob_gender_icl_fv_plus_fvunb_bias_female.append(prob_gender_icl_fv_plus_fvunb_bias_female)
            
            
            print("prob_gender_icl: ", get_bias(prob_gender_icl))
            print("prob_gender_fv: ", get_bias(prob_gender_fv))
            print("prob_icl_fv: ", get_bias(prob_icl_fv))
            print("prob_gender_icl_fv_minus_fvunb: ", get_bias(prob_gender_icl_fv_minus_fvunb))
            print("prob_gender_icl_fv_plus_fvunb: ", get_bias(prob_gender_icl_fv_plus_fvunb))



    print("======")
    print("avg_score_icl: ", sum(avg_score_icl) / len(avg_score_icl))
    print("avg_score_fv: ", sum(avg_score_fv) / len(avg_score_fv))
    print("avg_score_icl_fv: ", sum(avg_score_icl_fv) / len(avg_score_icl_fv))
    print("avg_score_icl_fv_plus_fvunb: ", sum(avg_score_icl_fv_plus_fvunb) / len(avg_score_icl_fv_plus_fvunb))
    print("avg_score_icl_fv_minus_fvunb: ", sum(avg_score_icl_fv_minus_fvunb) / len(avg_score_icl_fv_minus_fvunb))
    print("======")
    print("avg_score_icl_5: ", sum(avg_score_icl_5) / len(avg_score_icl))
    print("avg_score_fv_5: ", sum(avg_score_fv_5) / len(avg_score_fv))
    print("avg_score_icl_fv_5: ", sum(avg_score_icl_fv_5) / len(avg_score_icl_fv))
    print("avg_score_icl_fv_plus_fvunb_5: ", sum(avg_score_icl_fv_plus_fvunb_5) / len(avg_score_icl_fv_plus_fvunb))
    print("avg_score_icl_fv_minus_fvunb_5: ", sum(avg_score_icl_fv_minus_fvunb_5) / len(avg_score_icl_fv_minus_fvunb))
    print("======")
    print("avg_score_icl_10: ", sum(avg_score_icl_10) / len(avg_score_icl))
    print("avg_score_fv_10: ", sum(avg_score_fv_10) / len(avg_score_fv))
    print("avg_score_icl_fv_10: ", sum(avg_score_icl_fv_10) / len(avg_score_icl_fv))
    print("avg_score_icl_fv_plus_fvunb_10: ", sum(avg_score_icl_fv_plus_fvunb_10) / len(avg_score_icl_fv_plus_fvunb))
    print("avg_score_icl_fv_minus_fvunb_10: ", sum(avg_score_icl_fv_minus_fvunb_10) / len(avg_score_icl_fv_minus_fvunb))
    print("=======================================")
            
    print("total_prob_gender_zero_shot_bias_male: ", avg(total_prob_gender_zero_shot_bias_male))
    print("total_prob_gender_zero_shot_bias_female: ", avg(total_prob_gender_zero_shot_bias_female))
    print("total_prob_gender_icl_bias_male: ", avg(total_prob_gender_icl_bias_male))
    print("total_prob_gender_icl_bias_female: ", avg(total_prob_gender_icl_bias_female))
    print("total_prob_gender_fv_bias_male: ", avg(total_prob_gender_fv_bias_male))
    print("total_prob_gender_fv_bias_female: ", avg(total_prob_gender_fv_bias_female))
    print("total_prob_icl_fv_bias_male: ", avg(total_prob_icl_fv_bias_male))
    print("total_prob_icl_fv_bias_female: ", avg(total_prob_icl_fv_bias_female))
    print("total_prob_gender_icl_fv_minus_fvunb_bias_male: ", avg(total_prob_gender_icl_fv_minus_fvunb_bias_male))
    print("total_prob_gender_icl_fv_minus_fvunb_bias_female: ", avg(total_prob_gender_icl_fv_minus_fvunb_bias_female))
    print("total_prob_gender_icl_fv_plus_fvunb_bias_male: ", avg(total_prob_gender_icl_fv_plus_fvunb_bias_male))
    print("total_prob_gender_icl_fv_plus_fvunb_bias_female: ", avg(total_prob_gender_icl_fv_plus_fvunb_bias_female))

